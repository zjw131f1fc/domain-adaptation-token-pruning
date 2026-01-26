"""GQA 数据集准备器。

目标:
  - 从 HuggingFace vikhyatk/gqa 加载 train_balanced 和 val_balanced
  - 将每个图片的 qa 列表展开为单独的样本
  - 无类别 (has_category = False)，与 POPE/MME 保持一致的接口模式
  - 使用预拆分模式：train 使用 train_balanced，test 使用 val_balanced

数据结构:
  - 原始数据: {"image": PIL.Image, "qa": [{"question": str, "answer": str, "fullAnswer": str}, ...]}
  - 展开后: {"image": PIL.Image, "question": str, "answer": str, "fullAnswer": str}

性能优化:
  - 延迟图像加载: 加载阶段只存储索引，访问时才加载图像
  - 大幅提升加载速度，减少内存占用

评估方式:
  - 使用 Simple Accuracy（逐条评估）
  - 词级精确匹配：答案作为完整词出现在预测中即为正确
"""

from typing import List, Dict, Any, Union
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore


class GQADataset(BsesDataset):
    """GQA 数据集，支持延迟图像加载"""

    def __init__(self, samples: List[Dict[str, Any]], hf_datasets: Dict[str, Any] = None):
        super().__init__(samples)
        # hf_datasets: {split_name: dataset} 映射
        self._hf_datasets = hf_datasets or {}

    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """延迟加载：返回样本时才加载图像"""
        # 处理切片操作
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.samples)))
            return [self[i] for i in indices]

        sample = self.samples[idx].copy()

        # 如果 image 字段是 (split, index) 元组，则延迟加载
        if isinstance(sample.get('image'), tuple) and self._hf_datasets:
            split_name, hf_idx = sample['image']
            if split_name in self._hf_datasets:
                sample['image'] = self._hf_datasets[split_name][hf_idx]['image']

        return sample


class GQAPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = False  # GQA 不使用类别字段
        # 不需要字段映射，直接使用 answer 字段
        # 保存 HuggingFace dataset 引用用于延迟加载
        self._hf_datasets: Dict[str, Any] = {}

    def _load_train(self) -> List[Dict[str, Any]]:
        """加载 train_balanced 数据并展开 qa 列表"""
        ds = load_dataset("vikhyatk/gqa", split="train_balanced")
        self._hf_datasets['train'] = ds  # 保存引用
        samples: List[Dict[str, Any]] = []

        for i in range(len(ds)):
            item = ds[i]
            qa_list = item["qa"]

            # 将每个 qa 项展开为单独的样本
            for qa_item in qa_list:
                samples.append({
                    "image": ('train', i),  # 存储 (split, index) 而非图像
                    "question": qa_item["question"],
                    "answer": qa_item["answer"],
                })

        return samples

    def _load_val_as_test(self) -> List[Dict[str, Any]]:
        """加载 val_balanced 数据作为测试集并展开 qa 列表"""
        ds = load_dataset("vikhyatk/gqa", split="val_balanced")
        self._hf_datasets['test'] = ds  # 保存引用
        samples: List[Dict[str, Any]] = []

        for i in range(len(ds)):
            item = ds[i]
            qa_list = item["qa"]

            # 将每个 qa 项展开为单独的样本
            for qa_item in qa_list:
                samples.append({
                    "image": ('test', i),  # 存储 (split, index) 而非图像
                    "question": qa_item["question"],
                    "answer": qa_item["answer"],
                })

        return samples

    def _load_presplits(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载预拆分的数据"""
        data: Dict[str, List[Dict[str, Any]]] = {}

        # train 使用 train_balanced
        data['train'] = self._load_train()

        # test 使用 val_balanced（如果有 test 配置）
        if 'test' in self.split_cfg:
            data['test'] = self._load_val_as_test()

        return data

    def get(self) -> Dict[str, Any]:
        presplits = self._load_presplits()

        # 所有原始样本总集合用于 meta 统计
        all_samples: List[Dict[str, Any]] = []
        for lst in presplits.values():
            all_samples.extend(lst)

        self.detect_category(all_samples)
        applied_map = self.apply_field_map(all_samples)
        base_splits, placeholder = self.split_from_presplits(presplits)
        # 转换为 GQADataset（支持延迟加载）
        splits: Dict[str, GQADataset] = {}
        for name, ds in base_splits.items():
            splits[name] = GQADataset(ds.samples, self._hf_datasets)
        meta = self.build_meta(all_samples, splits, applied_map, placeholder)

        judge = self._build_judge()

        bundle = {
            'splits': splits,
            'meta': meta,
            'judge': judge,
        }
        self.print_report(bundle)
        return bundle

    def print_report(self, prepared: Dict[str, Any]):
        meta = prepared['meta']
        logger = getattr(self.config, 'logger', None)
        if logger is None:
            return

        self.base_report(meta)
        logger.info('[GQA] Presplit: True (train使用train_balanced，test使用val_balanced)')
        logger.info(f"[GQA] Loaded Samples: {meta['total']}")

        # 统计每个 split 的样本数量
        for name, ds in meta['split_sizes'].items():
            logger.info(f"[GQA] Split '{name}': {ds} samples")

    def _build_judge(self):
        """构建 judge 函数 - 词级精确匹配

        匹配规则：
        - 单词答案：答案作为完整词出现在预测中
        - 多词答案：答案作为连续子串出现在预测中

        支持单样本和批量评估两种模式。
        """

        def _normalize(s: Any) -> str:
            if s is None:
                return ''
            text = str(s).strip().lower()
            punct_table = str.maketrans({c: ' ' for c in "!?,.:;\"'`~()[]{}<>"})
            text = text.translate(punct_table)
            parts = [p for p in text.split() if p]
            return ' '.join(parts)

        def _is_match(pred_norm: str, ref_norm: str) -> bool:
            """词级精确匹配：ref 作为完整词出现在 pred 中"""
            if not ref_norm:
                return False
            pred_words = set(pred_norm.split())
            ref_words = ref_norm.split()
            # 单词答案：检查是否在 pred 词集合中
            if len(ref_words) == 1:
                return ref_words[0] in pred_words
            # 多词答案：检查是否作为连续子序列出现
            return ref_norm in pred_norm

        def _judge(pred, ref, sample=None, split_name: str = 'test') -> Dict[str, Any]:
            """支持单样本和批量评估"""
            # 批量评估模式
            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError("批量评估时 ref 也应为列表")
                if len(pred) != len(ref):
                    raise ValueError(f"pred/ref 长度不一致: {len(pred)} vs {len(ref)}")

                correct = 0
                for p, r in zip(pred, ref):
                    pred_norm = _normalize(p)
                    ref_norm = _normalize(r)
                    if _is_match(pred_norm, ref_norm):
                        correct += 1

                total = len(pred)
                return {
                    'correct': correct,
                    'total': total,
                    'accuracy': correct / total if total > 0 else 0.0,
                }

            # 单样本评估模式
            pred_norm = _normalize(pred)
            ref_norm = _normalize(ref)
            is_correct = _is_match(pred_norm, ref_norm)

            return {
                'correct': 1 if is_correct else 0,
                'total': 1,
                'accuracy': 1.0 if is_correct else 0.0,
            }

        return _judge
