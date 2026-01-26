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
  - 使用 Balanced Accuracy（按答案类别均衡计算）
  - 必须使用 aggregate_judge 进行聚合评估，不支持逐条评估
"""

from typing import List, Dict, Any, Union, Tuple
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

        # 标记需要聚合评估
        meta['requires_aggregate_eval'] = True

        judge = self._build_judge()
        aggregate_judge = self._build_aggregate_judge()

        bundle = {
            'splits': splits,
            'meta': meta,
            'judge': judge,
            'aggregate_judge': aggregate_judge
        }
        if True:
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
        """构建 judge 函数 - GQA 不支持逐条评估，调用时报错"""

        def _judge(pred, ref, sample=None, split_name: str = 'test'):
            raise NotImplementedError(
                "GQA 数据集需要使用聚合评估（aggregate_judge），不支持逐条评估。"
                "请在所有样本预测完成后调用 aggregate_judge(predictions, references, samples)。"
            )

        return _judge

    def _build_aggregate_judge(self):
        """构建 GQA 聚合评估函数 - 计算 Balanced Accuracy

        Balanced Accuracy 计算方式：
        1. 按答案值（answer）分组
        2. 对每个答案类别计算准确率：correct / total
        3. 对所有类别的准确率取平均

        这样可以避免高频答案（yes/no、常见颜色等）主导评估结果。
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

        def _aggregate_judge(
            predictions: List[str],
            references: List[str],
            samples: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """
            聚合评估函数

            Args:
                predictions: 所有预测结果
                references: 所有参考答案
                samples: 所有原始样本

            Returns:
                包含 balanced_accuracy 等指标的字典
            """
            if len(predictions) != len(references):
                raise ValueError(f"predictions ({len(predictions)}) 和 references ({len(references)}) 长度不一致")

            # 按答案值分组统计
            answer_groups: Dict[str, Dict[str, int]] = {}  # {answer: {'correct': n, 'total': n}}

            total_correct = 0
            for pred, ref in zip(predictions, references):
                ref_norm = _normalize(ref)
                pred_norm = _normalize(pred)

                if ref_norm not in answer_groups:
                    answer_groups[ref_norm] = {'correct': 0, 'total': 0}

                answer_groups[ref_norm]['total'] += 1

                if _is_match(pred_norm, ref_norm):
                    answer_groups[ref_norm]['correct'] += 1
                    total_correct += 1

            # 计算每个答案类别的准确率
            category_accs = []
            for answer_val, stats in answer_groups.items():
                if stats['total'] > 0:
                    acc = stats['correct'] / stats['total']
                    category_accs.append(acc)

            # Balanced Accuracy = 所有类别准确率的平均
            balanced_accuracy = sum(category_accs) / len(category_accs) if category_accs else 0.0

            # 普通准确率（作为参考）
            simple_accuracy = total_correct / len(predictions) if predictions else 0.0

            return {
                'balanced_accuracy': balanced_accuracy,
                'simple_accuracy': simple_accuracy,
                'num_answer_categories': len(answer_groups),
                'total_samples': len(predictions),
                'total_correct': total_correct,
            }

        return _aggregate_judge
