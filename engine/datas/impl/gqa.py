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
  - 使用宽松的子串匹配评估
  - 支持批量评估和单条评估
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
        judge = self._build_judge(meta, splits) if meta['total'] > 0 else self._build_judge_placeholder(meta)

        bundle = {'splits': splits, 'meta': meta, 'judge': judge}
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

    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, BsesDataset]):
        """构建 GQA 评估函数 - 使用宽松的子串匹配"""
        
        def _normalize(s: Any) -> str:
            if s is None:
                return ''
            text = str(s).strip().lower()
            punct_table = str.maketrans({c: ' ' for c in "!?,.:;\"'`~()[]{}<>"})
            text = text.translate(punct_table)
            parts = [p for p in text.split() if p]
            return ' '.join(parts)

        def _judge(pred, ref, sample=None, split_name: str = 'test'):
            # 批量评估
            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError("批量判定时 ref 也应为列表")
                total = len(pred)
                if len(ref) != total:
                    raise ValueError("pred/ref 长度不一致")
                
                correct = 0
                for p_raw, r_raw in zip(pred, ref):
                    p_norm = _normalize(p_raw)
                    r_norm = _normalize(r_raw)
                    # 宽松匹配：参考答案在预测答案中出现
                    if r_norm and r_norm in p_norm:
                        correct += 1
                
                return {"correct": correct, "total": total, "accuracy": (correct / total) if total > 0 else 0.0}
            
            # 单条评估
            p_norm = _normalize(pred)
            r_norm = _normalize(ref)
            is_correct = 1 if (r_norm and r_norm in p_norm) else 0
            return {"correct": is_correct, "total": 1, "accuracy": float(is_correct)}
        
        return _judge
