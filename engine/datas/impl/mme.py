"""MME 数据集准备器

目标: 子类只做最少工作:
  - 标记: 有类别 (has_category = True)
    - 使用父类 split_from_single 对单一列表进行拆分
  - 提供字段映射表 (例如 prompt -> question) 可由 config.dataset_settings['field_map'] 给出
  - 实现 get() 与 print_report()

父类 BasePreparer 提供:
  - 加载后的一致流程 (prepare): 类别检测 / 字段映射 / 自动切分
  - 基础报告辅助方法 base_report()

本文件只负责:
  - 加载 MME 原始数据 (使用 datasets.load_dataset)
  - 将其转换为标准字段结构
  - 调用父类准备流程并输出结果

性能优化:
  - 延迟图像加载: 加载阶段只存储索引，访问时才加载图像
  - 大幅提升加载速度，减少内存占用

评估方式:
  - 使用 MME 官方评分公式
  - 每张图有两个问题（一个 Yes，一个 No），通过 question_id 配对
  - acc = (TP + TN) / 任务问题数
  - acc_plus = 一张图两问均对数 / 图片数
  - score = acc × 100 + acc_plus × 100
  - 每个子任务最高 200 分
  - 必须使用 aggregate_judge 进行聚合评估，不支持逐条评估
"""

from typing import List, Dict, Any, Union
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore
import random


class MMEDataset(BsesDataset):
    """MME 数据集，支持延迟图像加载"""

    def __init__(self, samples: List[Dict[str, Any]], hf_dataset=None):
        super().__init__(samples)
        self._hf_dataset = hf_dataset

    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """延迟加载：返回样本时才加载图像"""
        # 处理切片操作
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.samples)))
            return [self[i] for i in indices]

        sample = self.samples[idx].copy()

        # 如果 image 字段是整数索引，则延迟加载
        if isinstance(sample.get('image'), int) and self._hf_dataset is not None:
            hf_idx = sample['image']
            sample['image'] = self._hf_dataset[hf_idx]['image']

        return sample


class MMEPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = True  # 明确声明具备类别
        self.field_map = {}  # 可由外部 config 指定, 默认空
        self._hf_dataset = None  # 保存 HuggingFace dataset 引用用于延迟加载

    def _load_all(self) -> List[Dict[str, Any]]:
        ds = load_dataset("lmms-lab/MME", split="test")
        self._hf_dataset = ds  # 保存引用
        out: List[Dict[str, Any]] = []
        for i in range(len(ds)):
            item = ds[i]
            out.append({
                "image": i,  # 存储索引而非图像（延迟加载）
                "question": item["question"],
                "answer": item["answer"],
                "category": item["category"],
                "question_id": item["question_id"],  # 用于配对同一张图的两个问题
            })
        random.shuffle(out)
        return out

    def get(self) -> Dict[str, Any]:
        samples = self._load_all()
        # 类别检测 (已设 True 但保持统一接口)
        self.detect_category(samples)
        # 字段映射
        applied_map = self.apply_field_map(samples)
        # 根据配置拆分 (单一列表)
        base_splits, placeholder = self.split_from_single(samples)
        # 转换为 MMEDataset（支持延迟加载）
        splits: Dict[str, MMEDataset] = {}
        for name, ds in base_splits.items():
            splits[name] = MMEDataset(ds.samples, self._hf_dataset)
        # 构建 meta
        meta = self.build_meta(samples, splits, applied_map, placeholder)

        # 标记需要聚合评估
        meta['requires_aggregate_eval'] = True

        judge = self._build_judge()
        aggregate_judge = self._build_aggregate_judge()

        bundle = {
            "splits": splits,
            "meta": meta,
            "judge": judge,
            "aggregate_judge": aggregate_judge
        }
        self.print_report(bundle)
        return bundle

    def print_report(self, prepared: Dict[str, Any]):
        meta = prepared["meta"]
        splits = prepared["splits"]
        logger = getattr(self.config, "logger", None)
        if logger is None:
            return
        self.base_report(meta)
        logger.info('[MME] Presplit: False (单列表随机拆分)')
        if meta["has_category"]:
            total_cat: Dict[Any, int] = {}
            for ds in splits.values():
                for i in range(len(ds)):
                    c = ds[i]["category"]
                    if c in total_cat:
                        total_cat[c] += 1
                    else:
                        total_cat[c] = 1
            logger.info("[MME] Global Category Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(total_cat.items(), key=lambda x: (-x[1], str(x[0])))))
            for name, ds in splits.items():
                cat_stat: Dict[Any, int] = {}
                for i in range(len(ds)):
                    c = ds[i]["category"]
                    if c in cat_stat:
                        cat_stat[c] += 1
                    else:
                        cat_stat[c] = 1
                logger.info(f"[MME] Split '{name}' Categories: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_stat.items(), key=lambda x: (-x[1], str(x[0])))))

    # ----- judge 构建 -----
    def _build_judge(self):
        """构建 judge 函数 - MME 不支持逐条评估，调用时报错"""

        def _judge(pred, ref, sample=None, split_name: str = 'test'):
            raise NotImplementedError(
                "MME 数据集需要使用聚合评估（aggregate_judge），不支持逐条评估。"
                "请在所有样本预测完成后调用 aggregate_judge(predictions, references, samples)。"
            )

        return _judge

    def _build_aggregate_judge(self):
        """构建 MME 聚合评估函数 - 使用官方评分公式

        MME 评分公式：
        1. 每张图有两个问题（通过 question_id 配对）
        2. acc = (TP + TN) / 任务问题数
        3. acc_plus = 一张图两问均对数 / 图片数
        4. score = acc × 100 + acc_plus × 100
        5. 每个子任务最高 200 分
        """
        import re

        def _normalize(s: Any) -> str:
            if s is None:
                return ''
            text = str(s).strip().lower()
            punct_table = str.maketrans({c: ' ' for c in "!?,.:;\"'`~()[]{}<>"})
            text = text.translate(punct_table)
            parts = [p for p in text.split() if p]
            return ' '.join(parts)

        def _parse_binary_answer(s: Any) -> str | None:
            norm = _normalize(s)
            if not norm:
                return None
            if norm in {"yes", "no"}:
                return norm

            first = norm.split()[0]
            if first in {"yes", "no"}:
                return first

            patterns = [
                r'\b(?:answer|response|prediction|output)(?:\s+is)?[:\s]+(yes|no)\b',
                r'\b(?:choose|pick|select)(?:\s+)?(yes|no)\b',
            ]
            for pattern in patterns:
                matches = list(re.finditer(pattern, norm, re.IGNORECASE))
                if matches:
                    return matches[-1].group(1).lower()
            return None

        def _is_match(pred_raw: Any, ref_raw: Any) -> bool:
            ref_norm = _normalize(ref_raw)
            if not ref_norm:
                return False

            ref_binary = _parse_binary_answer(ref_raw)
            if ref_binary is not None:
                return _parse_binary_answer(pred_raw) == ref_binary

            pred_norm = _normalize(pred_raw)
            return pred_norm == ref_norm

        def _aggregate_judge(
            predictions: List[str],
            references: List[str],
            samples: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """
            MME 聚合评估函数

            Args:
                predictions: 所有预测结果
                references: 所有参考答案
                samples: 所有原始样本（必须包含 category, answer, question_id 字段）

            Returns:
                包含 total_score、per_category 等指标的字典
            """
            if len(predictions) != len(references) or len(predictions) != len(samples):
                raise ValueError(
                    f"长度不一致: predictions={len(predictions)}, "
                    f"references={len(references)}, samples={len(samples)}"
                )

            # 按 category 和 question_id 分组
            # category_data[cat][qid] = [(pred, ref, is_correct), ...]
            category_data: Dict[str, Dict[str, List[bool]]] = {}

            total_correct = 0
            for pred, ref, sample in zip(predictions, references, samples):
                category = sample.get('category', 'unknown')
                question_id = sample.get('question_id', 'unknown')
                is_correct = _is_match(pred, ref)
                if is_correct:
                    total_correct += 1

                if category not in category_data:
                    category_data[category] = {}
                if question_id not in category_data[category]:
                    category_data[category][question_id] = []
                category_data[category][question_id].append(is_correct)

            # 计算每个 category 的得分
            per_category: Dict[str, Dict[str, Any]] = {}
            total_score = 0.0

            for category, qid_data in category_data.items():
                # 统计该 category 的问题数和图片数
                total_questions = 0
                correct_questions = 0
                total_images = 0
                both_correct_images = 0

                for qid, correct_list in qid_data.items():
                    total_questions += len(correct_list)
                    correct_questions += sum(correct_list)
                    total_images += 1
                    # 如果该 question_id 的所有问题都答对（通常是 2 个）
                    if all(correct_list):
                        both_correct_images += 1

                # acc = 普通准确率
                acc = correct_questions / total_questions if total_questions > 0 else 0.0
                # acc_plus = 双对率（图片级别）
                acc_plus = both_correct_images / total_images if total_images > 0 else 0.0
                # score = acc * 100 + acc_plus * 100，最高 200 分
                score = acc * 100 + acc_plus * 100

                per_category[category] = {
                    'acc': acc,
                    'acc_plus': acc_plus,
                    'score': score,
                    'total_questions': total_questions,
                    'correct_questions': correct_questions,
                    'total_images': total_images,
                    'both_correct_images': both_correct_images,
                }

                total_score += score

            # 普通准确率（作为参考）
            simple_accuracy = total_correct / len(predictions) if predictions else 0.0

            return {
                'total_score': total_score,
                'simple_accuracy': simple_accuracy,
                'num_categories': len(category_data),
                'total_samples': len(predictions),
                'total_correct': total_correct,
                'per_category': per_category,
            }

        return _aggregate_judge
