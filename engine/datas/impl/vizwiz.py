"""VizWiz 数据集准备器

目标:
  - 从 HuggingFace lmms-lab/VizWiz-VQA 加载
  - 使用 val split，按单数据集方式划分 train/test
  - 评估方式与 VQA v2 一致

数据结构:
  - image: PIL.Image
  - question: str
  - answers: List[str]

性能优化:
  - 延迟图像加载: 加载阶段只存储索引，访问时才加载图像
"""

from typing import List, Dict, Any, Union
from collections import Counter
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset as hf_load_dataset  # type: ignore
from tqdm import tqdm


class VizWizDataset(BsesDataset):
    """VizWiz 数据集，支持延迟图像加载"""

    def __init__(self, samples: List[Dict[str, Any]], hf_dataset=None):
        super().__init__(samples)
        self._hf_dataset = hf_dataset

    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """延迟加载：返回样本时才加载图像"""
        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self.samples)))
            return [self[i] for i in indices]

        sample = self.samples[idx].copy()

        # 如果 image 字段是 int 索引，则延迟加载
        if isinstance(sample.get('image'), int) and self._hf_dataset is not None:
            hf_idx = sample['image']
            sample['image'] = self._hf_dataset[hf_idx]['image']

        return sample


class VizWizPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        ds_cfg = self.config.dataset_settings
        self.use_category = ds_cfg.get('use_category', False)
        self.has_category = self.use_category
        self._hf_dataset = None

    def _get_most_common_answer(self, answers: List[str]) -> str:
        """从答案列表中选择最常见的答案"""
        if not answers:
            return ""
        counter = Counter(answers)
        return counter.most_common(1)[0][0]

    def _load_all(self) -> List[Dict[str, Any]]:
        """从 val split 加载所有数据"""
        ds = hf_load_dataset("lmms-lab/VizWiz-VQA", split="val")
        self._hf_dataset = ds
        samples: List[Dict[str, Any]] = []

        for i in tqdm(range(len(ds)), desc="VizWiz val", dynamic_ncols=True):
            item = ds[i]
            answers = item['answers']

            # 选择最常见答案作为训练答案
            train_answer = self._get_most_common_answer(answers)

            # 过滤掉无效答案
            if not train_answer or train_answer.lower().strip() in ['unanswerable', '']:
                continue

            question = item['question']
            question_with_prompt = f"{question} Answer the question using a single word or phrase."

            sample = {
                'image': i,  # 存储索引，延迟加载
                'question': question_with_prompt,
                'answers': answers,
                'answer': train_answer,
            }
            samples.append(sample)

        return samples

    def get(self) -> Dict[str, Any]:
        all_samples = self._load_all()

        self.detect_category(all_samples)
        applied_map = self.apply_field_map(all_samples)
        splits, placeholder = self.split_from_single(all_samples)

        # 转换为 VizWizDataset（支持延迟加载）
        viz_splits: Dict[str, VizWizDataset] = {}
        for name, ds in splits.items():
            viz_splits[name] = VizWizDataset(ds.samples, self._hf_dataset)

        meta = self.build_meta(all_samples, splits, applied_map, placeholder)
        judge = self._build_judge()

        bundle = {
            'splits': viz_splits,
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
        logger.info('[VizWiz] Source: lmms-lab/VizWiz-VQA (val split)')
        logger.info(f"[VizWiz] Loaded Samples: {meta['total']}")

    def _build_judge(self):
        """构建 judge 函数 - 与 VQA v2 评估方式一致"""
        ARTICLES = {"a", "an", "the"}
        NUMBER_MAP = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
            "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
        }
        punct_table = str.maketrans({c: ' ' for c in "!?,.:;\"'`~()[]{}<>"})

        def _normalize(s: Any) -> str:
            text = str(s).strip().lower()
            text = text.translate(punct_table)
            tokens = [t for t in text.split() if t]
            cleaned: List[str] = []
            for tok in tokens:
                if tok in ARTICLES:
                    continue
                if tok in NUMBER_MAP:
                    cleaned.append(NUMBER_MAP[tok])
                else:
                    cleaned.append(tok)
            return ' '.join(cleaned)

        def _is_unanswerable(ref) -> bool:
            """检查 GT 是否为 unanswerable"""
            if isinstance(ref, list):
                return all(_normalize(ans) in ['unanswerable', ''] for ans in ref)
            return _normalize(ref) in ['unanswerable', '']

        def _official_score(pred_norm: str, ref_list: List[str]) -> float:
            if not pred_norm or pred_norm.strip() == "":
                return 0.0
            count = 0
            for ans in ref_list:
                ans_norm = _normalize(ans)
                if ans_norm == pred_norm:
                    count += 1
            score = count / 3.0
            return 1.0 if score >= 1.0 else score

        def _judge(pred, ref, sample=None, split_name: str = 'train'):
            # 批量评估
            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError("批量判定时 ref 也应为列表")
                total = len(pred)
                if len(ref) != total:
                    raise ValueError("pred/ref 长度不一致")
                correct = 0
                for p_raw, r_raw in zip(pred, ref):
                    if _is_unanswerable(r_raw):
                        correct += 1.0
                        continue
                    p_norm = _normalize(p_raw)
                    if not p_norm or p_norm.strip() == "":
                        continue
                    if isinstance(r_raw, list):
                        score = _official_score(p_norm, r_raw)
                        correct += score
                    else:
                        r_norm = _normalize(r_raw)
                        correct += 1.0 if p_norm == r_norm else 0.0
                return {"correct": correct, "total": total, "accuracy": (correct / total) if total > 0 else 0.0}

            # 单条评估
            if _is_unanswerable(ref):
                return {'correct': 1.0, 'total': 1, 'accuracy': 1.0}

            pred_norm = _normalize(pred)
            if not pred_norm or pred_norm.strip() == "":
                return {'correct': 0.0, 'total': 1, 'accuracy': 0.0}

            if isinstance(ref, list):
                score = _official_score(pred_norm, ref)
                return {'correct': score, 'total': 1, 'accuracy': float(score)}
            else:
                ref_norm = _normalize(ref)
                score = 1.0 if pred_norm == ref_norm else 0.0
                return {'correct': score, 'total': 1, 'accuracy': float(score)}

        return _judge
