"""VizWiz 数据集准备器

目标:
  - 从 HuggingFace Multimodal-Fatima/VizWiz 加载
  - train 用于训练，validation 用于评估
  - 评估方式与 VQA v2 一致

数据结构:
  - image: PIL.Image
  - question: str
  - answers: List[str]
  - answer_type: str (可作为 category)

性能优化:
  - 延迟图像加载: 加载阶段只存储索引，访问时才加载图像
"""

from typing import List, Dict, Any, Union
from collections import Counter
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore


class VizWizDataset(BsesDataset):
    """VizWiz 数据集，支持延迟图像加载"""

    def __init__(self, samples: List[Dict[str, Any]], hf_datasets: Dict[str, Any] = None):
        super().__init__(samples)
        self._hf_datasets = hf_datasets or {}

    def __getitem__(self, idx: Union[int, slice]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """延迟加载：返回样本时才加载图像"""
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


class VizWizPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        ds_cfg = self.config.dataset_settings
        # 使用 answer_type 作为类别
        self.use_category = ds_cfg.get('use_category', True)
        self.has_category = self.use_category
        self._hf_datasets: Dict[str, Any] = {}

    def _get_most_common_answer(self, answers: List[str]) -> str:
        """从答案列表中选择最常见的答案"""
        if not answers:
            return ""
        counter = Counter(answers)
        return counter.most_common(1)[0][0]

    def _load_train(self) -> List[Dict[str, Any]]:
        """加载 train 数据"""
        ds = load_dataset("Multimodal-Fatima/VizWiz", split="train")
        self._hf_datasets['train'] = ds
        samples: List[Dict[str, Any]] = []

        for i in range(len(ds)):
            item = ds[i]
            answers = item['answers']
            answer_type = item.get('answer_type', 'unknown')

            # 选择最常见答案作为训练答案
            train_answer = self._get_most_common_answer(answers)

            # 过滤掉无效答案
            if not train_answer or train_answer.lower().strip() in ['unanswerable', '']:
                continue

            question = item['question']
            question_with_prompt = f"{question} Answer the question using a single word or phrase."

            sample = {
                'image': ('train', i),
                'question': question_with_prompt,
                'answers': answers,
                'answer': train_answer,
            }
            if self.use_category:
                sample['category'] = answer_type
            samples.append(sample)

        return samples

    def _load_val_as_test(self) -> List[Dict[str, Any]]:
        """加载 validation 数据作为测试集"""
        ds = load_dataset("Multimodal-Fatima/VizWiz", split="validation")
        self._hf_datasets['test'] = ds
        samples: List[Dict[str, Any]] = []

        for i in range(len(ds)):
            item = ds[i]
            answers = item['answers']
            answer_type = item.get('answer_type', 'unknown')

            train_answer = self._get_most_common_answer(answers)

            question = item['question']
            question_with_prompt = f"{question} Answer the question using a single word or phrase."

            sample = {
                'image': ('test', i),
                'question': question_with_prompt,
                'answers': answers,
                'answer': train_answer if train_answer else 'unanswerable',
            }
            if self.use_category:
                sample['category'] = answer_type
            samples.append(sample)

        return samples

    def _load_presplits(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载预拆分的数据"""
        data: Dict[str, List[Dict[str, Any]]] = {}
        data['train'] = self._load_train()
        if 'test' in self.split_cfg:
            data['test'] = self._load_val_as_test()
        return data

    def get(self) -> Dict[str, Any]:
        presplits = self._load_presplits()

        all_samples: List[Dict[str, Any]] = []
        for lst in presplits.values():
            all_samples.extend(lst)

        self.detect_category(all_samples)
        applied_map = self.apply_field_map(all_samples)
        base_splits, placeholder = self.split_from_presplits(presplits)

        splits: Dict[str, VizWizDataset] = {}
        for name, ds in base_splits.items():
            splits[name] = VizWizDataset(ds.samples, self._hf_datasets)

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
        splits = prepared['splits']
        logger = getattr(self.config, 'logger', None)
        if logger is None:
            return

        self.base_report(meta)
        logger.info('[VizWiz] Presplit: True (train训练，validation评估)')
        logger.info(f"[VizWiz] Loaded Samples: {meta['total']}")

        if self.use_category and meta['has_category']:
            cat_stat: Dict[Any, int] = {}
            for ds in splits.values():
                for i in range(len(ds)):
                    c = ds.samples[i].get('category', 'unknown')
                    cat_stat[c] = cat_stat.get(c, 0) + 1
            logger.info("[VizWiz] Category Distribution: " + ", ".join(
                f"{c}:{n}" for c, n in sorted(cat_stat.items(), key=lambda x: (-x[1], str(x[0])))
            ))

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
                if ans_norm == pred_norm or ans_norm in pred_norm or pred_norm in ans_norm:
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
                        correct += 1.0 if (p_norm == r_norm or r_norm in p_norm or p_norm in r_norm) else 0.0
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
                score = 1.0 if (pred_norm == ref_norm or ref_norm in pred_norm or pred_norm in ref_norm) else 0.0
                return {'correct': score, 'total': 1, 'accuracy': float(score)}

        return _judge
