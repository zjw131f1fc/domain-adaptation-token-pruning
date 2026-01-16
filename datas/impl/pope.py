"""POPE 数据集准备器。

结构与 MME 相同: 包含字段 image/question/answer/category, 使用父类 BasePreparer 提供的统一拆分与元信息构建流程。

加载源: HuggingFace datasets -> lmms-lab/POPE (split='test')。

使用方式:
  在 config.dataset_settings 中设置:
    name: 'vqa-pope'
    split: {...}
    (可选) field_map: {src: dst, ...}
    (可选) category_priority: {enable: True, values: [{split: mode}, ...]}

不做异常捕获, 直接抛出错误以便快速定位配置问题。
"""
from typing import List, Dict, Any
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore


class POPEDataset(BsesDataset):
    pass


class POPEPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = True  # 与 MME 相同: 存在类别字段
        # field_map 可由外部 config.dataset_settings['field_map'] 指定, 这里不主动设置默认映射

    def _load_all(self) -> List[Dict[str, Any]]:
        ds = load_dataset("lmms-lab/POPE", "default", split="test")
        out: List[Dict[str, Any]] = []
        for i in range(len(ds)):
            item = ds[i]
            # 与 MME 一致的标准化字段
            # 在问题末尾添加提示，并将答案首字母大写，与 MME 格式保持一致
            question = item["question"] + " Please answer yes or no."
            answer = item["answer"].capitalize()  # "yes" -> "Yes", "no" -> "No"
            out.append({
                "image": item["image"],
                "question": question,
                "answer": answer,
                "category": item["category"],
            })
        return out

    def get(self) -> Dict[str, Any]:
        samples = self._load_all()
        self.detect_category(samples)
        applied_map = self.apply_field_map(samples)
        splits, placeholder = self.split_from_single(samples)
        meta = self.build_meta(samples, splits, applied_map, placeholder)
        judge = self._build_judge(meta, splits) if meta['total'] > 0 else self._build_judge_placeholder(meta)
        bundle = {"splits": splits, "meta": meta, "judge": judge}
        if True:
            self.print_report(bundle)
        return bundle

    def print_report(self, prepared: Dict[str, Any]):
        meta = prepared["meta"]
        splits = prepared["splits"]
        logger = getattr(self.config, "logger", None)
        if logger is None:
            return
        self.base_report(meta)
        logger.info('[POPE] Presplit: False (单列表随机拆分)')
        if meta["has_category"]:
            total_cat: Dict[Any, int] = {}
            for ds in splits.values():
                for i in range(len(ds)):
                    c = ds[i]["category"]
                    if c in total_cat:
                        total_cat[c] += 1
                    else:
                        total_cat[c] = 1
            logger.info("[POPE] Global Category Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(total_cat.items(), key=lambda x: (-x[1], str(x[0])))))
            for name, ds in splits.items():
                cat_stat: Dict[Any, int] = {}
                for i in range(len(ds)):
                    c = ds[i]["category"]
                    if c in cat_stat:
                        cat_stat[c] += 1
                    else:
                        cat_stat[c] = 1
                logger.info(f"[POPE] Split '{name}' Categories: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_stat.items(), key=lambda x: (-x[1], str(x[0])))))

    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, BsesDataset]):
        def _normalize(s: Any) -> str:
            if s is None:
                return ''
            text = str(s).strip().lower()
            punct_table = str.maketrans({c: ' ' for c in "!?,.:;\"'`~()[]{}<>"})
            text = text.translate(punct_table)
            parts = [p for p in text.split() if p]
            return ' '.join(parts)
        def _judge(pred, ref, sample=None, split_name: str = 'test'):
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
                    if r_norm and r_norm in p_norm:
                        correct += 1
                return {"correct": correct, "total": total, "accuracy": (correct / total) if total > 0 else 0.0}
            p_norm = _normalize(pred)
            r_norm = _normalize(ref)
            is_correct = 1 if (r_norm and r_norm in p_norm) else 0
            return {"correct": is_correct, "total": 1, "accuracy": float(is_correct)}
        return _judge
