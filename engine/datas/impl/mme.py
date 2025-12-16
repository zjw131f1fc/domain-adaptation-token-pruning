"""MME 数据集准备器 (进一步精简)

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
"""

from typing import List, Dict, Any
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore
import random


class MMEDataset(BsesDataset):
    pass


class MMEPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = True  # 明确声明具备类别
        self.field_map = {}  # 可由外部 config 指定, 默认空

    def _load_all(self) -> List[Dict[str, Any]]:
        ds = load_dataset("lmms-lab/MME", split="test")
        out: List[Dict[str, Any]] = []
        for i in range(len(ds)):
            item = ds[i]
            out.append({
                "image": item["image"],
                "question": item["question"],
                "answer": item["answer"],
                "category": item["category"],
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
        splits, placeholder = self.split_from_single(samples)
        # 构建 meta
        meta = self.build_meta(samples, splits, applied_map, placeholder)
        # judge
        judge = self._build_judge(meta, splits) if meta['total'] > 0 else self._build_judge_placeholder(meta)
        bundle = {"splits": splits, "meta": meta, "judge": judge}
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
    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, BsesDataset]):
        # 宽松评估：大小写忽略 + 去标点 + 多候选答案拆分 + 编辑距离相似度 + Jaccard 词集合相似度 + 子串匹配
        # 支持可选阈值: dataset_settings['lenient_edit_sim'], ['lenient_jaccard']

        def _normalize(s: Any) -> str:
            if s is None:
                return ''
            text = str(s).strip().lower()
            punct_table = str.maketrans({c: ' ' for c in "!?,.:;\"'`~()[]{}<>"})
            text = text.translate(punct_table)
            parts = [p for p in text.split() if p]
            return ' '.join(parts)
        def _judge(pred, ref, sample=None, split_name: str = 'test'):
            # 批量: ref_norm 必须是 p_norm 的子串
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
            # 单条: 只允许答案在模型输出中出现
            p_norm = _normalize(pred)
            r_norm = _normalize(ref)
            is_correct = 1 if (r_norm and r_norm in p_norm) else 0
            return {"correct": is_correct, "total": 1, "accuracy": float(is_correct)}
        return _judge

