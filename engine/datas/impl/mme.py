"""MME 数据集准备器 (单选题)

仿照 MMBench 的实现风格，提供统一接口：
  get() -> { 'splits': {name: Dataset}, 'meta': meta_dict, 'judge': callable }

源数据: HuggingFace datasets -> "lmms-lab/MME"
  - 包含两个子集: default, default (实际上是同一个)
  - dev split: 有答案，用于本地评估
  - test split: 需要提交到官网评测

注意：当前实现使用 dev split 进行本地评估。
      最终评测请使用 test split 并提交到 MME 官网。

数据集字段: question, A, B, C, D, answer, category, image
  - A/B/C/D: 选项文本 (字符串)
  - question: 问题文本
  - answer: 正确答案选项字母 (A/B/C/D)
  - category: 题目类别
  - image: 图像对象

拆分策略: 与 MMBench 相同，按 config.dataset_settings['split'] 进行:
  - dev -> train (有答案，可本地评估)
  - test -> test (需官网提交)

性能优化:
  - 延迟图像加载: 加载阶段只存储索引，访问时才加载图像
  - 大幅提升加载速度，减少内存占用

judge 逻辑 (单选):
  - 单条: pred 归一化大写后与正确选项字母完全相同视为正确
  - 若 pred 给出的是选项完整文本，将尝试匹配 A/B/C/D 文本定位到其字母
  - 批量: 对齐 zip(pred, ref)，分别判定

归一化规则:
  - 去除首尾空白
  - 大写化字母
  - 仅保留首个非空 token (用于防止模型回答 "A. xxx")

无 try/except 包装，配置 / 数据异常直接抛出。
"""

from typing import List, Dict, Any, Union, Tuple
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore


class MMEDataset(BsesDataset):
    """MME 数据集，支持延迟图像加载"""

    def __init__(self, samples: List[Dict[str, Any]], hf_datasets: Dict[str, Any] = None):
        super().__init__(samples)
        # hf_datasets: {split: dataset} 映射
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
            split, hf_idx = sample['image']
            if split in self._hf_datasets:
                sample['image'] = self._hf_datasets[split][hf_idx]['image']

        return sample


class MMEPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        # 明确存在 category 字段
        self.has_category = True
        # 保存 HuggingFace dataset 引用用于延迟加载
        self._hf_datasets: Dict[str, Any] = {}

    def _load_split(self, split: str) -> List[Dict[str, Any]]:
        """加载指定 split 的数据

        Args:
            split: 'dev' 或 'test'

        Returns:
            样本列表
        """
        ds = load_dataset("lmms-lab/MME", split=split)
        self._hf_datasets[split] = ds  # 保存引用
        out: List[Dict[str, Any]] = []

        for i in range(len(ds)):
            item = ds[i]
            q_raw = item['question']
            opt_lines = [
                f"A. {item['A']}",
                f"B. {item['B']}",
                f"C. {item['C']}",
                f"D. {item['D']}",
            ]
            instr = "Choose the correct answer from A/B/C/D and output only one letter (A, B, C, or D)."
            full_q = f"{q_raw}\n" + "\n".join(opt_lines) + f"\n{instr}"
            out.append({
                'image': (split, i),  # 存储 (split, index) 而非图像
                'question': full_q,
                'A': item['A'],
                'B': item['B'],
                'C': item['C'],
                'D': item['D'],
                'answer': item['answer'],
                'category': item['category'],
                'raw_question': q_raw,
            })
        return out

    def _load_presplits(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载预拆分数据"""
        data: Dict[str, List[Dict[str, Any]]] = {}
        # dev -> train (有答案，可本地评估)
        data['train'] = self._load_split('dev')
        # test 若在 split 配置中出现则加载
        if 'test' in self.split_cfg:
            data['test'] = self._load_split('test')
        return data

    def get(self) -> Dict[str, Any]:
        presplits = self._load_presplits()
        # 汇总所有样本用于统计
        all_samples: List[Dict[str, Any]] = []
        for lst in presplits.values():
            all_samples.extend(lst)
        self.detect_category(all_samples)
        applied_map = self.apply_field_map(all_samples)
        base_splits, placeholder = self.split_from_presplits(presplits)
        # 转换为 MMEDataset（支持延迟加载）
        splits: Dict[str, MMEDataset] = {}
        for name, ds in base_splits.items():
            splits[name] = MMEDataset(ds.samples, self._hf_datasets)
        meta = self.build_meta(all_samples, splits, applied_map, placeholder)
        judge = self._build_judge(meta, splits) if meta['total'] > 0 else self._build_judge_placeholder(meta)
        bundle = {'splits': splits, 'meta': meta, 'judge': judge}
        if True:
            self.print_report(bundle)
        return bundle

    def print_report(self, prepared: Dict[str, Any]):
        meta = prepared['meta']
        splits = prepared['splits']
        logger = getattr(self.config, 'logger', None)
        if logger is None:
            return
        self.base_report(meta)
        logger.info('[MME] Presplit: True (dev/test)')
        logger.info('[MME] 注意: 当前使用 dev split 进行本地评估，最终评测请使用 test split 提交到官网')
        if meta['has_category']:
            total_cat: Dict[Any, int] = {}
            for ds in splits.values():
                for i in range(len(ds)):
                    c = ds[i]['category']
                    if c in total_cat:
                        total_cat[c] += 1
                    else:
                        total_cat[c] = 1
            logger.info("[MME] Global Category Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(total_cat.items(), key=lambda x: (-x[1], str(x[0])))))

    # ---- judge ----
    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, MMEDataset]):
        def _norm_option_key(s: str) -> str:
            t = str(s).strip().upper()
            if not t:
                return ''
            # 取第一个非分隔 token (防止模型输出 "A." / "A)" 等)
            for sep in ['.', ')', ':']:
                if sep in t:
                    t = t.split(sep, 1)[0].strip()
            # 若出现空格, 仅取首 token
            if ' ' in t:
                t = t.split()[0]
            return t

        def _map_full_text_to_letter(sample: Dict[str, Any], pred_text: str) -> str:
            """如果 pred_text 与某个选项文本(归一化后)匹配, 返回其字母; 否则返回原归一化结果"""
            pt = pred_text.strip().lower()
            cand_map = {}
            for k in ['A', 'B', 'C', 'D']:
                cand_map[k] = str(sample.get(k, '')).strip().lower()
            for letter, text in cand_map.items():
                if pt == text:
                    return letter
            return _norm_option_key(pred_text)

        def _judge(pred, ref, sample=None, split_name: str = 'test'):
            def _single(p_raw, r_raw, sample_item):
                letter_ref = _norm_option_key(r_raw)
                # 预测可能是字母或完整选项文本
                letter_pred = _norm_option_key(p_raw)
                # 若仍未直接是 A-D, 尝试匹配完整文本
                if letter_pred not in ('A', 'B', 'C', 'D') and sample_item is not None:
                    letter_pred = _map_full_text_to_letter(sample_item, str(p_raw))
                is_correct = 1 if letter_pred == letter_ref and letter_ref in ('A', 'B', 'C', 'D') else 0
                return is_correct

            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError('批量判定时 ref 也应为列表')
                total = len(pred)
                if len(ref) != total:
                    raise ValueError('pred/ref 长度不一致')
                correct = 0
                if sample is not None and isinstance(sample, list):
                    for p, r, smp in zip(pred, ref, sample):
                        correct += _single(p, r, smp)
                else:
                    for p, r in zip(pred, ref):
                        correct += _single(p, r, None)
                return {'correct': correct, 'total': total, 'accuracy': (correct / total) if total > 0 else 0.0}

            # 单条
            is_correct = _single(pred, ref, sample)
            return {'correct': is_correct, 'total': 1, 'accuracy': float(is_correct)}

        return _judge
