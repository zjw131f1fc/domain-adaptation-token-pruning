"""ScienceQA 数据集准备器

数据来源: HuggingFace datasets -> derek-thomas/ScienceQA

要点:
 1. answer 保留原始 0-based 索引 (整数)；同时提供 answer_letter 方便直接字母评测
 2. 图像字段使用原始 image
 3. 使用 task 作为 category; 原 task 字段仍保留
 4. skill 字段原样保留 (若为 list 合并为分号分隔字符串)

拆分: 直接利用 HF 官方 split (train / validation / test)。若配置里使用 'val' 作为键则映射到 'validation'。
支持 category_priority (需 enable: True) / 占位 split / all split 等 BasePreparer 机制。

judge: 支持以下预测格式之一：
    - 0-based 索引 (int)
    - 数字字符串 ("0","1", ...)
    - 字母 (A/B/C/...)
    - 完整选项文本 (大小写与首尾空白忽略)
内部统一对齐到索引比较。
"""
from typing import Dict, Any, List
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore
from PIL import Image  # type: ignore


class ScienceQADataset(BsesDataset):
    pass


class ScienceQAPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = True  # 使用 task 作为 category

    def _load_split(self, split: str) -> List[Dict[str, Any]]:
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        out: List[Dict[str, Any]] = []
        skipped_no_image = 0
        for i in range(len(ds)):
            item = ds[i]
            img = item['image']
            # 过滤掉 image 为空或类型异常的样本（评估阶段要求为 PIL.Image.Image）
            if not isinstance(img, Image.Image):
                skipped_no_image += 1
                continue
            question = item['question']
            choices = item['choices']
            ans_index = item['answer']  # 保留 0-based
            letter = chr(ord('A') + ans_index)
            opt_lines = []
            for idx, opt in enumerate(choices):
                opt_lines.append(f"({idx}) {opt}")
            total_opts = len(choices)
            instr = f"Only output the numeric index of the correct option (0~{total_opts-1}). Return only the number (do not output text)."
            skill_val = item['skill']
            if isinstance(skill_val, list):
                skill_norm = '; '.join(str(s) for s in skill_val)
            else:
                skill_norm = skill_val if skill_val is not None else ''
            task = item['task']
            header_lines = [f"Task: {task}"]
            if skill_norm:
                header_lines.append(f"Skill: {skill_norm}")
            header_block = "\n".join(header_lines)
            full_q = header_block + "\n" + question + "\n" + "\n".join(opt_lines) + f"\n{instr}"
            sample = {
                'image': img,
                'question': full_q,
                'raw_question': question,
                'choices': choices,
                'answer': ans_index,          # 0-based
                'answer_letter': letter,      # 冗余字母
                'category': task,
                'task': task,
                'hint': item['hint'],
                'lecture': item['lecture'],
                'solution': item['solution'],
                'skill': skill_norm,
            }
            out.append(sample)
        logger = getattr(self.config, 'logger', None)
        if logger is not None:
            logger.info(f"[ScienceQA] Split '{split}': filtered {skipped_no_image} samples with no/invalid image; kept {len(out)}.")
        return out

    def _load_presplits(self) -> Dict[str, List[Dict[str, Any]]]:
        pres: Dict[str, List[Dict[str, Any]]] = {}
        requested = set(self.split_cfg.keys())
        # train
        if 'train' in requested:
            pres['train'] = self._load_split('train')
        # validation (config 可用 'val' 或 'validation')
        if 'val' in requested or 'validation' in requested:
            pres['val'] = self._load_split('validation')
        # test
        if 'test' in requested:
            pres['test'] = self._load_split('test')
        return pres

    def get(self) -> Dict[str, Any]:
        presplits = self._load_presplits()
        all_samples: List[Dict[str, Any]] = []
        for lst in presplits.values():
            all_samples.extend(lst)
        # 额外安全过滤 (防止上游未来改动导致 None 泄漏)
        before_total = len(all_samples)
        all_samples = [s for s in all_samples if isinstance(s.get('image'), Image.Image)]
        after_total = len(all_samples)
        leak_filtered = before_total - after_total
        if leak_filtered > 0:
            logger = getattr(self.config, 'logger', None)
            if logger is not None:
                logger.warning(f"[ScienceQA] Post-load safeguard filtered {leak_filtered} leaked samples with invalid image; final total {after_total}.")
        self.detect_category(all_samples)
        applied_map = self.apply_field_map(all_samples)
        splits, placeholder = self.split_from_presplits(presplits)
        meta = self.build_meta(all_samples, splits, applied_map, placeholder)
        judge = self._build_judge(meta, splits) if meta['total'] > 0 else self._build_judge_placeholder(meta) # type: ignore
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
        loaded_names = ", ".join(sorted(splits.keys()))
        logger.info(f"[ScienceQA] Presplit Loaded: {loaded_names}")
        cat_stat: Dict[Any, int] = {}
        for ds in splits.values():
            for i in range(len(ds)):
                c = ds[i]['category']
                cat_stat[c] = cat_stat.get(c, 0) + 1
        logger.info("[ScienceQA] Global Task Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_stat.items(), key=lambda x: (-x[1], str(x[0])))))
        # 每个 split 内部任务分布
        for name, ds in splits.items():
            sub_stat: Dict[Any, int] = {}
            for i in range(len(ds)):
                c = ds[i]['category']
                sub_stat[c] = sub_stat.get(c, 0) + 1
            logger.info(f"[ScienceQA] Split '{name}' Task Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(sub_stat.items(), key=lambda x: (-x[1], str(x[0])))))

    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, ScienceQADataset]):
        def _parse(pred_raw: Any, sample: Dict[str, Any]) -> int:
            # 尝试: 纯数字 -> 在字符串中抓第一个数字 (Answer: 2) -> 完整选项文本匹配
            if isinstance(pred_raw, int):
                return pred_raw
            text = str(pred_raw).strip()
            # 仅允许 ASCII 数字 0-9；忽略其他 Unicode 数字 (如 ①②③)
            if text and all('0' <= ch <= '9' for ch in text):
                return int(text)
            # 抓取第一个连续 ASCII 数字序列
            num_buf = ''
            for ch in text:
                if '0' <= ch <= '9':
                    num_buf += ch
                elif num_buf:
                    break
            if num_buf and all('0' <= ch <= '9' for ch in num_buf):
                return int(num_buf)
            # 完整文本匹配
            choices = sample['choices'] if sample is not None and 'choices' in sample else []
            low = text.lower()
            for idx, opt in enumerate(choices):
                if low == str(opt).strip().lower():
                    return idx
            return -999
        def _judge(pred, ref, sample=None, split_name: str = 'val'):
            def _single(p_raw, r_raw, smp):
                ref_index = r_raw if isinstance(r_raw, int) else int(r_raw)
                pred_index = _parse(p_raw, smp)
                # print(f"[ScienceQA Judge] pred_raw: {p_raw} -> pred_index: {pred_index}; ref_index: {ref_index}")
                return 1 if pred_index == ref_index else 0
            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError('批量判定时 ref 也应为列表')
                if len(pred) != len(ref):
                    raise ValueError('pred/ref 长度不一致')
                correct = 0
                if sample is not None and isinstance(sample, list):
                    for p, r, smp in zip(pred, ref, sample):
                        correct += _single(p, r, smp)
                else:
                    for p, r in zip(pred, ref):
                        correct += _single(p, r, None)
                total = len(pred)
                return {'correct': correct, 'total': total, 'accuracy': (correct/total) if total > 0 else 0.0}
            is_correct = _single(pred, ref, sample)
            return {'correct': is_correct, 'total': 1, 'accuracy': float(is_correct)}
        return _judge