"""ScienceQA 数据集准备器

数据来源: HuggingFace datasets -> derek-thomas/ScienceQA

要点:
 1. answer 保留原始 0-based 索引 (整数)；同时提供 answer_letter 方便直接字母评测
 2. 图像字段使用原始 image
 3. 使用 task 作为 category; 原 task 字段仍保留
 4. skill 字段原样保留 (若为 list 合并为分号分隔字符串)

性能优化:
  - 延迟图像加载: 加载阶段只存储索引，访问时才加载图像
  - 大幅提升加载速度，减少内存占用

拆分: 直接利用 HF 官方 split (train / validation / test)。若配置里使用 'val' 作为键则映射到 'validation'。
支持 category_priority (需 enable: True) / 占位 split / all split 等 BasePreparer 机制。

judge: 支持以下预测格式之一：
    - 0-based 索引 (int)
    - 数字字符串 ("0","1", ...)
    - 字母 (A/B/C/...)
    - 完整选项文本 (大小写与首尾空白忽略)
内部统一对齐到索引比较。
"""
from typing import Dict, Any, List, Union
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore
from PIL import Image  # type: ignore


class ScienceQADataset(BsesDataset):
    """ScienceQA 数据集，支持延迟图像加载"""

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


class ScienceQAPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = True  # 使用 task 作为 category
        # 保存 HuggingFace dataset 引用用于延迟加载
        self._hf_datasets: Dict[str, Any] = {}

    def _load_split(self, split: str) -> tuple:
        """加载指定 split 的数据

        Returns:
            tuple: (有图样本列表, 无图样本列表)
        """
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        self._hf_datasets[split] = ds  # 保存引用
        out: List[Dict[str, Any]] = []
        no_image_samples: List[Dict[str, Any]] = []
        for i in range(len(ds)):
            item = ds[i]
            img = item['image']
            has_image = isinstance(img, Image.Image)
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
                'image': (split, i) if has_image else None,  # 无图样本存 None
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
            if has_image:
                out.append(sample)
            else:
                no_image_samples.append(sample)
        logger = getattr(self.config, 'logger', None)
        if logger is not None:
            logger.info(f"[ScienceQA] Split '{split}': {len(no_image_samples)} samples with no/invalid image; {len(out)} with image.")
        return out, no_image_samples

    def _load_presplits(self) -> tuple:
        """加载预拆分数据

        Returns:
            tuple: (有图样本字典, 无图样本字典)
        """
        pres: Dict[str, List[Dict[str, Any]]] = {}
        no_image_pres: Dict[str, List[Dict[str, Any]]] = {}
        requested = set(self.split_cfg.keys())
        # train
        if 'train' in requested:
            samples, no_image = self._load_split('train')
            pres['train'] = samples
            no_image_pres['train'] = no_image
        # validation (config 可用 'val' 或 'validation')
        if 'val' in requested or 'validation' in requested:
            samples, no_image = self._load_split('validation')
            pres['val'] = samples
            no_image_pres['val'] = no_image
        # test
        if 'test' in requested:
            samples, no_image = self._load_split('test')
            pres['test'] = samples
            no_image_pres['test'] = no_image
        return pres, no_image_pres

    def get(self) -> Dict[str, Any]:
        presplits, no_image_presplits = self._load_presplits()
        all_samples: List[Dict[str, Any]] = []
        for lst in presplits.values():
            all_samples.extend(lst)
        # 注意：延迟加载模式下，安全过滤需要调整
        # 由于此时 image 是元组而非 PIL.Image，跳过 safeguard 检查
        # 原始的 _load_split 已经做了过滤
        self.detect_category(all_samples)
        applied_map = self.apply_field_map(all_samples)
        base_splits, placeholder = self.split_from_presplits(presplits)
        # 转换为 ScienceQADataset（支持延迟加载）
        splits: Dict[str, ScienceQADataset] = {}
        for name, ds in base_splits.items():
            splits[name] = ScienceQADataset(ds.samples, self._hf_datasets)
        meta = self.build_meta(all_samples, splits, applied_map, placeholder)
        # 将无图样本按 split 存入 meta（供评估时使用）
        meta['no_image_samples'] = no_image_presplits
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