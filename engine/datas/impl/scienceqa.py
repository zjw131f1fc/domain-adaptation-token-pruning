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
from io import BytesIO
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore
from PIL import Image  # type: ignore


def _ensure_pil_image(img: Any) -> Image.Image | None:
    """确保图像是 PIL Image 格式

    处理 HuggingFace datasets 可能返回的多种格式：
    - PIL.Image: 直接返回
    - dict with 'bytes': 从 bytes 解码
    - dict with 'path': 从路径加载
    - None: 返回 None
    """
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, dict):
        if 'bytes' in img and img['bytes'] is not None:
            return Image.open(BytesIO(img['bytes']))
        if 'path' in img and img['path'] is not None:
            return Image.open(img['path'])
    return None


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
                raw_img = self._hf_datasets[split_name][hf_idx]['image']
                # 确保转换为 PIL Image（处理 dict 格式）
                pil_img = _ensure_pil_image(raw_img)
                # 预先 resize 到 384x384，避免大图导致处理缓慢
                if pil_img is not None:
                    max_size = 384
                    if max(pil_img.size) > max_size:
                        pil_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                sample['image'] = pil_img

        return sample


class ScienceQAPreparer(BasePreparer):
    # 序列长度阈值（超过此值的样本会被过滤）
    # 576 vision tokens + 文本 tokens，总长度不超过此值
    MAX_SEQ_LENGTH = 1000

    def __init__(self, config):
        super().__init__(config)
        self.has_category = True  # 使用 task 作为 category
        # 保存 HuggingFace dataset 引用用于延迟加载
        self._hf_datasets: Dict[str, Any] = {}
        # 加载 tokenizer 用于长度过滤
        self._tokenizer = None
        # 是否过滤超长样本（仅训练时过滤，评估时保留）
        self._filter_long_samples = config.dataset_settings.get('filter_long_samples', True)

    def _get_tokenizer(self):
        """延迟加载 tokenizer"""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            hf_cache = self.config.global_settings.get('hf_cache_dir', None)
            self._tokenizer = AutoTokenizer.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                cache_dir=hf_cache
            )
        return self._tokenizer

    def _estimate_seq_length(self, question: str, answer: str) -> int:
        """估算序列长度（576 vision tokens + 文本 tokens）"""
        tokenizer = self._get_tokenizer()
        eos = tokenizer.eos_token or "</s>"
        prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}{eos}"
        tokens = tokenizer(prompt, return_tensors="pt")
        return tokens['input_ids'].shape[1] + 576  # 加上 vision tokens

    def _load_split(self, split: str) -> tuple:
        """加载指定 split 的数据

        Returns:
            tuple: (有图样本列表, 无图样本列表)
        """
        ds = load_dataset("derek-thomas/ScienceQA", split=split)
        self._hf_datasets[split] = ds  # 保存引用
        out: List[Dict[str, Any]] = []
        no_image_samples: List[Dict[str, Any]] = []
        filtered_count = 0

        # 调试：统计图像类型
        image_type_stats: Dict[str, int] = {}
        sample_by_type: Dict[str, Any] = {}

        for i in range(len(ds)):
            item = ds[i]
            img = item['image']

            # 记录图像类型
            img_type = type(img).__name__
            image_type_stats[img_type] = image_type_stats.get(img_type, 0) + 1
            if img_type not in sample_by_type and img is not None:
                sample_by_type[img_type] = (i, img)

            # 检查是否有有效图像（支持 PIL Image 和 dict 格式）
            pil_img = _ensure_pil_image(img)
            has_image = pil_img is not None
            question = item['question']
            choices = item['choices']
            ans_index = item['answer']  # 保留 0-based
            letter = chr(ord('A') + ans_index)
            opt_lines = []
            for idx, opt in enumerate(choices):
                opt_lines.append(f"({chr(ord('A') + idx)}) {opt}")
            total_opts = len(choices)
            instr = f"Answer with the option letter (A, B, C, or D) at the end."
            skill_val = item['skill']
            if isinstance(skill_val, list):
                skill_norm = '; '.join(str(s) for s in skill_val)
            else:
                skill_norm = skill_val if skill_val is not None else ''
            task = item['task']
            hint = item.get('hint', '') or ''
            lecture = item.get('lecture', '') or ''

            # 构建 prompt：包含背景知识和提示
            prompt_parts = []

            # 背景知识（lecture）放在最前面
            if lecture.strip():
                prompt_parts.append(f"Background: {lecture.strip()}")

            # 提示信息
            if hint.strip():
                prompt_parts.append(f"Hint: {hint.strip()}")

            # 问题和选项
            prompt_parts.append(question)
            prompt_parts.append("\n".join(opt_lines))
            prompt_parts.append(instr)

            full_q = "\n".join(prompt_parts)

            # 过滤超长样本（仅对有图样本过滤，且仅当 filter_long_samples=True 时）
            if has_image and self._filter_long_samples:
                seq_len = self._estimate_seq_length(full_q, letter)
                if seq_len > self.MAX_SEQ_LENGTH:
                    filtered_count += 1
                    continue

            sample = {
                'image': (split, i) if has_image else None,  # 无图样本存 None
                'question': full_q,
                'raw_question': question,
                'choices': choices,
                'answer': letter,             # 字母形式 (A/B/C/D)，供训练 loss 使用
                'answer_index': ans_index,    # 0-based 索引，供评估 judge 使用
                'answer_letter': letter,      # 冗余字母（兼容）
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
            if filtered_count > 0:
                logger.info(f"[ScienceQA] Split '{split}': 过滤了 {filtered_count} 个超长样本 (>{self.MAX_SEQ_LENGTH} tokens)")
            # 输出图像类型统计，帮助诊断 HuggingFace datasets 解码问题
            if image_type_stats:
                type_info = ", ".join(f"{k}: {v}" for k, v in sorted(image_type_stats.items()))
                logger.info(f"[ScienceQA] Split '{split}' image type distribution: {type_info}")
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
            """解析预测结果，支持多种格式：
            - 字母 (A/B/C/D)
            - 数字 (0/1/2/3)
            - 文本中提取最后出现的字母/数字
            """
            if isinstance(pred_raw, int):
                return pred_raw
            text = str(pred_raw).strip()
            if not text:
                return -999

            # 1. 尝试从末尾提取字母答案 (A/B/C/D)
            # 常见格式: "The answer is A", "...so the answer is (B)", "A", "Answer: C"
            import re
            # 从后往前找最后一个独立的 A/B/C/D
            # 匹配: 句末的字母、括号内的字母、"answer is X" 等
            letter_patterns = [
                r'[(\[]\s*([A-Da-d])\s*[)\]]',  # (A), [B]
                r'(?:answer|option|choice)(?:\s+is)?[:\s]+([A-Da-d])\b',  # answer is A, answer: B
                r'\b([A-Da-d])\s*[.。]?\s*$',  # 句末的字母
                r'^([A-Da-d])$',  # 只有一个字母
            ]
            for pattern in letter_patterns:
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if matches:
                    letter = matches[-1].group(1).upper()
                    return ord(letter) - ord('A')

            # 2. 尝试纯数字
            if text.isdigit():
                return int(text)

            # 3. 从文本中提取第一个数字
            num_match = re.search(r'\d+', text)
            if num_match:
                return int(num_match.group())

            # 4. 完整选项文本匹配
            choices = sample.get('choices', []) if sample else []
            low = text.lower()
            for idx, opt in enumerate(choices):
                if low == str(opt).strip().lower():
                    return idx

            return -999
        def _judge(pred, ref, sample=None, split_name: str = 'val'):
            def _get_ref_index(r_raw) -> int:
                """将 reference 转换为 0-based 索引"""
                if isinstance(r_raw, int):
                    return r_raw
                r_str = str(r_raw).strip().upper()
                # 处理字母形式 A/B/C/D
                if len(r_str) == 1 and r_str in 'ABCD':
                    return ord(r_str) - ord('A')
                # 尝试数字
                if r_str.isdigit():
                    return int(r_str)
                return -999
            def _single(p_raw, r_raw, smp):
                ref_index = _get_ref_index(r_raw)
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