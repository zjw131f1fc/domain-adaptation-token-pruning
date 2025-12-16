"""MMBench 数据集准备器 (单选题)

仿照 MME/POPE 的实现风格，提供统一接口：
  get() -> { 'splits': {name: Dataset}, 'meta': meta_dict, 'judge': callable }

源数据: HuggingFace datasets -> "lmms-lab/MMBench" (使用 split='dev' 或 'test'?).
官方数据集包含字段: A, B, C, D, question, answer, category, image
  - A/B/C/D: 选项文本 (字符串)
  - question: 问题文本
  - answer: 正确答案选项字母 (例如 'A','B','C','D')
  - category: 题目类别
  - image: 图像对象 (PIL 或 array) 由 datasets 返回

拆分策略: 与 MME 相同，按 config.dataset_settings['split'] 进行:
  - 支持 float / int / -1 / 'all' 及 category_priority (需 enable: True)

judge 逻辑 (单选):
  - 单条: pred 若是字符串，归一化大写后与正确选项字母完全相同视为 1，否则 0
    * 若 pred 给出的是选项完整文本，将尝试匹配 A/B/C/D 文本 (归一化) 定位到其字母
  - 批量: 对齐 zip(pred, ref)，分别判定

归一化规则:
  - 去除首尾空白
  - 大写化字母
  - 仅保留首个非空 token (用于防止模型回答 "A. xxx")

无 try/except 包装，配置 / 数据异常直接抛出。
"""
import os
from typing import Dict, Any, List
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore
import pandas as pd


class MMBDataset(BsesDataset):
    pass


class MMBenchPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        # 明确存在 category 字段
        self.has_category = True
        # 可通过 dataset_settings['field_map'] 自定义映射，这里不强制

    # --- 预拆分加载：dev -> train, test -> test ---
    def _load_dev(self) -> List[Dict[str, Any]]:
        # 原实现中使用 test split，这里改为 dev split 作为训练集合
        subsets = ['cn', 'en']
        merged: List[Dict[str, Any]] = []
        for sub in subsets:
            ds = load_dataset("lmms-lab/MMBench", sub, split="dev")
            for i in range(len(ds)):
                item = ds[i]
                base_cat = item['category']
                new_cat = f"{base_cat}-{sub}" if base_cat is not None else sub
                q_raw = item['question']
                opt_lines = [
                    f"A. {item['A']}",
                    f"B. {item['B']}",
                    f"C. {item['C']}",
                    f"D. {item['D']}",
                ]
                instr = "Choose the correct answer from A/B/C/D and output only one letter (A, B, C, or D)."
                full_q = f"{q_raw}\n" + "\n".join(opt_lines) + f"\n{instr}"
                merged.append({
                    'image': item['image'],
                    'question': full_q,
                    'A': item['A'],
                    'B': item['B'],
                    'C': item['C'],
                    'D': item['D'],
                    'answer': item['answer'],
                    'category': new_cat,
                    'raw_question': q_raw,
                    'subset': sub,
                })
        return merged

    def _load_test(self) -> List[Dict[str, Any]]:
        # test split: 不保证有答案? 数据集中 answer 字段仍存在 (保持评估一致)
        subsets = ['cn', 'en']
        merged: List[Dict[str, Any]] = []
        for sub in subsets:
            ds = load_dataset("lmms-lab/MMBench", sub, split="test")
            for i in range(len(ds)):
                item = ds[i]
                base_cat = item['category']
                new_cat = f"{base_cat}-{sub}" if base_cat is not None else sub
                q_raw = item['question']
                opt_lines = [
                    f"A. {item['A']}",
                    f"B. {item['B']}",
                    f"C. {item['C']}",
                    f"D. {item['D']}",
                ]
                instr = "Choose the correct answer from A/B/C/D and output only one letter (A, B, C, or D)."
                full_q = f"{q_raw}\n" + "\n".join(opt_lines) + f"\n{instr}"
                merged.append({
                    'image': item['image'],
                    'question': full_q,
                    'A': item['A'],
                    'B': item['B'],
                    'C': item['C'],
                    'D': item['D'],
                    'answer': item['answer'],
                    'category': new_cat,
                    'raw_question': q_raw,
                    'subset': sub,
                })
        return merged

    def _load_presplits(self) -> Dict[str, List[Dict[str, Any]]]:
        data: Dict[str, List[Dict[str, Any]]] = {}
        # dev -> train 永远加载
        data['train'] = self._load_dev()
        # test 若在 split 配置中出现则加载
        if 'test' in self.split_cfg:
            data['test'] = self._load_test()
        return data

    def get(self) -> Dict[str, Any]:
        presplits = self._load_presplits()
        # 汇总所有样本用于统计
        all_samples: List[Dict[str, Any]] = []
        for lst in presplits.values():
            all_samples.extend(lst)
        self.detect_category(all_samples)
        applied_map = self.apply_field_map(all_samples)
        splits, placeholder = self.split_from_presplits(presplits)
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
        logger.info('[MMB] Presplit: True (dev/test 原始多子集)')
        if meta['has_category']:
            total_cat: Dict[Any, int] = {}
            for ds in splits.values():
                for i in range(len(ds)):
                    c = ds[i]['category']
                    if c in total_cat:
                        total_cat[c] += 1
                    else:
                        total_cat[c] = 1
            logger.info("[MMB] Global Category Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(total_cat.items(), key=lambda x: (-x[1], str(x[0])))))

    # ---- judge ----
    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, MMBDataset]):
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
            for k in ['A','B','C','D']:
                cand_map[k] = str(sample[k]).strip().lower()
            for letter, text in cand_map.items():
                if pt == text:
                    return letter
            return _norm_option_key(pred_text)

        # debug judge print 控制
        debug_enabled = False
        debug_print_count = {"count": 0}
        debug_print_limit = 5
        logger = getattr(self.config, 'logger', None)

        # 创建MMBench提交文件
        backbone_name = self._sanitize_name(self.config.backbone_settings['name'])
        dataset_name = self._sanitize_name(self.config.dataset_settings['dataset_name'])
        timestamp = getattr(self.config, 'timestamp', 'unknown')
        # 获取当前evaluate mode (此时已经是单个mode字符串，不是列表)
        eval_mode = self.config.evaluation_settings['eval_mode']
        eval_mode_name = self._sanitize_name(str(eval_mode))
        # 提交文件放置到按数据集分层的日志目录下: logs/<dataset_name>/
        log_root = self.config.file_settings['log_dir']
        dataset_log_dir = os.path.join(log_root, dataset_name)
        if not os.path.exists(dataset_log_dir):
            os.makedirs(dataset_log_dir, exist_ok=True)
        submit_filename = f"submit_{timestamp}_{backbone_name}_{dataset_name}_{eval_mode_name}.xlsx"
        submit_path = os.path.join(dataset_log_dir, submit_filename)
        
        # 初始化提交文件数据结构
        submission_data = []

        def _judge(pred, ref, sample=None, split_name: str = 'test'):
            # ref: 正确答案字母 (A/B/C/D)
            def _single(p_raw, r_raw, sample_item, save_submission=False):
                letter_ref = _norm_option_key(r_raw)
                # 预测可能是字母或完整选项文本
                letter_pred = _norm_option_key(p_raw)
                # 若仍未直接是 A-D, 尝试匹配完整文本
                if letter_pred not in ('A','B','C','D') and sample_item is not None:
                    letter_pred = _map_full_text_to_letter(sample_item, str(p_raw))
                
                # 如果是test split，保存到提交文件
                if save_submission and sample_item is not None:
                    submission_row = {
                        'index': sample_item['index'],
                        'question': sample_item['raw_question'],
                        'A': sample_item['A'],
                        'B': sample_item['B'],
                        'C': sample_item['C'],
                        'D': sample_item['D'],
                        'answer': letter_ref,
                        'prediction': letter_pred,
                        'category': sample_item['category'],
                        'subset': sample_item['subset'],
                    }
                    submission_data.append(submission_row)
                
                is_correct = 1 if letter_pred == letter_ref and letter_ref in ('A','B','C','D') else 0
                # debug 打印
                if debug_enabled and debug_print_count["count"] < debug_print_limit and split_name != 'train' and logger is not None:
                    logger.info(f"[DEBUG_JUDGE] 答案: {letter_ref} | 预测: {letter_pred} | 判定: {is_correct}")
                    debug_print_count["count"] += 1
                return is_correct

            # 判断是否为test split并需要保存提交文件
            save_to_submission = (split_name == 'test')

            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError('批量判定时 ref 也应为列表')
                total = len(pred)
                if len(ref) != total:
                    raise ValueError('pred/ref 长度不一致')
                correct = 0
                if sample is not None and isinstance(sample, list):
                    for idx, (p, r, smp) in enumerate(zip(pred, ref, sample)):
                        correct += _single(p, r, smp, save_to_submission)
                else:
                    for p, r in zip(pred, ref):
                        correct += _single(p, r, None, save_to_submission)
                        
                # 如果是test split，保存提交文件
                if save_to_submission and submission_data:
                    df = pd.DataFrame(submission_data)
                    df.to_excel(submit_path, index=False)
                    if logger:
                        logger.info(f"[MMB] 保存提交文件到: {submit_path} ({len(submission_data)} 条记录)")
                
                return {'correct': correct, 'total': total, 'accuracy': (correct/total) if total>0 else 0.0}
            
            # 单条
            is_correct = _single(pred, ref, sample, save_to_submission)
            
            # 如果是test split单条评估，也要保存（实时保存，防止中断丢失数据）
            if save_to_submission and submission_data:
                df = pd.DataFrame(submission_data)
                df.to_excel(submit_path, index=False)
                
            return {'correct': is_correct, 'total': 1, 'accuracy': float(is_correct)}
        return _judge

    def _sanitize_name(self, text: str) -> str:
        """清理文件名中的特殊字符"""
        seg = text.split('/')[-1].lower()
        out = []
        prev_dash = False
        for ch in seg:
            if ch.isalnum():
                out.append(ch)
                prev_dash = False
            else:
                if not prev_dash:
                    out.append('-')
                    prev_dash = True
        return ''.join(out).strip('-')
