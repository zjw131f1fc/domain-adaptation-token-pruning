"""SEED-Bench 数据集准备器

目标: 
  - 从 HuggingFace lmms-lab/SEED-Bench 加载 text split
  - 使用父类提供的分片方法（像 MME 那样）
  - 处理多选问题，将选项包装到问题里
  - 支持 A/B/C/D 答案格式的评估

数据结构:
  - 原始数据: {"answer": "A", "choice_a": "...", "choice_b": "...", "choice_c": "...", "choice_d": "...", "question": "...", "image": PIL.Image}
  - 处理后: {"image": PIL.Image, "question": "问题\n(A) 选项A\n(B) 选项B\n(C) 选项C\n(D) 选项D\n请选择正确答案的字母(A/B/C/D):", "answer": "A", "choices": ["选项A", "选项B", "选项C", "选项D"]}

评估方式:
  - 支持 A/B/C/D 字母答案
  - 支持数字索引答案 (0/1/2/3)
  - 支持完整选项文本匹配
"""

from typing import List, Dict, Any
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore
from tqdm import tqdm


class SEEDBenchDataset(BsesDataset):
    pass


class SEEDBenchPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = False  # SEED-Bench 不使用类别字段
        # fast_load_no_random: 仅按 split 中需要的绝对数量 (int / -1 / 0) 顺序截取, 不做随机, 减少图像解码 IO
        # 仅在 int/ -1 / 0 时生效; 比例 (float) 或 'all' 无法在未知总数前确定数量 => 回退为完整加载
        ds_cfg = self.config.dataset_settings
        self.fast_load_no_random = ds_cfg['fast_load_no_random']

    def _load_all(self) -> List[Dict[str, Any]]:
        """加载 SEED-Bench text split 数据"""
        ds = load_dataset("lmms-lab/SEED-Bench", split="test")
        samples: List[Dict[str, Any]] = []
        filtered_count = 0  # 记录过滤掉的数据数量
        
        # 计算 fast load 限额 - 考虑所有split的总和
        limit = None
        if self.fast_load_no_random:
            total_needed = 0
            for split_name, split_value in self.split_cfg.items():
                if isinstance(split_value, int):
                    if split_value == -1:
                        total_needed += 1
                    elif split_value >= 0:
                        total_needed += split_value
                # 其他类型 (float/'all') 不提前截断，回退为完整加载
                else:
                    total_needed = None
                    break
            
            if total_needed is not None and total_needed > 0:
                limit = total_needed
        
        for i in tqdm(range(len(ds)), desc="Loading SEED-Bench"):
            if limit is not None and len(samples) >= limit:
                break
            item = ds[i]
            
            # 检查image字段是否为列表，且列表长度不为1
            image_data = item["image"]
            if isinstance(image_data, list):
                if len(image_data) != 1:
                    filtered_count += 1
                    continue  # 跳过这个样本
                else:
                    # 提取列表中的单个图片
                    image_data = image_data[0]
            
            # 提取选项
            choices = [
                item["choice_a"],
                item["choice_b"], 
                item["choice_c"],
                item["choice_d"]
            ]
            
            # 构建完整问题，包含选项和提示
            opt_lines = []
            for idx, choice in enumerate(choices):
                letter = chr(ord('A') + idx)
                opt_lines.append(f"({letter}) {choice}")
            
            # 添加英文提示词让AI回答字母
            instruction = "Please select the correct answer letter (A/B/C/D):"
            full_question = item["question"] + "\n" + "\n".join(opt_lines) + f"\n{instruction}"
            
            samples.append({
                "image": image_data,  # 直接使用单个图片
                "question": full_question,
                "raw_question": item["question"],
                "answer": item["answer"],  # A/B/C/D 格式
                "choices": choices,
            })
        
        # 保存过滤统计信息到实例变量
        self.filtered_count = filtered_count
        return samples

    def get(self) -> Dict[str, Any]:
        samples = self._load_all()
        # 类别检测（已设 False 但保持统一接口）
        self.detect_category(samples)
        # 字段映射
        applied_map = self.apply_field_map(samples)
        # 根据配置拆分（单一列表，像 MME 那样）
        splits, placeholder = self.split_from_single(samples)
        # 构建 meta
        meta = self.build_meta(samples, splits, applied_map, placeholder)
        # judge
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
        logger.info('[SEED-Bench] Presplit: False (单列表随机拆分)')
        logger.info(f"[SEED-Bench] Loaded Samples: {meta['total']}")
        
        # 报告过滤掉的样本数量
        filtered_count = getattr(self, 'filtered_count', 0)
        logger.info(f"[SEED-Bench] Filtered Samples (image list length != 1): {filtered_count}")
        
        # 统计答案分布
        answer_stats: Dict[str, int] = {}
        for ds in splits.values():
            for i in range(len(ds)):
                answer = ds[i]["answer"]
                answer_stats[answer] = answer_stats.get(answer, 0) + 1
        
        logger.info("[SEED-Bench] Global Answer Distribution: " + ", ".join(f"{k}:{v}" for k, v in sorted(answer_stats.items())))

    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, SEEDBenchDataset]):
        """构建 SEED-Bench 评估函数 - 支持多种答案格式"""
        
        def _parse_answer(pred_raw: Any, sample: Dict[str, Any] = None) -> str:
            """解析预测答案，返回标准化的字母答案"""
            if isinstance(pred_raw, str):
                text = pred_raw.strip().upper()
            else:
                text = str(pred_raw).strip().upper()
            
            # 直接匹配字母 A/B/C/D
            if text in ['A', 'B', 'C', 'D']:
                return text
            
            # 尝试从文本中提取字母
            for letter in ['A', 'B', 'C', 'D']:
                if letter in text:
                    return letter
            
            # 尝试数字索引转换 (0->A, 1->B, 2->C, 3->D)
            if text in ['0', '1', '2', '3']:
                idx = int(text)
                return chr(ord('A') + idx)
            
            # 尝试从文本中提取数字
            import re
            numbers = re.findall(r'\d+', text)
            if numbers:
                num = int(numbers[0])
                if 0 <= num <= 3:
                    return chr(ord('A') + num)
            
            # 完整选项文本匹配
            if sample and 'choices' in sample:
                choices = sample['choices']
                text_lower = text.lower()
                for idx, choice in enumerate(choices):
                    if text_lower == choice.strip().lower():
                        return chr(ord('A') + idx)
            
            return None

        def _judge(pred, ref, sample=None, split_name: str = 'test'):
            # 批量评估
            if isinstance(pred, list):
                if not isinstance(ref, list):
                    raise TypeError("批量判定时 ref 也应为列表")
                total = len(pred)
                if len(ref) != total:
                    raise ValueError("pred/ref 长度不一致")
                
                correct = 0
                samples_list = sample if isinstance(sample, list) else [sample] * total
                for i, (p_raw, r_raw) in enumerate(zip(pred, ref)):
                    pred_letter = _parse_answer(p_raw, samples_list[i] if i < len(samples_list) else None)
                    ref_letter = str(r_raw).strip().upper()
                    
                    if pred_letter and pred_letter == ref_letter:
                        correct += 1
                
                return {"correct": correct, "total": total, "accuracy": (correct / total) if total > 0 else 0.0}
            
            # 单条评估
            pred_letter = _parse_answer(pred, sample)
            ref_letter = str(ref).strip().upper()
            is_correct = 1 if (pred_letter and pred_letter == ref_letter) else 0
            return {"correct": is_correct, "total": 1, "accuracy": float(is_correct)}
        
        return _judge
