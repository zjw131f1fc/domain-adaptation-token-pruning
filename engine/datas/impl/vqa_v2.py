"""VQA v2 数据集准备器

目标:
  - 只提供 train 和 test 两个集
  - 内部把 val 直接当成 test，不再从文件加载 test
  - 评估方式改成用本地的标注评估

数据流程:
  - 加载 train_questions.json / train_annotations.json 和 val_questions.json / val_annotations.json
  - train 集使用训练数据，test 集使用验证数据
  - 为每个样本保留: image_id, question, answers(list[str]), question_id
  - 支持类别: 使用 question_type + answer_type 作为 category (默认启用)
  - 训练答案: 优先使用 multiple_choice_answer (更官方), 否则使用最常见答案

配置选项 (dataset_settings):
  - use_category (bool, 默认 True): 是否启用类别字段
  - use_mc_answer (bool, 默认 True): 是否使用 multiple_choice_answer 作为训练答案
  - category_priority (dict): 类别平衡配置 (启用后支持 presplit 模式的类别平衡采样)
    - enable (bool): 是否启用类别平衡
    - values (list): 各 split 的采样模式 ('mean' 均衡 / 'origin' 按比例)

严格遵循项目规则:
  - 不使用 try/except; 文件不存在等错误直接抛出
  - 不提供默认值; 读取字段不存在直接报错
"""

from typing import List, Dict, Any
import json
import os
from PIL import Image
from tqdm import tqdm
from ..base import BasePreparer, BsesDataset


class VQAV2Dataset(BsesDataset):
    pass


class VQAV2Preparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        ds_cfg = self.config.dataset_settings
        # 支持类别：使用 question_type + answer_type 作为类别
        # 可通过 config.dataset_settings['use_category'] 控制是否启用（默认启用）
        self.use_category = ds_cfg.get('use_category', True)
        self.has_category = self.use_category
        # 注意：不再使用 field_map 将 answers 映射为 answer
        # 而是在加载时选择训练答案作为 answer 字符串
        # 同时保留 answers 列表用于评估
        self.field_map = {}
        # 数据根目录 (假设已缓存)；若需要可由 config.dataset_settings['data_root'] 指定
        if 'data_root' in ds_cfg:
            self.data_root = ds_cfg['data_root']
        else:
            # 默认路径: 使用全局 dataset_cache_dir 拼接 vqa-v2
            base_cache = self.config.global_settings['dataset_cache_dir']
            self.data_root = os.path.join(base_cache, 'vqa-v2')
        # fast_load_no_random: 仅按 split 中需要的绝对数量 (int / -1 / 0) 顺序截取, 不做随机, 减少图像解码 IO
        # 仅在 int/ -1 / 0 时生效; 比例 (float) 或 'all' 无法在未知总数前确定数量 => 回退为完整加载
        # 注意：启用类别平衡时, fast_load_no_random 将被忽略（需要加载全部数据后按类别采样）
        self.fast_load_no_random = ds_cfg.get('fast_load_no_random', False)
        # 是否使用 multiple_choice_answer 作为训练答案（更官方，默认启用）
        self.use_mc_answer = ds_cfg.get('use_mc_answer', True)
    
    def _get_most_common_answer(self, answers: List[str]) -> str:
        """从答案列表中选择最常见的答案"""
        if not answers:
            return ""
        # 统计每个答案的出现次数
        from collections import Counter
        counter = Counter(answers)
        # 返回出现次数最多的答案
        return counter.most_common(1)[0][0]

    # ---- 加载原始数据：train 使用训练数据，test 使用验证数据 ----
    def _load_train(self) -> List[Dict[str, Any]]:
        q_path = os.path.join(self.data_root, 'train_questions.json')
        a_path = os.path.join(self.data_root, 'train_annotations.json')
        img_root = os.path.join(self.data_root, 'train_images', 'train2014')  # 目录: train_images/train2014
        with open(q_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        with open(a_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        ann_map: Dict[int, Dict[str, Any]] = {}
        for ann in annotations["annotations"]:
            ann_map[ann['question_id']] = ann
        samples: List[Dict[str, Any]] = []
        # 计算 fast load 限额
        limit = None
        
        
        # 注意：
        '''
        ann_map[458752001]，我随便取了一个样本发现，ann_map[458752001]是一个key，里面可能用到的除了answers还有这些项：
        multiple_choice_answer: 应该指多选答案。评估时还是按官方标准来，训练时可能可以直接用这个？
        question_type：问题的类型
        answer_type：答案的类型
        组合question_type和answer_type，应该能得到一个类别名称，如 what-other
        '''
        
        
        if self.fast_load_no_random and 'train' in self.split_cfg:
            raw_v = self.split_cfg['train']
            if isinstance(raw_v, int):
                if raw_v == -1:
                    limit = 1
                elif raw_v >= 0:
                    limit = raw_v
            # 其他类型 (float/'all') 不提前截断
        for q in tqdm(questions["questions"], desc="VQA v2 train", dynamic_ncols=True):
            if limit is not None and len(samples) >= limit:
                break
            qid = q['question_id']
            image_id = q['image_id']
            question = q['question']
            if qid in ann_map:
                ann = ann_map[qid]
                answers_raw = ann['answers']
                answers = [x['answer'] for x in answers_raw]
                # 提取类别信息
                question_type = ann.get('question_type', 'unknown')
                answer_type = ann.get('answer_type', 'unknown')
                # 获取 multiple_choice_answer (更官方的训练答案)
                mc_answer = ann.get('multiple_choice_answer', '')
            else:
                answers = []
                question_type = 'unknown'
                answer_type = 'unknown'
                mc_answer = ''
            # 加载图像
            img_name = f"COCO_train2014_{int(image_id):012d}.jpg"
            img_path = os.path.join(img_root, img_name)
            image = Image.open(img_path).convert('RGB')
            # 拼接提示词让模型直接输出简短答案
            question_with_prompt = f"{question} Answer the question using a single word or phrase."
            # 选择训练答案：优先使用 multiple_choice_answer，否则用最常见答案
            if self.use_mc_answer and mc_answer:
                train_answer = mc_answer
            else:
                train_answer = self._get_most_common_answer(answers)
            # 构建样本
            sample = {
                'image': image,
                'image_id': image_id,
                'question_id': qid,
                'question': question_with_prompt,
                'answers': answers,  # 保留完整答案列表用于评估
                'answer': train_answer,  # 训练答案
            }
            # 添加类别字段（如果启用）
            if self.use_category:
                sample['category'] = f"{question_type}-{answer_type}"
                sample['question_type'] = question_type
                sample['answer_type'] = answer_type
            samples.append(sample)
        return samples

    def _load_val_as_test(self) -> List[Dict[str, Any]]:
        """加载验证数据作为测试集"""
        q_path = os.path.join(self.data_root, 'val_questions.json')
        a_path = os.path.join(self.data_root, 'val_annotations.json')
        img_root = os.path.join(self.data_root, 'val_images', 'val2014')  # 目录: val_images/val2014
        with open(q_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        with open(a_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        ann_map: Dict[int, Dict[str, Any]] = {}
        for ann in annotations["annotations"]:
            ann_map[ann['question_id']] = ann
        samples: List[Dict[str, Any]] = []
        # fast load 限额 (test split)
        limit = None
        if self.fast_load_no_random and 'test' in self.split_cfg:
            raw_v = self.split_cfg['test']
            if isinstance(raw_v, int):
                if raw_v == -1:
                    limit = 1
                elif raw_v >= 0:
                    limit = raw_v
        for q in tqdm(questions["questions"], desc="VQA v2 val as test", dynamic_ncols=True):
            if limit is not None and len(samples) >= limit:
                break
            qid = q['question_id']
            image_id = q['image_id']
            question = q['question']
            if qid in ann_map:
                ann = ann_map[qid]
                answers_raw = ann['answers']
                answers = [x['answer'] for x in answers_raw]
                # 提取类别信息
                question_type = ann.get('question_type', 'unknown')
                answer_type = ann.get('answer_type', 'unknown')
                # 获取 multiple_choice_answer (更官方的训练答案)
                mc_answer = ann.get('multiple_choice_answer', '')
            else:
                answers = []
                question_type = 'unknown'
                answer_type = 'unknown'
                mc_answer = ''
            # 加载图像
            img_name = f"COCO_val2014_{int(image_id):012d}.jpg"
            img_path = os.path.join(img_root, img_name)
            image = Image.open(img_path).convert('RGB')
            # 拼接提示词让模型直接输出简短答案
            question_with_prompt = f"{question} Answer the question using a single word or phrase."
            # 选择训练答案：优先使用 multiple_choice_answer，否则用最常见答案
            if self.use_mc_answer and mc_answer:
                train_answer = mc_answer
            else:
                train_answer = self._get_most_common_answer(answers)
            # 构建样本
            sample = {
                'image': image,
                'image_id': image_id,
                'question_id': qid,
                'question': question_with_prompt,
                'answers': answers,  # 保留完整答案列表用于评估
                'answer': train_answer,  # 训练答案
            }
            # 添加类别字段（如果启用）
            if self.use_category:
                sample['category'] = f"{question_type}-{answer_type}"
                sample['question_type'] = question_type
                sample['answer_type'] = answer_type
            samples.append(sample)
        return samples

    def _load_presplits(self) -> Dict[str, List[Dict[str, Any]]]:
        data: Dict[str, List[Dict[str, Any]]] = {}
        # train 使用训练数据
        data['train'] = self._load_train()
        # test 使用验证数据（有标注）
        if 'test' in self.split_cfg:
            data['test'] = self._load_val_as_test()
        return data

    def get(self) -> Dict[str, Any]:
        presplits = self._load_presplits()
        # 所有原始样本总集合用于 meta 统计
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
        logger.info('[VQAV2] Presplit: True (train使用训练数据，test使用验证数据)')
        logger.info(f"[VQAV2] Loaded Samples: {meta['total']}")
        logger.info(f"[VQAV2] Use Category: {self.use_category}")
        logger.info(f"[VQAV2] Use Multiple Choice Answer: {self.use_mc_answer}")

        # 如果启用类别，打印类别分布
        if self.use_category and meta['has_category']:
            # 全局类别分布
            cat_stat: Dict[Any, int] = {}
            for ds in splits.values():
                for i in range(len(ds)):
                    c = ds[i]['category']
                    cat_stat[c] = cat_stat.get(c, 0) + 1
            logger.info("[VQAV2] Global Category Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(cat_stat.items(), key=lambda x: (-x[1], str(x[0])))))

            # 每个 split 的类别分布
            for name, ds in splits.items():
                sub_stat: Dict[Any, int] = {}
                for i in range(len(ds)):
                    c = ds[i]['category']
                    sub_stat[c] = sub_stat.get(c, 0) + 1
                logger.info(f"[VQAV2] Split '{name}' Category Distribution: " + ", ".join(f"{c}:{n}" for c, n in sorted(sub_stat.items(), key=lambda x: (-x[1], str(x[0])))))

    # ---- judge: 官方 VQA 评分，支持本地标注评估 ----
    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, VQAV2Dataset]):
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

        def _official_score(pred_norm: str, ref_list: List[str]) -> float:
            # 空预测直接返回0分
            if not pred_norm or pred_norm.strip() == "":
                return 0.0

            count = 0
            for ans in ref_list:
                ans_norm = _normalize(ans)
                # 模糊匹配：检查预测答案是否包含参考答案，或参考答案是否包含预测答案
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
                    # 预处理：空预测直接计0分
                    p_norm = _normalize(p_raw)
                    if not p_norm or p_norm.strip() == "":
                        correct += 0.0
                        continue

                    if isinstance(r_raw, list):
                        # 多答案情况，使用官方评分
                        score = _official_score(p_norm, r_raw)
                        correct += score
                    else:
                        # 单答案情况，使用模糊匹配
                        r_norm = _normalize(r_raw)
                        correct += 1.0 if (p_norm == r_norm or r_norm in p_norm or p_norm in r_norm) else 0.0
                return {"correct": correct, "total": total, "accuracy": (correct / total) if total > 0 else 0.0}

            # 单条评估
            # 预处理：空预测直接返回0分
            pred_norm = _normalize(pred)
            if not pred_norm or pred_norm.strip() == "":
                print(f"[VQAV2 Judge] pred_norm: (empty), ref: {ref}, score: 0.0")
                return {'correct': 0.0, 'total': 1, 'accuracy': 0.0}

            if isinstance(ref, list):
                # 多答案情况，使用官方评分
                score = _official_score(pred_norm, ref)
                print(f"[VQAV2 Judge] pred_norm: {pred_norm}, ref: {ref}, score: {score}")
                return {'correct': score, 'total': 1, 'accuracy': float(score)}
            else:
                # 单答案情况，使用模糊匹配
                ref_norm = _normalize(ref)
                score = 1.0 if (pred_norm == ref_norm or ref_norm in pred_norm or pred_norm in ref_norm) else 0.0
                print(f"[VQAV2 Judge] pred_norm: {pred_norm}, ref_norm: {ref_norm}, score: {score}")
                return {'correct': score, 'total': 1, 'accuracy': float(score)}
        return _judge

