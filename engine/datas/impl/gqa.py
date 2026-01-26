"""GQA 数据集准备器。

目标:
  - 从 HuggingFace vikhyatk/gqa 加载 train_balanced 和 val_balanced
  - 将每个图片的 qa 列表展开为单独的样本
  - 无类别 (has_category = False)，与 POPE/MME 保持一致的接口模式
  - 使用预拆分模式：train 使用 train_balanced，test 使用 val_balanced

数据结构:
  - 原始数据: {"image": PIL.Image, "qa": [{"question": str, "answer": str, "fullAnswer": str}, ...]}
  - 展开后: {"image": PIL.Image, "question": str, "answer": str, "fullAnswer": str}

性能优化:
  - 延迟图像加载: 加载阶段只存储索引，访问时才加载图像
  - 大幅提升加载速度，减少内存占用

评估方式:
  - 使用 Simple Accuracy（与 LAVIS/BLIP 评估方式一致）
  - 预处理：数字词转数字、移除冠词、处理缩写
  - 精确匹配：pred == gt_answer
  - 参考：https://github.com/salesforce/LAVIS/blob/main/lavis/tasks/vqa.py
"""

import re
from typing import List, Dict, Any, Union
from ..base import BasePreparer, BsesDataset
from datasets import load_dataset  # type: ignore


class VQAEval:
    """VQA 评估预处理工具

    参考 LAVIS 实现：https://github.com/salesforce/LAVIS/blob/main/lavis/tasks/vqa.py
    """

    def __init__(self):
        self.contractions = {
            "aint": "ain't", "arent": "aren't", "cant": "can't",
            "couldve": "could've", "couldnt": "couldn't",
            "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
            "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
            "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
            "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
            "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
            "howd": "how'd", "howll": "how'll", "hows": "how's",
            "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
            "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
            "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
            "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've", "mightve": "might've",
            "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
            "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
            "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
            "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
            "somebodys": "somebody's", "someoned": "someone'd",
            "someoned've": "someone'd've", "someone'dve": "someone'd've",
            "someonell": "someone'll", "someones": "someone's",
            "somethingd": "something'd", "somethingd've": "something'd've",
            "something'dve": "something'd've", "somethingll": "something'll",
            "thats": "that's", "thered": "there'd", "thered've": "there'd've",
            "there'dve": "there'd've", "therere": "there're", "theres": "there's",
            "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
            "theyll": "they'll", "theyre": "they're", "theyve": "they've",
            "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
            "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
            "whatll": "what'll", "whatre": "what're", "whats": "what's",
            "whatve": "what've", "whens": "when's", "whered": "where'd",
            "wheres": "where's", "whereve": "where've", "whod": "who'd",
            "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
            "whos": "who's", "whove": "who've", "whyll": "why'll",
            "whyre": "why're", "whys": "why's", "wont": "won't",
            "wouldve": "would've", "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
            "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
            "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
            "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
            "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
            "nine": "9", "ten": "10",
        }
        self.articles = ["a", "an", "the"]
        self.periodStrip = re.compile(r"(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile(r"(\d)(,)(\d)")
        self.punct = [
            ";", r"/", "[", "]", '"', "{", "}", "(", ")", "=", "+", "\\",
            "_", "-", ">", "<", "@", "`", ",", "?", "!",
        ]

    def processPunctuation(self, inText: str) -> str:
        """处理标点符号"""
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (
                re.search(self.commaStrip, inText) is not None
            ):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText)
        return outText

    def processDigitArticle(self, inText: str) -> str:
        """处理数字词和冠词"""
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.get(word, word)
            if word not in self.articles:
                outText.append(word)
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        return " ".join(outText)

    def process(self, text: str) -> str:
        """完整预处理流程"""
        if text is None:
            return ""
        text = str(text).strip()
        text = self.processPunctuation(text)
        text = self.processDigitArticle(text)
        return text


class GQADataset(BsesDataset):
    """GQA 数据集，支持延迟图像加载"""

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


class GQAPreparer(BasePreparer):
    def __init__(self, config):
        super().__init__(config)
        self.has_category = False  # GQA 不使用类别字段
        # 不需要字段映射，直接使用 answer 字段
        # 保存 HuggingFace dataset 引用用于延迟加载
        self._hf_datasets: Dict[str, Any] = {}

    def _load_train(self) -> List[Dict[str, Any]]:
        """加载 train_balanced 数据并展开 qa 列表"""
        ds = load_dataset("vikhyatk/gqa", split="train_balanced")
        self._hf_datasets['train'] = ds  # 保存引用
        samples: List[Dict[str, Any]] = []

        for i in range(len(ds)):
            item = ds[i]
            qa_list = item["qa"]

            # 将每个 qa 项展开为单独的样本
            for qa_item in qa_list:
                samples.append({
                    "image": ('train', i),  # 存储 (split, index) 而非图像
                    "question": qa_item["question"],
                    "answer": qa_item["answer"],
                })

        return samples

    def _load_val_as_test(self) -> List[Dict[str, Any]]:
        """加载 val_balanced 数据作为测试集并展开 qa 列表"""
        ds = load_dataset("vikhyatk/gqa", split="val_balanced")
        self._hf_datasets['test'] = ds  # 保存引用
        samples: List[Dict[str, Any]] = []

        for i in range(len(ds)):
            item = ds[i]
            qa_list = item["qa"]

            # 将每个 qa 项展开为单独的样本
            for qa_item in qa_list:
                samples.append({
                    "image": ('test', i),  # 存储 (split, index) 而非图像
                    "question": qa_item["question"],
                    "answer": qa_item["answer"],
                })

        return samples

    def _load_presplits(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载预拆分的数据"""
        data: Dict[str, List[Dict[str, Any]]] = {}

        # train 使用 train_balanced
        data['train'] = self._load_train()

        # test 使用 val_balanced（如果有 test 配置）
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
        base_splits, placeholder = self.split_from_presplits(presplits)
        # 转换为 GQADataset（支持延迟加载）
        splits: Dict[str, GQADataset] = {}
        for name, ds in base_splits.items():
            splits[name] = GQADataset(ds.samples, self._hf_datasets)
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
        logger = getattr(self.config, 'logger', None)
        if logger is None:
            return

        self.base_report(meta)
        logger.info('[GQA] Presplit: True (train使用train_balanced，test使用val_balanced)')
        logger.info(f"[GQA] Loaded Samples: {meta['total']}")

        # 统计每个 split 的样本数量
        for name, ds in meta['split_sizes'].items():
            logger.info(f"[GQA] Split '{name}': {ds} samples")

    def _build_judge(self):
        """构建 judge 函数 - 使用 VQA 标准评估方式

        参考 LAVIS 实现：
        1. 预处理：标点、数字词转换、移除冠词
        2. 精确匹配：processed_pred == processed_answer
        """
        vqa_eval = VQAEval()

        def _judge(pred: str, ref: str, sample=None, split_name: str = 'test') -> Dict[str, Any]:
            """单样本评估"""
            # 预处理预测和参考答案
            pred_processed = vqa_eval.process(pred)
            ref_processed = vqa_eval.process(ref)

            # 精确匹配
            is_correct = pred_processed == ref_processed

            return {
                'correct': 1 if is_correct else 0,
                'total': 1,
                'accuracy': 1.0 if is_correct else 0.0,
            }

        # 添加批量评估方法
        def judge_batch(predictions: List[str], references: List[str]) -> Dict[str, Any]:
            """批量评估"""
            total_correct = 0
            for pred, ref in zip(predictions, references):
                pred_processed = vqa_eval.process(pred)
                ref_processed = vqa_eval.process(ref)
                if pred_processed == ref_processed:
                    total_correct += 1

            accuracy = total_correct / len(predictions) if predictions else 0.0
            return {
                'correct': total_correct,
                'total': len(predictions),
                'accuracy': accuracy,
            }

        # 将批量方法绑定到 judge 函数
        _judge.batch = judge_batch

        return _judge
