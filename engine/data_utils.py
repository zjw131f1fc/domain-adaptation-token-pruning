"""数据预处理工具函数"""

import torch
from typing import Dict, Any, List


def preprocess_batch(
    batch: List[Dict[str, Any]],
    processor,
    device: torch.device,
    max_length: int = 1024,
    mode: str = "train"
) -> Dict[str, Any]:
    """预处理一个 batch 的数据

    Args:
        batch: 样本列表，每个样本包含 'image', 'question', 'answer'(train mode)
        processor: LLaVA processor
        device: 目标设备
        max_length: 最大序列长度
        mode: 'train' 或 'inference'

    Returns:
        预处理后的数据字典
    """
    images = [sample['image'] for sample in batch]
    questions = [sample['question'] for sample in batch]

    if mode == "train":
        answers = [sample['answer'] for sample in batch]
        prompts = []
        # 获取 EOS token
        eos_token = processor.tokenizer.eos_token or "</s>"
        for q, a in zip(questions, answers):
            # 答案首字母大写，与 compute_task_loss 中的处理保持一致
            # ASSISTANT: 后面加空格，compute_task_loss 中 tokenize 时也要加空格前缀
            # 添加 EOS token，确保模型学会何时停止生成
            prompt = f"USER: <image>\n{q}\nASSISTANT: {a.capitalize()}{eos_token}"
            prompts.append(prompt)
    else:
        answers = None
        prompts = []
        for q in questions:
            prompt = f"USER: <image>\n{q}\nASSISTANT:"
            prompts.append(prompt)

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    input_ids = inputs['input_ids']
    batch_size, seq_len = input_ids.shape

    image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
    n_vision_tokens = 576

    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(image_positions) > 0:
        vision_start = image_positions[0].item()
        vision_end = vision_start + n_vision_tokens
    else:
        vision_start = 1
        vision_end = vision_start + n_vision_tokens

    assistant_ids = processor.tokenizer.encode("\nASSISTANT:", add_special_tokens=False)
    if assistant_ids[0] == 29871:
        assistant_ids = assistant_ids[1:]

    assistant_positions = []
    for i in range(batch_size):
        ids = input_ids[i].tolist()
        found = False
        for j in range(len(ids) - len(assistant_ids) + 1):
            if ids[j:j+len(assistant_ids)] == assistant_ids:
                assistant_positions.append(j + len(assistant_ids))
                found = True
                break
        if not found:
            raise ValueError(f"Cannot find ASSISTANT: in sample {i}")

    question_starts = [vision_end] * batch_size
    question_ends = assistant_positions

    result = {
        'inputs': inputs,
        'images': images,
        'questions': questions,
        'vision_start': vision_start,
        'vision_end': vision_end,
        'question_starts': question_starts,
        'question_ends': question_ends,
        'n_vision': n_vision_tokens,
    }

    if mode == "train":
        answer_starts = assistant_positions

        pad_token_id = processor.tokenizer.pad_token_id
        answer_ends = []
        for i in range(batch_size):
            ids = input_ids[i].tolist()
            end_pos = seq_len
            for j in range(answer_starts[i], seq_len):
                if ids[j] == pad_token_id:
                    end_pos = j
                    break
            answer_ends.append(end_pos)

        for i in range(batch_size):
            if answer_ends[i] <= answer_starts[i]:
                raise ValueError(f"Empty answer region in sample {i}")

        result['answers'] = answers
        result['answer_starts'] = answer_starts
        result['answer_ends'] = answer_ends

    return result


class SimpleDataset(torch.utils.data.Dataset):
    """简单的数据集包装器"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """DataLoader 的 collate 函数

    直接返回 batch，不做额外处理（预处理在 train_step 中进行）
    """
    return batch
