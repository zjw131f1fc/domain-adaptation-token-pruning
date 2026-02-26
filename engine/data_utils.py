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


def preprocess_batch_qwen2vl(
    batch: List[Dict[str, Any]],
    processor,
    device: torch.device,
    max_length: int = 2048,
    mode: str = "train",
    target_image_size: int = 672,  # 672 = 48 * 14，产生 48x48/4 = 576 个 vision tokens（与 LLaVA 一致）
) -> Dict[str, Any]:
    """Qwen2-VL 专用的 batch 预处理函数

    与 LLaVA 的主要区别：
    1. Chat template: <|im_start|>role\ncontent<|im_end|>
    2. Vision tokens: <|vision_start|><|image_pad|>...<|vision_end|>
    3. image_token_id = 151655 (<|image_pad|>)

    Qwen2-VL vision tokens 计算：
    - patch_size = 14, merge_size = 2
    - 对于 HxW 图像: n_tokens = (H/14/2) * (W/14/2) = H*W / 784
    - 392x392 -> 196 tokens, 560x560 -> 400 tokens, 672x672 -> 576 tokens
    """
    from PIL import Image

    # Resize 图像到固定大小，确保 vision tokens 数量一致
    images = []
    for sample in batch:
        img = sample['image']
        if isinstance(img, Image.Image):
            # Resize 到目标大小（保持正方形）
            img = img.convert('RGB').resize((target_image_size, target_image_size), Image.Resampling.LANCZOS)
        images.append(img)

    questions = [sample['question'] for sample in batch]

    # 构建 Qwen2-VL 格式的 messages
    if mode == "train":
        answers = [sample['answer'] for sample in batch]
        messages_list = []
        for q, a in zip(questions, answers):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": q},
                    ],
                },
                {
                    "role": "assistant",
                    "content": a.capitalize(),
                },
            ]
            messages_list.append(messages)
    else:
        answers = None
        messages_list = []
        for q in questions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "placeholder"},
                        {"type": "text", "text": q},
                    ],
                },
            ]
            messages_list.append(messages)

    # 使用 chat template 生成 prompts
    prompts = [
        processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=(mode != "train")
        )
        for msgs in messages_list
    ]

    # 处理输入
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

    # Qwen2-VL 的 image_token_id = 151655 (<|image_pad|>)
    image_token_id = 151655

    # 找到 vision tokens 的位置（第一个样本）
    image_mask = input_ids[0] == image_token_id
    image_positions = image_mask.nonzero(as_tuple=True)[0]

    if len(image_positions) > 0:
        vision_start = image_positions[0].item()
        vision_end = image_positions[-1].item() + 1
        n_vision_tokens = vision_end - vision_start
    else:
        # fallback
        vision_start = 1
        n_vision_tokens = 576
        vision_end = vision_start + n_vision_tokens

    # 找到 assistant 回答的起始位置
    # Qwen2-VL 格式: <|im_start|>assistant\n
    assistant_start_ids = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )

    assistant_positions = []
    for i in range(batch_size):
        ids = input_ids[i].tolist()
        found = False
        for j in range(len(ids) - len(assistant_start_ids) + 1):
            if ids[j:j+len(assistant_start_ids)] == assistant_start_ids:
                assistant_positions.append(j + len(assistant_start_ids))
                found = True
                break
        if not found:
            raise ValueError(f"Cannot find assistant start in sample {i}")

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

        # 找到答案结束位置（<|im_end|> 或 pad）
        im_end_id = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
        pad_token_id = processor.tokenizer.pad_token_id

        answer_ends = []
        for i in range(batch_size):
            ids = input_ids[i].tolist()
            end_pos = seq_len
            for j in range(answer_starts[i], seq_len):
                if ids[j] == im_end_id or ids[j] == pad_token_id:
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
