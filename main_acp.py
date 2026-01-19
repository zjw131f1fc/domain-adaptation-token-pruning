#!/usr/bin/env python
"""Attention Consistency Pruning - 完整训练脚本

基于新的 Attention Consistency Pruning 架构的完整训练流程。

特点：
1. 直接继承 LlavaForConditionalGeneration，不使用 hook
2. 在剪枝层计算 h_real 和 h_fake
3. 每个 answer token 独立判别
4. 使用统一的数据加载器和配置系统
"""

import os
import sys

# ============================================================
# 环境变量设置（在任何其他 import 之前）
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/root/autodl-tmp/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置加载器
from engine.configs.loader import load_config


# ============================================================
# 模型加载与预处理
# ============================================================

def load_model(config, device: torch.device):
    """加载可剪枝的 LLaVA 模型"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    logger = config.logger
    method_cfg = config.method_settings
    backbone_cfg = config.backbone_settings
    global_cfg = config.global_settings

    # 获取数据类型配置
    dtype_str = global_cfg.get('dtype', 'float32')
    dtype_mapping = {
        'float16': torch.float16,
        'fp16': torch.float16,
        'float32': torch.float32,
        'fp32': torch.float32,
        'bfloat16': torch.bfloat16,
        'bf16': torch.bfloat16,
    }
    torch_dtype = dtype_mapping.get(dtype_str, torch.float32)
    logger.info(f"Using dtype: {dtype_str} -> {torch_dtype}")

    # 获取剪枝层配置
    pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])
    pruner_d_internal = method_cfg.get('pruner_d_internal', 128)
    pruner_n_heads = method_cfg.get('pruner_n_heads', 4)
    disc_d_head = method_cfg.get('disc_d_head', 64)
    disc_d_hidden = method_cfg.get('disc_d_d', 128)
    temperature = method_cfg.get('temperature', 1.0)
    dropout = method_cfg.get('pruner_dropout', 0.1)
    disc_spectral_norm = method_cfg.get('disc_use_spectral_norm', False)

    # 模型路径
    model_name = backbone_cfg.get('name', 'llava-1.5-7b')
    model_mapping = {
        'llava-1.5-7b': 'llava-hf/llava-1.5-7b-hf',
        'llava-1.5-13b': 'llava-hf/llava-1.5-13b-hf',
    }
    model_path = model_mapping.get(model_name, model_name)

    logger.info(f"Loading base model from {model_path}...")

    # 加载基础模型和处理器
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map='auto',
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # 设置 padding side 为 right（参考 TRAINING_NOTES）
    processor.tokenizer.padding_side = "right"

    # 将 processor 附加到模型
    base_model.processor = processor

    # 创建可剪枝模型
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration

    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=pruning_layers,
        pruner_d_internal=pruner_d_internal,
        pruner_n_heads=pruner_n_heads,
        disc_d_head=disc_d_head,
        disc_d_hidden=disc_d_hidden,
        temperature=temperature,
        dropout=dropout,
    )

    # 冻结基础模型
    model.freeze_base_model()

    logger.info(f"Model loaded. Pruning layers: {pruning_layers}")
    logger.info(f"Trainable parameters: Pruners={sum(p.numel() for p in model.get_pruner_parameters()):,}, Discriminators={sum(p.numel() for p in model.get_discriminator_parameters()):,}")

    return model, processor


def preprocess_batch(
    batch: List[Dict[str, Any]],
    processor,
    device: torch.device,
    max_length: int = 1024,
    mode: str = "train"  # "train" 或 "inference"
) -> Dict[str, Any]:
    """预处理一个 batch 的数据

    参数:
        batch: 数据列表，每个元素包含 image, question, (answer)
        processor: LLaVA processor
        device: 设备
        max_length: 最大序列长度（需要 > 576 vision tokens + 问题 + 答案）
        mode: "train" 训练模式（包含 answer），"inference" 推理模式（不包含 answer）

    返回:
        预处理后的输入字典
    """
    images = [sample['image'] for sample in batch]
    questions = [sample['question'] for sample in batch]

    if mode == "train":
        answers = [sample['answer'] for sample in batch]
        # 构建 prompt（训练时需要包含 answer）
        prompts = []
        for q, a in zip(questions, answers):
            prompt = f"USER: <image>\n{q}\nASSISTANT: {a}"
            prompts.append(prompt)
    else:
        # 推理模式：不包含 answer
        answers = None
        prompts = []
        for q in questions:
            prompt = f"USER: <image>\n{q}\nASSISTANT:"
            prompts.append(prompt)

    # 使用 processor 处理（不截断，避免破坏 image tokens）
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    # 找到各个区域的位置
    input_ids = inputs['input_ids']
    batch_size, seq_len = input_ids.shape

    # 找 image token 位置
    # LLaVA 使用特殊的 image token id
    # 注意：实际的 vision tokens 会替换掉 <image> placeholder
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')

    # 对于 LLaVA 1.5，vision tokens 数量固定为 576 (24x24 patches)
    n_vision_tokens = 576

    # 找第一个样本的位置（假设 batch 内位置一致）
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(image_positions) > 0:
        vision_start = image_positions[0].item()
        vision_end = vision_start + n_vision_tokens
    else:
        # 如果找不到 image token，可能是因为已经被替换
        # 尝试通过序列结构推断
        vision_start = 1  # 跳过 BOS
        vision_end = vision_start + n_vision_tokens

    # Question 位置：从 vision_end 到 ASSISTANT: 之前
    # Answer 位置：从 ASSISTANT: 之后开始
    # 注意：LLaMA tokenizer 会在编码时添加前导空格 token (29871)
    # 实际序列中 "\nASSISTANT:" 编码为 [13, 22933, 9047, 13566, 29901]
    # 但 encode("\nASSISTANT:") 返回 [29871, 13, 22933, ...]
    # 需要跳过开头的空格 token
    assistant_ids = processor.tokenizer.encode("\nASSISTANT:", add_special_tokens=False)
    # 跳过 SentencePiece 自动添加的前导空格 token (29871 = ▁)
    if assistant_ids[0] == 29871:
        assistant_ids = assistant_ids[1:]

    # 找到 ASSISTANT: 的位置（对于推理和训练都需要）
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
            raise ValueError(f"Cannot find ASSISTANT: in sample {i}. assistant_ids={assistant_ids}, ids[-30:]={ids[-30:]}")

    # question 区域：从 vision_end 到 ASSISTANT: 之前
    question_starts = [vision_end] * batch_size
    question_ends = assistant_positions  # 每个样本的 question 结束位置

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
        # 训练模式：需要 answer 位置
        answer_starts = assistant_positions

        # 找到每个样本的实际 answer 结束位置（排除 padding）
        pad_token_id = processor.tokenizer.pad_token_id
        answer_ends = []
        for i in range(batch_size):
            ids = input_ids[i].tolist()
            # 从 answer_start 开始找第一个 pad token
            end_pos = seq_len
            for j in range(answer_starts[i], seq_len):
                if ids[j] == pad_token_id:
                    end_pos = j
                    break
            answer_ends.append(end_pos)

        # 确保 answer 区域非空
        for i in range(batch_size):
            if answer_ends[i] <= answer_starts[i]:
                raise ValueError(f"Empty answer region in sample {i}. answer_start={answer_starts[i]}, answer_end={answer_ends[i]}, answer='{answers[i]}'")

        result['answers'] = answers
        result['answer_starts'] = answer_starts
        result['answer_ends'] = answer_ends

    return result


# ============================================================
# 损失计算
# ============================================================

def compute_task_loss(
    logits: torch.Tensor,
    answer_starts: List[int],
    answers: List[str],
    tokenizer,
    device: torch.device
) -> torch.Tensor:
    """计算 task loss (cross entropy)

    参数:
        logits: (batch, seq_len, vocab_size)
        answer_starts: 每个样本的 answer 开始位置
        answers: ground truth 答案列表
        tokenizer: tokenizer

    返回:
        task_loss: 标量
    """
    batch_size = logits.shape[0]
    total_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        # 将 answer 转为首字母大写（参考 TRAINING_NOTES）
        answer = answers[i].capitalize()

        # Tokenize answer
        answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        if len(answer_ids) == 0:
            continue

        # 预测位置：answer_start - 1 开始（causal LM）
        pred_start = answer_starts[i] - 1
        pred_end = min(pred_start + len(answer_ids), logits.shape[1] - 1)

        if pred_start < 0 or pred_end <= pred_start:
            continue

        # 获取 logits 和 targets
        pred_logits = logits[i, pred_start:pred_end]
        target_len = min(len(answer_ids), pred_end - pred_start)
        targets = torch.tensor(answer_ids[:target_len], device=device)

        # Cross entropy
        loss = F.cross_entropy(pred_logits, targets)
        total_loss = total_loss + loss

    return total_loss / batch_size if batch_size > 0 else total_loss


# ============================================================
# 训练步骤
# ============================================================

def train_step(
    batch: List[Dict[str, Any]],
    model,
    processor,
    config,
    current_step: int,
    total_steps: int,
    device: torch.device
) -> Dict[str, Any]:
    """执行一个训练步骤

    返回:
        losses_dict: 包含各项损失和统计信息
    """
    method_cfg = config.method_settings

    # === Temperature Annealing ===
    temperature = method_cfg.get('temperature', 1.0)
    temperature_min = method_cfg.get('temperature_min', 0.5)
    anneal_rate = method_cfg.get('temperature_anneal_rate', 0.4)

    progress = current_step / total_steps if total_steps > 0 else 0
    if progress < anneal_rate:
        current_temp = temperature - (progress / anneal_rate) * (temperature - temperature_min)
    else:
        current_temp = temperature_min

    model.set_temperature(current_temp)

    # === 预处理 ===
    prep = preprocess_batch(batch, processor, device)
    inputs = prep['inputs']

    # === Forward ===
    model.train()

    output = model(
        input_ids=inputs['input_ids'],
        pixel_values=inputs['pixel_values'],
        attention_mask=inputs['attention_mask'],
        vision_start=prep['vision_start'],
        vision_end=prep['vision_end'],
        question_starts=prep['question_starts'],
        question_ends=prep['question_ends'],
        answer_starts=prep['answer_starts'],
        answer_ends=prep['answer_ends'],
        return_pruning_info=True,
    )

    # === 计算 Losses ===
    losses = {}
    stats = {'temperature': current_temp}

    # 1. Task Loss
    task_loss = compute_task_loss(
        output.logits,
        prep['answer_starts'],
        prep['answers'],
        processor.tokenizer,
        device
    )
    losses['task_loss'] = task_loss
    stats['raw_task_loss'] = task_loss.item()

    # 2. 如果有剪枝信息，计算 GAN 相关 losses
    if output.pruning_infos and len(output.pruning_infos) > 0:
        h_real_dict = {idx: info['h_real'] for idx, info in output.pruning_infos.items()}
        h_fake_dict = {idx: info['h_fake'] for idx, info in output.pruning_infos.items()}

        # Adversarial Loss
        adv_loss = model.disc_manager.compute_adv_loss(h_fake_dict)
        losses['adv_loss'] = adv_loss
        stats['raw_adv_loss'] = adv_loss.item()

        # Discriminator Loss
        disc_loss = model.disc_manager.compute_disc_loss(h_real_dict, h_fake_dict)
        losses['disc_loss'] = disc_loss
        stats['raw_disc_loss'] = disc_loss.item()

        # Discriminator Accuracy
        acc_info = model.disc_manager.compute_accuracy(h_real_dict, h_fake_dict)
        stats['disc_accuracy'] = acc_info['overall']
        stats['disc_real_acc'] = acc_info['real_acc']
        stats['disc_fake_acc'] = acc_info['fake_acc']
        # 每层判别器的准确率
        for layer_idx, (real_acc, fake_acc) in acc_info['per_layer'].items():
            stats[f'disc_L{layer_idx}_real_acc'] = real_acc
            stats[f'disc_L{layer_idx}_fake_acc'] = fake_acc
            stats[f'disc_L{layer_idx}_acc'] = (real_acc + fake_acc) / 2

        # Sparsity Loss - 约束 LLM 平均每层的保留比例
        # 剪枝是累积的：L4 剪枝后，L14 在剩余 tokens 上再剪枝
        target_token_num = method_cfg.get('target_token_num', 144)
        n_vision = prep['n_vision']
        target_ratio = target_token_num / n_vision

        # 获取 LLM 总层数和剪枝层索引
        total_layers = len(model.base_model.language_model.layers)
        pruning_layers = sorted(output.pruning_infos.keys())

        # 计算累积保留率
        # 层 0 到第一个剪枝层之前：100% 保留
        # 每个剪枝层到下一个剪枝层之前：累积保留率
        weighted_kept = torch.tensor(0.0, device=device)
        cumulative_ratio = torch.tensor(1.0, device=device)  # 累积保留率

        for i, layer_idx in enumerate(pruning_layers):
            # 剪枝层之前的层数（使用之前的累积保留率）
            if i == 0:
                n_layers_before = layer_idx  # 层 0 到 layer_idx-1
                weighted_kept = weighted_kept + n_layers_before * 1.0  # 第一个剪枝层之前是 100%

            # 该剪枝层的保留率（相对于当前剩余 tokens）
            hard_mask = output.pruning_infos[layer_idx]['hard_mask']
            layer_kept_ratio = hard_mask.mean()
            stats[f'L{layer_idx}_kept'] = layer_kept_ratio.item()

            # 更新累积保留率
            cumulative_ratio = cumulative_ratio * layer_kept_ratio

            # 该剪枝层影响的层数
            if i < len(pruning_layers) - 1:
                n_affected = pruning_layers[i + 1] - layer_idx  # 到下一个剪枝层
            else:
                n_affected = total_layers - layer_idx  # 到最后一层

            weighted_kept = weighted_kept + n_affected * cumulative_ratio
            stats[f'L{layer_idx}_cumulative'] = cumulative_ratio.item()
            stats[f'L{layer_idx}_affected'] = n_affected

        # LLM 平均每层的保留比例
        avg_kept_ratio_tensor = weighted_kept / total_layers

        # sparsity_loss = |平均保留比例 - 目标比例|
        sparsity_loss = torch.abs(avg_kept_ratio_tensor - target_ratio)

        losses['sparsity_loss'] = sparsity_loss
        stats['raw_sparsity_loss'] = sparsity_loss.item()
        stats['avg_kept_ratio'] = avg_kept_ratio_tensor.item()
        stats['target_kept_ratio'] = target_ratio
        stats['total_layers'] = total_layers

    # === 应用权重 ===
    task_weight = method_cfg.get('task_loss_weight', 1.0)
    adv_weight = method_cfg.get('adv_loss_weight', 0.5)
    sparsity_weight = method_cfg.get('sparsity_weight', 0.2)

    # 动态权重 warmup
    warmup_ratio = method_cfg.get('loss_weight_warmup_ratio', 0.0)
    if warmup_ratio > 0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2

        task_weight_start = method_cfg.get('task_loss_weight_start', task_weight)
        adv_weight_start = method_cfg.get('adv_loss_weight_start', adv_weight)

        task_weight = task_weight_start + (task_weight - task_weight_start) * cosine_factor.item()
        adv_weight = adv_weight_start + (adv_weight - adv_weight_start) * cosine_factor.item()

    # Sparsity warmup
    if method_cfg.get('sparsity_warmup_enable', False):
        sparsity_warmup_ratio = method_cfg.get('sparsity_warmup_ratio', 0.2)
        sparsity_weight_max = method_cfg.get('sparsity_weight_max', sparsity_weight)
        if progress < sparsity_warmup_ratio:
            sparsity_weight = sparsity_weight + (sparsity_weight_max - sparsity_weight) * (progress / sparsity_warmup_ratio)
        else:
            sparsity_weight = sparsity_weight_max

    stats['task_weight'] = task_weight
    stats['adv_weight'] = adv_weight
    stats['sparsity_weight'] = sparsity_weight

    # 加权损失
    weighted_losses = {
        'task_loss': losses['task_loss'] * task_weight,
    }
    if 'adv_loss' in losses:
        weighted_losses['adv_loss'] = losses['adv_loss'] * adv_weight
    if 'sparsity_loss' in losses:
        weighted_losses['sparsity_loss'] = losses['sparsity_loss'] * sparsity_weight
    if 'disc_loss' in losses:
        weighted_losses['disc_loss'] = losses['disc_loss']

    return {
        'losses': weighted_losses,
        'stats': stats,
        'pruning_infos': {idx: info for idx, info in output.pruning_infos.items()} if output.pruning_infos else None,
    }


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    processor,
    dataset,
    judge,
    config,
    device: torch.device,
    max_samples: int = 500,
    mode: str = "origin",  # "origin" 或 "hard"
) -> Dict[str, float]:
    """评估模型

    参数:
        model: 模型
        processor: processor
        dataset: 数据集
        judge: 评估函数
        config: 配置
        device: 设备
        max_samples: 最大评估样本数
        mode: 评估模式
            - "origin": 不剪枝（baseline）
            - "hard": 使用剪枝

    返回:
        评估结果字典
    """
    model.eval()

    n_samples = min(len(dataset), max_samples)
    predictions = []
    references = []
    kept_ratios = []
    layer_kept_ratios = {}  # {layer_idx: [ratios]}

    desc = f"Evaluating ({mode})"

    for i in tqdm(range(n_samples), desc=desc):
        sample = dataset[i]

        # 根据模式选择生成方式
        if mode == "hard":
            # 使用 preprocess_batch 获取 question 位置（推理模式，不含 answer）
            preprocessed = preprocess_batch(
                batch=[sample],
                processor=processor,
                device=device,
                mode="inference"
            )
            inputs = preprocessed['inputs']

            # 带硬剪枝的生成（物理删除 tokens，减少 FLOPS）
            output_ids, stats = model.generate_with_hard_pruning(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs.get('attention_mask'),
                vision_start=preprocessed['vision_start'],
                vision_end=preprocessed['vision_end'],
                question_starts=preprocessed['question_starts'],
                question_ends=preprocessed['question_ends'],
                max_new_tokens=32,
            )

            # 收集保留率统计
            if 'avg_kept_ratio' in stats:
                kept_ratios.append(stats['avg_kept_ratio'])
            # 收集每层的保留率和 n_kept
            for key, value in stats.items():
                if key.startswith('L') and '_kept' in key:
                    layer_idx = int(key[1:].split('_')[0])
                    if key.endswith('_n_kept'):
                        # 绝对保留数量
                        if f'{layer_idx}_n_kept' not in layer_kept_ratios:
                            layer_kept_ratios[f'{layer_idx}_n_kept'] = []
                        layer_kept_ratios[f'{layer_idx}_n_kept'].append(value)
                    elif key == f'L{layer_idx}_kept':
                        # 保留率（排除 _n_kept）
                        if layer_idx not in layer_kept_ratios:
                            layer_kept_ratios[layer_idx] = []
                        layer_kept_ratios[layer_idx].append(value)
        else:
            # 不剪枝的生成（baseline）
            prompt = f"USER: <image>\n{sample['question']}\nASSISTANT:"
            inputs = processor(
                text=prompt,
                images=sample['image'],
                return_tensors="pt",
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        # 解码
        generated = processor.decode(output_ids[0], skip_special_tokens=True)

        # 提取答案（ASSISTANT: 之后的部分）
        if "ASSISTANT:" in generated:
            pred = generated.split("ASSISTANT:")[-1].strip()
        else:
            pred = generated.strip()

        predictions.append(pred)

        # 参考答案
        if 'answers' in sample:
            references.append(sample['answers'])
        else:
            references.append(sample['answer'])

    # 使用 judge 评估
    result = judge(predictions, references)

    eval_result = {
        'accuracy': result['accuracy'],
        'correct': result['correct'],
        'total': result['total'],
        'mode': mode,
    }

    # 添加平均保留率
    if kept_ratios:
        eval_result['avg_kept_ratio'] = sum(kept_ratios) / len(kept_ratios)

    # 添加每层的平均保留率和 n_kept
    for key, values in layer_kept_ratios.items():
        if isinstance(key, int):
            # 保留率
            eval_result[f'L{key}_kept'] = sum(values) / len(values)
        elif isinstance(key, str) and key.endswith('_n_kept'):
            # 绝对保留数量
            layer_idx = key.split('_')[0]
            eval_result[f'L{layer_idx}_n_kept'] = sum(values) / len(values)

    return eval_result


# ============================================================
# 主训练循环
# ============================================================

def train(config):
    """主训练函数"""
    logger = config.logger
    method_cfg = config.method_settings

    # 获取剪枝层配置
    pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])

    # 启用 anomaly detection 来定位 inplace 操作
    torch.autograd.set_detect_anomaly(True)

    # 设置
    device = torch.device(config.global_settings.get('device', 'cuda'))
    seed = config.global_settings.get('seed', 42)
    torch.manual_seed(seed)

    # 加载模型
    model, processor = load_model(config, device)

    # 加载数据
    logger.info("Loading dataset...")
    from engine.datas.loader import load_dataset
    data_bundle = load_dataset(config)

    train_dataset = data_bundle['splits']['train']
    test_dataset = data_bundle['splits'].get('test', None)
    judge = data_bundle['judge']

    dataset_name = config.dataset_settings.get('name', 'unknown')
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Train samples: {len(train_dataset)}")
    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")

    # 创建优化器
    trainer_cfg = config.trainer_settings.get('dl_settings', {})
    opt_cfg = trainer_cfg.get('optimizers', {})

    pruner_lr = opt_cfg.get('layer_pruners', {}).get('lr', 1e-4)
    disc_lr = opt_cfg.get('discriminator', {}).get('lr', 1.5e-4)

    pruner_optimizer = torch.optim.Adam(model.get_pruner_parameters(), lr=pruner_lr)
    disc_optimizer = torch.optim.Adam(model.get_discriminator_parameters(), lr=disc_lr)

    # 训练参数
    epochs = trainer_cfg.get('epochs', 1)
    batch_size = trainer_cfg.get('batch_size', 4)
    print_every = trainer_cfg.get('print_loss_every_batches', 50)
    eval_every = trainer_cfg.get('eval_every_batches', 1000)
    eval_max_samples = trainer_cfg.get('eval_max_samples', 500)
    save_every = trainer_cfg.get('save_every_batches', 3000)
    grad_clip = trainer_cfg.get('grad_clip_max_norm', None)

    # 计算总步数
    total_batches = (len(train_dataset) + batch_size - 1) // batch_size
    total_steps = epochs * total_batches

    logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, total_batches={total_batches}")
    logger.info(f"Total steps: {total_steps}, Pruner LR: {pruner_lr}, Disc LR: {disc_lr}")

    # 保存目录
    save_dir = Path(config.global_settings.get('save_dir', './outputs/checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    global_step = 0
    cached_origin_result = None  # 缓存 origin 评估结果（只计算一次）

    for epoch in range(epochs):
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*60}")

        # 打乱数据
        indices = torch.randperm(len(train_dataset)).tolist()

        epoch_losses = defaultdict(float)
        epoch_stats = defaultdict(float)
        n_batches = 0

        for batch_start in tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [train_dataset[i] for i in batch_indices]

            # 训练步骤
            result = train_step(
                batch=batch,
                model=model,
                processor=processor,
                config=config,
                current_step=global_step,
                total_steps=total_steps,
                device=device,
            )

            losses = result['losses']
            stats = result['stats']

            # === 先 backward 所有 loss，再 step ===
            # （因为 adv_loss 需要通过 discriminator 计算梯度，
            #  如果先 step discriminator 会导致计算图不一致）

            # 1. Discriminator backward
            disc_optimizer.zero_grad()
            disc_has_grad = 'disc_loss' in losses and losses['disc_loss'].requires_grad
            if disc_has_grad:
                losses['disc_loss'].backward(retain_graph=True)

            # 2. Pruner backward
            pruner_optimizer.zero_grad()
            pruner_total = sum(v for k, v in losses.items() if k != 'disc_loss')
            pruner_has_grad = pruner_total.requires_grad
            if pruner_has_grad:
                pruner_total.backward()

            # 3. Clip gradients and step
            if disc_has_grad:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.get_discriminator_parameters(), grad_clip)
                disc_optimizer.step()

            if pruner_has_grad:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.get_pruner_parameters(), grad_clip)
                pruner_optimizer.step()

            # 4. 判别器过强时重新初始化（每层单独判断）
            disc_reinit_threshold = method_cfg.get('disc_reinit_threshold', 0.85)
            reinited_layers = []
            for layer_idx in pruning_layers:
                acc_key = f'disc_L{layer_idx}_acc'
                if acc_key in stats and stats[acc_key] > disc_reinit_threshold:
                    model.disc_manager.reinit_layer(layer_idx)
                    reinited_layers.append(f"L{layer_idx}({stats[acc_key]:.2%})")
            if reinited_layers:
                # 重新初始化 optimizer 状态
                disc_optimizer = torch.optim.Adam(model.get_discriminator_parameters(), lr=disc_lr)
                logger.info(f"  [REINIT] Discriminators reinited: {', '.join(reinited_layers)} (threshold: {disc_reinit_threshold:.0%})")

            # 累计统计
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            for k, v in stats.items():
                epoch_stats[k] += v
            n_batches += 1
            global_step += 1

            # 打印
            if global_step % print_every == 0:
                avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
                avg_stats = {k: v / n_batches for k, v in epoch_stats.items()}

                loss_str = ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items())
                logger.info(f"Step {global_step}: {loss_str}")
                if 'avg_kept_ratio' in avg_stats:
                    # 打印每层的累积保留率（相对于原始 576 tokens）
                    layer_ratios = []
                    for layer_idx in pruning_layers:
                        cumulative_key = f'L{layer_idx}_cumulative'
                        kept_key = f'L{layer_idx}_kept'
                        if cumulative_key in avg_stats:
                            # 显示累积保留率
                            layer_ratios.append(f"L{layer_idx}={avg_stats[cumulative_key]:.2%}")
                        elif kept_key in avg_stats:
                            layer_ratios.append(f"L{layer_idx}={avg_stats[kept_key]:.2%}")
                    layer_str = ", ".join(layer_ratios)
                    logger.info(f"  Kept ratio: {avg_stats['avg_kept_ratio']:.2%} (target: {avg_stats['target_kept_ratio']:.2%}) [{layer_str}]")
                if 'disc_accuracy' in avg_stats:
                    # 打印每层判别器准确率
                    disc_layer_accs = []
                    for layer_idx in pruning_layers:
                        acc_key = f'disc_L{layer_idx}_acc'
                        if acc_key in avg_stats:
                            disc_layer_accs.append(f"L{layer_idx}={avg_stats[acc_key]:.2%}")
                    disc_str = ", ".join(disc_layer_accs)
                    logger.info(f"  Disc acc: {avg_stats['disc_accuracy']:.2%} [{disc_str}]")

                # 打印梯度统计
                pruner_grad_norms = []
                disc_grad_norms = []
                for p in model.get_pruner_parameters():
                    if p.grad is not None:
                        pruner_grad_norms.append(p.grad.norm().item())
                for p in model.get_discriminator_parameters():
                    if p.grad is not None:
                        disc_grad_norms.append(p.grad.norm().item())

                if pruner_grad_norms:
                    logger.info(f"  Pruner grad: mean={sum(pruner_grad_norms)/len(pruner_grad_norms):.6f}, "
                               f"max={max(pruner_grad_norms):.6f}, min={min(pruner_grad_norms):.6f}")
                if disc_grad_norms:
                    logger.info(f"  Disc grad: mean={sum(disc_grad_norms)/len(disc_grad_norms):.6f}, "
                               f"max={max(disc_grad_norms):.6f}, min={min(disc_grad_norms):.6f}")

            # 评估
            if test_dataset and global_step % eval_every == 0:
                logger.info(f"Evaluating at step {global_step}...")

                # 评估两种模式
                eval_modes = config.evaluation_settings.get('eval_mode', ['origin', 'hard'])
                for eval_mode in eval_modes:
                    if eval_mode == 'origin':
                        # origin 只计算一次，后续使用缓存
                        if cached_origin_result is None:
                            eval_result = evaluate(
                                model, processor, test_dataset, judge, config, device,
                                max_samples=eval_max_samples,
                                mode=eval_mode
                            )
                            cached_origin_result = eval_result
                        else:
                            eval_result = cached_origin_result
                        logger.info(f"  [{eval_mode}] Accuracy: {eval_result['accuracy']:.2%} (cached)")
                    else:
                        eval_result = evaluate(
                            model, processor, test_dataset, judge, config, device,
                            max_samples=eval_max_samples,
                            mode=eval_mode
                        )
                        logger.info(f"  [{eval_mode}] Accuracy: {eval_result['accuracy']:.2%}")
                        if 'avg_kept_ratio' in eval_result:
                            # 打印每层保留率（相对于原始 576 tokens）
                            layer_ratios = []
                            for layer_idx in pruning_layers:
                                kept_key = f'L{layer_idx}_kept'
                                n_kept_key = f'L{layer_idx}_n_kept'
                                if kept_key in eval_result:
                                    if n_kept_key in eval_result:
                                        layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}({int(eval_result[n_kept_key])})")
                                    else:
                                        layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}")
                            layer_str = ", ".join(layer_ratios)
                            logger.info(f"  [{eval_mode}] Avg kept ratio: {eval_result['avg_kept_ratio']:.2%} [{layer_str}]")

                model.train()

            # 保存
            if global_step % save_every == 0:
                ckpt_path = save_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'pruner_state_dict': model.pruner_manager.state_dict(),
                    'disc_state_dict': model.disc_manager.state_dict(),
                    'pruner_optimizer': pruner_optimizer.state_dict(),
                    'disc_optimizer': disc_optimizer.state_dict(),
                }, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

        # Epoch 结束
        logger.info(f"Epoch {epoch + 1} completed.")

    # 最终保存
    final_path = save_dir / "checkpoint_final.pt"
    torch.save({
        'step': global_step,
        'pruner_state_dict': model.pruner_manager.state_dict(),
        'disc_state_dict': model.disc_manager.state_dict(),
    }, final_path)
    logger.info(f"Training completed. Final checkpoint saved to {final_path}")

    # 最终评估
    if test_dataset:
        logger.info("Final evaluation...")
        eval_modes = config.evaluation_settings.get('eval_mode', ['origin', 'hard'])
        for eval_mode in eval_modes:
            if eval_mode == 'origin' and cached_origin_result is not None:
                # 使用缓存的 origin 结果
                eval_result = cached_origin_result
                logger.info(f"[{eval_mode}] Final accuracy: {eval_result['accuracy']:.2%} (cached)")
            else:
                eval_result = evaluate(
                    model, processor, test_dataset, judge, config, device,
                    max_samples=eval_max_samples,
                    mode=eval_mode
                )
                logger.info(f"[{eval_mode}] Final accuracy: {eval_result['accuracy']:.2%}")
                if 'avg_kept_ratio' in eval_result:
                    # 打印每层保留率（相对于原始 576 tokens）
                    layer_ratios = []
                    for layer_idx in pruning_layers:
                        kept_key = f'L{layer_idx}_kept'
                        n_kept_key = f'L{layer_idx}_n_kept'
                        if kept_key in eval_result:
                            if n_kept_key in eval_result:
                                layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}({int(eval_result[n_kept_key])})")
                            else:
                                layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}")
                    layer_str = ", ".join(layer_ratios)
                    logger.info(f"[{eval_mode}] Avg kept ratio: {eval_result['avg_kept_ratio']:.2%} [{layer_str}]")


# ============================================================
# 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Consistency Pruning Training")
    parser.add_argument('--config', type=str, default='configs/vision_token_pruning.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    print("=" * 60)
    print("Attention Consistency Pruning - Training")
    print("=" * 60)

    # 加载配置（使用统一的配置加载器）
    config = load_config(override_file=args.config)

    # 日志记录
    logger = config.logger
    logger.info("Starting Attention Consistency Pruning training...")

    # 训练
    train(config)


if __name__ == "__main__":
    main()
