#!/usr/bin/env python
"""Attention Consistency Pruning - DDP 分布式训练脚本

使用 PyTorch DistributedDataParallel (DDP) 进行多卡训练。

启动方式：
    torchrun --nproc_per_node=4 main_acp_ddp.py --config configs/vision_token_pruning.yaml

特点：
1. 使用 torchrun 启动，自动设置分布式环境
2. 只对可训练模块使用 DDP（base_model 冻结，无需 DDP）
3. 使用 DistributedSampler 进行数据分发
4. 只在 rank 0 进行日志记录和模型保存
"""

import os
import sys
import math

# 不要硬编码 CUDA_VISIBLE_DEVICES，让 torchrun 自动处理
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置加载器
from engine.configs.loader import load_config


# ============================================================
# 分布式工具函数
# ============================================================

def setup_distributed():
    """初始化分布式环境

    使用 torchrun 启动时，环境变量会自动设置
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否是主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """在所有进程间平均 tensor"""
    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor


# ============================================================
# 模型加载与预处理
# ============================================================

def load_model(config, device: torch.device, local_rank: int):
    """加载可剪枝的 LLaVA 模型（DDP 兼容版本）

    关键改动：
    1. 不使用 device_map='auto'，手动放置到指定 device
    2. 返回可训练模块列表用于 DDP 包装
    """
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    logger = config.logger if is_main_process() else None
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
    if logger:
        logger.info(f"Using dtype: {dtype_str} -> {torch_dtype}")

    # 获取剪枝层配置
    pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])
    pruner_d_internal = method_cfg.get('pruner_d_internal', 128)
    pruner_n_heads = method_cfg.get('pruner_n_heads', 4)
    disc_d_hidden = method_cfg.get('disc_d_d', 256)
    temperature = method_cfg.get('temperature', 1.0)
    dropout = method_cfg.get('pruner_dropout', 0.1)
    disc_spectral_norm = method_cfg.get('disc_use_spectral_norm', False)

    # Gumbel mode: 'always', 'never', 'hybrid'
    # 初始化时根据 mode 设置 use_gumbel_noise
    gumbel_mode = method_cfg.get('gumbel_mode', 'never')
    if gumbel_mode == 'always':
        use_gumbel_noise = True
    elif gumbel_mode == 'hybrid':
        # hybrid 模式初始时使用 noise（阶段1）
        use_gumbel_noise = True
    else:
        use_gumbel_noise = False

    # Adapter 配置
    adapter_type = method_cfg.get('adapter_type', 'lightweight')
    adapter_bottleneck = method_cfg.get('adapter_bottleneck', None)
    use_separated_adapters = method_cfg.get('use_separated_adapters', False)
    vision_adapter_bottleneck = method_cfg.get('vision_adapter_bottleneck', 256)
    text_adapter_bottleneck = method_cfg.get('text_adapter_bottleneck', 256)
    generator_adapter_bottleneck = method_cfg.get('generator_adapter_bottleneck', 512)

    # 剪枝阈值（sigmoid 后的阈值，用于训练第三阶段和推理）
    pruning_threshold = method_cfg.get('pruning_threshold', 0.5)

    # 模型路径
    model_name = backbone_cfg.get('name', 'llava-1.5-7b')
    model_mapping = {
        'llava-1.5-7b': 'llava-hf/llava-1.5-7b-hf',
        'llava-1.5-13b': 'llava-hf/llava-1.5-13b-hf',
    }
    model_path = model_mapping.get(model_name, model_name)

    if logger:
        logger.info(f"Loading base model from {model_path}...")

    # 加载基础模型 - 不使用 device_map='auto'
    # 先加载到 CPU，然后手动移动到指定 device
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        # 不使用 device_map，让模型先加载到 CPU
        device_map=None,
        low_cpu_mem_usage=True,
    )

    # 手动移动到指定设备
    base_model = base_model.to(device)

    processor = AutoProcessor.from_pretrained(model_path)

    # 设置 padding side 为 right
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
        disc_d_hidden=disc_d_hidden,
        adapter_bottleneck=adapter_bottleneck,
        adapter_type=adapter_type,
        use_separated_adapters=use_separated_adapters,
        vision_adapter_bottleneck=vision_adapter_bottleneck,
        text_adapter_bottleneck=text_adapter_bottleneck,
        generator_adapter_bottleneck=generator_adapter_bottleneck,
        temperature=temperature,
        dropout=dropout,
        disc_use_spectral_norm=disc_spectral_norm,
        use_gumbel_noise=use_gumbel_noise,
        pruning_threshold=pruning_threshold,
    )

    # 冻结基础模型
    model.freeze_base_model()

    if logger:
        logger.info(f"Model loaded. Pruning layers: {pruning_layers}, gumbel_mode: {gumbel_mode}")
        logger.info(f"Trainable parameters: Pruners={sum(p.numel() for p in model.get_pruner_parameters()):,}, "
                   f"Adapters={sum(p.numel() for p in model.get_adapter_parameters()):,}, "
                   f"Discriminators={sum(p.numel() for p in model.get_discriminator_parameters()):,}")

    return model, processor


def sync_gradients(model):
    """手动同步所有可训练参数的梯度

    由于我们的模型结构特殊（冻结主干 + 可训练小模块），
    使用手动梯度同步比 DDP 更灵活。

    这个函数会对所有有梯度的参数执行 all_reduce 平均。
    """
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    # 收集所有需要同步的梯度
    grads = []
    for param in model.get_pruner_parameters():
        if param.grad is not None:
            grads.append(param.grad.data)
    for param in model.get_adapter_parameters():
        if param.grad is not None:
            grads.append(param.grad.data)
    for param in model.get_discriminator_parameters():
        if param.grad is not None:
            grads.append(param.grad.data)

    # 合并成一个大 tensor 以减少通信开销
    if grads:
        # 扁平化所有梯度
        flat_grads = torch.cat([g.flatten() for g in grads])

        # All-reduce（求和后平均）
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads.div_(world_size)

        # 写回原始梯度
        offset = 0
        for grad in grads:
            numel = grad.numel()
            grad.copy_(flat_grads[offset:offset + numel].view_as(grad))
            offset += numel


def broadcast_model_params(model, src: int = 0):
    """从 src 进程广播模型参数到所有进程

    在训练开始前调用，确保所有进程的模型参数一致。
    """
    if not dist.is_initialized():
        return

    for param in model.get_pruner_parameters():
        dist.broadcast(param.data, src=src)
    for param in model.get_adapter_parameters():
        dist.broadcast(param.data, src=src)
    for param in model.get_discriminator_parameters():
        dist.broadcast(param.data, src=src)


def preprocess_batch(
    batch: List[Dict[str, Any]],
    processor,
    device: torch.device,
    max_length: int = 1024,
    mode: str = "train"
) -> Dict[str, Any]:
    """预处理一个 batch 的数据（与原版相同）"""
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


# ============================================================
# 损失计算
# ============================================================

def compute_task_loss(
    logits: torch.Tensor,
    answer_starts: List[int],
    answers: List[str],
    tokenizer,
    device: torch.device,
    debug: bool = False
) -> torch.Tensor:
    """计算 task loss (cross entropy)"""
    batch_size = logits.shape[0]
    total_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        # 不加空格前缀，因为 answer_starts 指向的是答案的第一个 token（空格在前面）
        answer = answers[i].capitalize()
        answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        if len(answer_ids) == 0:
            continue

        # 添加 EOS token，确保模型学会何时停止生成
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            answer_ids = answer_ids + [eos_token_id]

        pred_start = answer_starts[i] - 1
        pred_end = min(pred_start + len(answer_ids), logits.shape[1])

        if pred_start < 0 or pred_end <= pred_start:
            continue

        pred_logits = logits[i, pred_start:pred_end]
        target_len = min(len(answer_ids), pred_end - pred_start)
        targets = torch.tensor(answer_ids[:target_len], device=device)

        # Debug: 打印实际参与计算的 pred_logits 和 targets
        if debug and i == 0:
            pred_token_ids = pred_logits.argmax(dim=-1).tolist()
            target_ids = targets.tolist()
            print(f"[DEBUG task_loss] answer='{answer}'")
            print(f"[DEBUG task_loss] pred_start={pred_start}, pred_end={pred_end}, logits.shape={logits.shape}")
            print(f"[DEBUG task_loss] targets (answer_ids): {target_ids} -> '{tokenizer.decode(target_ids)}'")
            print(f"[DEBUG task_loss] pred_token_ids (argmax): {pred_token_ids} -> '{tokenizer.decode(pred_token_ids)}'")
            print(f"[DEBUG task_loss] match: {pred_token_ids == target_ids}")

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
    """执行一个训练步骤"""
    method_cfg = config.method_settings

    # === Gumbel Mode 三阶段调度 ===
    gumbel_mode = method_cfg.get('gumbel_mode', 'never')
    skip_phase1 = method_cfg.get('skip_phase1', False)
    progress = current_step / total_steps if total_steps > 0 else 0

    if gumbel_mode == 'hybrid':
        # 混合三阶段策略
        phase1_end = method_cfg.get('hybrid_phase1_end', 0.5)
        phase2_end = method_cfg.get('hybrid_phase2_end', 0.8)
        phase1_temp_start = method_cfg.get('hybrid_phase1_temp_start', 1.5)
        phase1_temp_end = method_cfg.get('hybrid_phase1_temp_end', 0.1)
        phase3_temp = method_cfg.get('hybrid_phase3_temp', 0.1)

        if skip_phase1:
            # 跳过阶段1，直接从阶段2开始
            # 重新映射 progress：将 [0, 1] 映射到 [phase1_end, 1]
            # 这样 progress=0 对应原来的 phase1_end，progress=1 对应原来的 1
            effective_progress = phase1_end + progress * (1 - phase1_end)
            if effective_progress < phase2_end:
                # 阶段2：稳定期 - 低温 + Gumbel noise
                current_temp = phase1_temp_end
                use_gumbel_noise = True
                current_phase = 2
            else:
                # 阶段3：确定性微调期 - 关闭 noise，训练=推理
                current_temp = phase3_temp
                use_gumbel_noise = False
                current_phase = 3
        else:
            # 正常三阶段
            if progress < phase1_end:
                # 阶段1：探索期 - 温度退火 + Gumbel noise
                phase_progress = progress / phase1_end
                current_temp = phase1_temp_start - phase_progress * (phase1_temp_start - phase1_temp_end)
                use_gumbel_noise = True
                current_phase = 1
            elif progress < phase2_end:
                # 阶段2：稳定期 - 低温 + Gumbel noise
                current_temp = phase1_temp_end
                use_gumbel_noise = True
                current_phase = 2
            else:
                # 阶段3：确定性微调期 - 关闭 noise，训练=推理
                current_temp = phase3_temp
                use_gumbel_noise = False
                current_phase = 3

        model.set_use_gumbel_noise(use_gumbel_noise)
    elif gumbel_mode == 'always':
        # 始终使用 Gumbel noise（旧的温度退火逻辑）
        temperature = method_cfg.get('temperature', 1.0)
        temperature_min = method_cfg.get('temperature_min', 0.5)
        anneal_rate = method_cfg.get('temperature_anneal_rate', 0.4)

        if progress < anneal_rate:
            current_temp = temperature - (progress / anneal_rate) * (temperature - temperature_min)
        else:
            current_temp = temperature_min
        use_gumbel_noise = True
        current_phase = 0
        model.set_use_gumbel_noise(True)
    else:
        # never: 纯 STE，不使用 Gumbel noise
        current_temp = method_cfg.get('temperature_min', 0.1)
        use_gumbel_noise = False
        current_phase = 0
        model.set_use_gumbel_noise(False)

    model.set_temperature(current_temp)

    # === 预处理 ===
    max_length = config.trainer_settings.get('dl_settings', {}).get('max_length', 2048)
    prep = preprocess_batch(batch, processor, device, max_length=max_length)
    inputs = prep['inputs']

    # === Forward ===
    model.train()

    # 是否阻止 adv_loss 梯度流向 pruner
    detach_adv_from_pruner = method_cfg.get('detach_adv_from_pruner', False)

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
        detach_h_fake_for_adv=detach_adv_from_pruner,
    )

    # === 计算 Losses ===
    losses = {}
    stats = {
        'temperature': current_temp,
        'use_gumbel_noise': use_gumbel_noise,
    }
    if gumbel_mode == 'hybrid':
        stats['hybrid_phase'] = current_phase

    # 1. Task Loss（不做物理删除，位置不变，直接使用原始 answer_starts）
    task_loss = compute_task_loss(
        output.logits,
        prep['answer_starts'],
        prep['answers'],
        processor.tokenizer,
        device,
        debug=(current_step <= 3)  # 前几步打印调试信息
    )
    losses['task_loss'] = task_loss
    stats['raw_task_loss'] = task_loss.item()

    # 2. 如果有剪枝信息，计算 GAN 相关 losses
    if output.pruning_infos and len(output.pruning_infos) > 0:
        h_real_dict = {idx: info['h_real'] for idx, info in output.pruning_infos.items()}
        h_fake_dict = {idx: info['h_fake'] for idx, info in output.pruning_infos.items()}

        loss_type = method_cfg.get('disc_loss_type', 'bce')
        gp_weight = method_cfg.get('disc_gp_weight', 10.0)

        warmup_ratio = method_cfg.get('pruner_warmup_ratio', 0.0)
        in_warmup = current_step < total_steps * warmup_ratio
        gan_weight = 0.0 if in_warmup else 1.0
        stats['in_warmup'] = in_warmup

        # 获取 disc_manager（可能被 DDP 包装）
        disc_manager = model.disc_manager.module if hasattr(model.disc_manager, 'module') else model.disc_manager

        adv_loss = disc_manager.compute_adv_loss(h_fake_dict, loss_type=loss_type)
        losses['adv_loss'] = adv_loss * gan_weight
        stats['raw_adv_loss'] = adv_loss.item()

        disc_loss = disc_manager.compute_disc_loss(h_real_dict, h_fake_dict, loss_type=loss_type, gp_weight=gp_weight)
        losses['disc_loss'] = disc_loss * gan_weight
        stats['raw_disc_loss'] = disc_loss.item()

        acc_info = disc_manager.compute_accuracy(h_real_dict, h_fake_dict)
        stats['disc_accuracy'] = acc_info['overall']
        stats['disc_real_acc'] = acc_info['real_acc']
        stats['disc_fake_acc'] = acc_info['fake_acc']
        stats['disc_per_layer'] = acc_info['per_layer']

        # Sparsity Loss
        target_token_num = method_cfg.get('target_token_num', 144)
        n_vision = prep['n_vision']
        final_target_ratio = target_token_num / n_vision

        # 剪枝目标退火：从 100% 保留逐渐退火到目标值
        # 如果 skip_phase1=True，跳过退火，直接使用目标值
        sparsity_anneal_ratio = method_cfg.get('sparsity_anneal_ratio', 0.0)
        if skip_phase1:
            # 跳过阶段1，直接使用目标稀疏度
            target_ratio = final_target_ratio
        elif sparsity_anneal_ratio > 0 and progress < sparsity_anneal_ratio:
            # 使用余弦退火：从 1.0 平滑过渡到 final_target_ratio
            anneal_progress = progress / sparsity_anneal_ratio
            # 余弦退火：(1 + cos(π * t)) / 2 从 1 到 0
            cosine_factor = (1 + math.cos(math.pi * anneal_progress)) / 2
            target_ratio = final_target_ratio + (1.0 - final_target_ratio) * cosine_factor
        else:
            target_ratio = final_target_ratio

        total_layers = len(model.base_model.language_model.layers)
        pruning_layers = sorted(output.pruning_infos.keys())
        n_pruning_layers = len(pruning_layers)

        # 获取 sparsity loss 模式
        sparsity_loss_mode = method_cfg.get('sparsity_loss_mode', 'exact')  # 'exact' 或 'harmonic'

        # Debug: 前几步打印 sparsity 计算详情
        debug_sparsity = (current_step <= 3)

        # 收集各层的累积保留率
        cumulative_ratios = []
        for layer_idx in pruning_layers:
            cumulative_mask = output.pruning_infos[layer_idx]['cumulative_mask']
            # DEBUG: 检查 cumulative_mask 的值
            if debug_sparsity:
                print(f"[DEBUG sparsity L{layer_idx}] cumulative_mask: min={cumulative_mask.min().item():.4f}, max={cumulative_mask.max().item():.4f}, mean={cumulative_mask.mean().item():.4f}")
                if torch.isnan(cumulative_mask).any() or torch.isinf(cumulative_mask).any():
                    print(f"  WARNING: cumulative_mask has NaN/Inf!")
            cumulative_ratio = cumulative_mask.float().mean()
            cumulative_ratios.append(cumulative_ratio)
            stats[f'L{layer_idx}_kept'] = cumulative_ratio.item()

        # 计算独立保留率 p_i = 当前层的 current_mask 的平均值
        # 不再使用除法 cumulative_r_i / cumulative_r_{i-1}，避免梯度计算问题
        independent_ratios = []
        for i, layer_idx in enumerate(pruning_layers):
            current_mask = output.pruning_infos[layer_idx].get('current_mask')
            if current_mask is None:
                # 向后兼容：如果没有 current_mask，使用除法计算
                if i == 0:
                    p_i = cumulative_ratios[i]
                else:
                    prev_cum = cumulative_ratios[i - 1].clamp(min=1e-6)
                    p_i = cumulative_ratios[i] / prev_cum
            else:
                p_i = current_mask.float().mean()
            p_i = p_i.clamp(min=1e-6, max=1.0)
            independent_ratios.append(p_i)
            if debug_sparsity:
                print(f"[DEBUG sparsity] p_{i} (L{layer_idx}) = {p_i.item():.4f}")

        # 计算各段的层数 [n0, n1, n2, n3]
        n_segments = []
        for i, layer_idx in enumerate(pruning_layers):
            if i == 0:
                n_segments.append(layer_idx)  # n0: 剪枝前的层数
            if i < n_pruning_layers - 1:
                n_segments.append(pruning_layers[i + 1] - layer_idx)
            else:
                n_segments.append(total_layers - layer_idx)

        if sparsity_loss_mode == 'exact':
            # === 精确加权平均方案 ===
            # avg = (n0*1 + n1*p1 + n2*p1*p2 + n3*p1*p2*p3) / total_layers
            avg_kept = torch.tensor(0.0, device=device)
            avg_kept = avg_kept + n_segments[0] * 1.0  # 剪枝前的层，保留率=1
            cumulative_product = torch.tensor(1.0, device=device)
            for i in range(n_pruning_layers):
                cumulative_product = cumulative_product * independent_ratios[i]
                avg_kept = avg_kept + n_segments[i + 1] * cumulative_product
            avg_kept = avg_kept / total_layers
            sparsity_loss = torch.abs(avg_kept - target_ratio)

            if debug_sparsity:
                print(f"[DEBUG sparsity] mode=exact, n_vision={n_vision}, target_ratio={target_ratio:.4f}")
                print(f"[DEBUG sparsity] n_segments={n_segments}, independent_ratios={[r.item() for r in independent_ratios]}")
                print(f"[DEBUG sparsity] avg_kept={avg_kept.item():.4f}, sparsity_loss={sparsity_loss.item():.4f}")

        else:  # harmonic
            # === 调和平均近似方案 ===
            # hm = n / Σ(1/p_i)
            inv_sum = sum(1.0 / p for p in independent_ratios)
            hm = n_pruning_layers / inv_sum

            # avg_approx = (n0*1 + n1*hm + n2*hm^2 + n3*hm^3) / total_layers
            avg_approx = torch.tensor(0.0, device=device)
            avg_approx = avg_approx + n_segments[0] * 1.0
            hm_power = hm
            for i in range(1, len(n_segments)):
                avg_approx = avg_approx + n_segments[i] * hm_power
                hm_power = hm_power * hm
            avg_approx = avg_approx / total_layers
            sparsity_loss = torch.abs(avg_approx - target_ratio)

            if debug_sparsity:
                print(f"[DEBUG sparsity] mode=harmonic, n_vision={n_vision}, target_ratio={target_ratio:.4f}")
                print(f"[DEBUG sparsity] n_segments={n_segments}, independent_ratios={[r.item() for r in independent_ratios]}")
                print(f"[DEBUG sparsity] hm={hm.item():.4f}, avg_approx={avg_approx.item():.4f}, sparsity_loss={sparsity_loss.item():.4f}")

            stats['harmonic_mean'] = hm.item()

        # DEBUG: 启用 sparsity_loss 来排查 NaN 梯度来源
        losses['sparsity_loss'] = sparsity_loss
        stats['raw_sparsity_loss'] = sparsity_loss.item()
        # 显示平均每层保留率（与 sparsity loss 约束的目标一致）
        if sparsity_loss_mode == 'exact':
            stats['avg_kept_ratio'] = avg_kept.item()
        else:
            stats['avg_kept_ratio'] = avg_approx.item()
        stats['final_kept_ratio'] = cumulative_ratios[-1].item()  # 最后一层的累积保留率
        stats['target_kept_ratio'] = target_ratio
        stats['total_layers'] = total_layers

    # === Per-Pruner Tightening Loss: 惩罚每个 pruner 的保留率 ===
    # 使每个 pruner 独立地倾向于剪掉更多 tokens
    # tightening_weights 可以是单个值或 list（对应每个剪枝层）
    tightening_weights_cfg = method_cfg.get('tightening_weights', [])
    if tightening_weights_cfg and output.pruning_infos:
        pruning_layers_sorted = sorted(output.pruning_infos.keys())

        # 如果是单个值，扩展为 list
        if isinstance(tightening_weights_cfg, (int, float)):
            tightening_weights_cfg = [tightening_weights_cfg] * len(pruning_layers_sorted)

        tightening_loss_total = torch.tensor(0.0, device=device)
        for i, layer_idx in enumerate(pruning_layers_sorted):
            # 获取该层的权重
            if i < len(tightening_weights_cfg):
                layer_weight = tightening_weights_cfg[i]
            else:
                layer_weight = 0.0

            if layer_weight <= 0:
                continue

            # 使用当前层的 mask（不是累积 mask）
            current_mask = output.pruning_infos[layer_idx].get('current_mask')
            if current_mask is None:
                current_mask = output.pruning_infos[layer_idx].get('cumulative_mask')
            if current_mask is not None:
                layer_kept_ratio = current_mask.float().mean()
                tightening_loss_total = tightening_loss_total + layer_weight * layer_kept_ratio
                stats[f'L{layer_idx}_tightening'] = layer_kept_ratio.item()

        if tightening_loss_total > 0:
            losses['tightening_loss'] = tightening_loss_total
            stats['tightening_loss'] = tightening_loss_total.item()

    # === Entropy 正则损失：鼓励 logits 生成极端值 ===
    # 最小化 entropy 会让 sigmoid(logits) 接近 0 或 1，使训练和推理行为一致
    entropy_weight = method_cfg.get('entropy_weight', 0.0)
    if entropy_weight > 0 and output.pruning_infos:
        entropy_losses = []
        for layer_idx in output.pruning_infos:
            keep_logits = output.pruning_infos[layer_idx].get('keep_logits')
            if keep_logits is not None:
                # p = sigmoid(keep_logits)
                p = torch.sigmoid(keep_logits.float())
                # entropy = -p * log(p) - (1-p) * log(1-p)
                # 使用 clamp 避免 log(0)
                p_clamped = p.clamp(min=1e-7, max=1-1e-7)
                entropy = -p_clamped * torch.log(p_clamped) - (1 - p_clamped) * torch.log(1 - p_clamped)
                entropy_losses.append(entropy.mean())
        if entropy_losses:
            entropy_loss = torch.stack(entropy_losses).mean()
            losses['entropy_loss'] = entropy_loss
            stats['entropy_loss'] = entropy_loss.item()

    # === 应用权重 ===
    task_weight = method_cfg.get('task_loss_weight', 1.0)
    adv_weight = method_cfg.get('adv_loss_weight', 0.5)
    sparsity_weight = method_cfg.get('sparsity_weight', 0.2)

    warmup_ratio = method_cfg.get('loss_weight_warmup_ratio', 0.0)
    if warmup_ratio > 0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2

        task_weight_start = method_cfg.get('task_loss_weight_start', task_weight)
        adv_weight_start = method_cfg.get('adv_loss_weight_start', adv_weight)

        task_weight = task_weight_start + (task_weight - task_weight_start) * cosine_factor.item()
        adv_weight = adv_weight_start + (adv_weight - adv_weight_start) * cosine_factor.item()

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

    weighted_losses = {
        'task_loss': losses['task_loss'] * task_weight,
    }
    if 'adv_loss' in losses:
        weighted_losses['adv_loss'] = losses['adv_loss'] * adv_weight
    if 'sparsity_loss' in losses:
        weighted_losses['sparsity_loss'] = losses['sparsity_loss'] * sparsity_weight
    if 'entropy_loss' in losses:
        weighted_losses['entropy_loss'] = losses['entropy_loss'] * entropy_weight
    if 'tightening_loss' in losses:
        weighted_losses['tightening_loss'] = losses['tightening_loss']  # 权重已在计算时应用
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
    mode: str = "origin",
    distributed: bool = False,
    aggregate_judge=None,
    requires_aggregate_eval: bool = False,
) -> Dict[str, float]:
    """评估模型

    Args:
        distributed: 是否使用分布式评估（所有 rank 参与）
        aggregate_judge: 聚合评估函数（用于 MME/GQA 等需要全量评估的数据集）
        requires_aggregate_eval: 是否需要聚合评估
    """
    model.eval()

    # 设置评估时的温度
    method_cfg = config.method_settings
    eval_temp = method_cfg.get('eval_temperature', method_cfg.get('temperature_min', 0.1))
    model.set_temperature(eval_temp)
    model.set_use_gumbel_noise(False)  # 评估时不使用 Gumbel noise

    # 获取 max_length 配置
    max_length = config.trainer_settings.get('dl_settings', {}).get('max_length', 2048)

    n_samples = min(len(dataset), max_samples)

    # 分布式评估：每个 rank 处理一部分数据
    if distributed and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        indices = list(range(n_samples))
        # 每个 rank 处理自己的分片
        local_indices = indices[rank::world_size]
    else:
        local_indices = list(range(n_samples))

    predictions = []
    references = []
    samples_for_aggregate = []  # 用于聚合评估
    kept_ratios = []
    layer_kept_ratios = {}

    pruning_layers = config.method_settings.get('pruning_layers', [4, 14, 24])
    desc = f"Evaluating ({mode})"

    # 只在主进程显示进度条
    show_progress = is_main_process()

    # 中间统计日志间隔（按全局步数计算）
    # 确保 local_log_interval 不超过每卡实际处理的样本数，否则日志永不触发
    log_interval = 200
    local_samples = len(local_indices)
    if distributed and dist.is_initialized():
        world_size = dist.get_world_size()
        # 每个 rank 处理 local_log_interval 个样本时，全局约处理 log_interval 个
        # 同时确保至少打印 4 次中间日志（如果样本数足够）
        local_log_interval = max(1, min(log_interval // world_size, local_samples // 4))
    else:
        world_size = 1
        local_log_interval = max(1, min(log_interval, local_samples // 4))

    for step_idx, i in enumerate(tqdm(local_indices, desc=desc, disable=not show_progress), start=1):
        sample = dataset[i]

        if mode == "hard":
            preprocessed = preprocess_batch(
                batch=[sample],
                processor=processor,
                device=device,
                max_length=max_length,
                mode="inference"
            )
            inputs = preprocessed['inputs']

            # Debug: 同时用训练路径计算保留率
            debug_train_ratios = {}
            if step_idx <= 5 and is_main_process():
                # 用训练路径（model()）计算保留率
                model.eval()
                with torch.no_grad():
                    output_train = model(
                        input_ids=inputs['input_ids'],
                        pixel_values=inputs['pixel_values'],
                        attention_mask=inputs['attention_mask'],
                        vision_start=preprocessed['vision_start'],
                        vision_end=preprocessed['vision_end'],
                        question_starts=preprocessed['question_starts'],
                        question_ends=preprocessed['question_ends'],
                        answer_starts=[preprocessed['question_ends'][0]],
                        answer_ends=[preprocessed['question_ends'][0] + 1],
                        return_pruning_info=True,
                    )
                for layer_idx in pruning_layers:
                    if layer_idx in output_train.pruning_infos:
                        cumulative_mask = output_train.pruning_infos[layer_idx]['cumulative_mask']
                        debug_train_ratios[layer_idx] = cumulative_mask.float().mean().item()

            output_ids, stats = model.generate_with_hard_pruning(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                attention_mask=inputs.get('attention_mask'),
                vision_start=preprocessed['vision_start'],
                vision_end=preprocessed['vision_end'],
                question_starts=preprocessed['question_starts'],
                question_ends=preprocessed['question_ends'],
                max_new_tokens=32,
                debug_generate=(step_idx <= 3 and is_main_process()),  # 前 3 个样本打印 debug
            )

            # Debug: 对比训练路径和推理路径的保留率
            if step_idx <= 5 and is_main_process() and debug_train_ratios:
                print(f"[Debug Eval {step_idx}] 训练路径 vs 推理路径:")
                for layer_idx in pruning_layers:
                    train_ratio = debug_train_ratios.get(layer_idx, 0)
                    infer_ratio = stats.get(f'L{layer_idx}_kept', 0)
                    diff = abs(train_ratio - infer_ratio)
                    print(f"  L{layer_idx}: train={train_ratio:.2%}, infer={infer_ratio:.2%}, diff={diff:.4f}")

            if 'avg_kept_ratio' in stats:
                kept_ratios.append(stats['avg_kept_ratio'])
            for key, value in stats.items():
                if key.startswith('L') and '_kept' in key:
                    layer_idx = int(key[1:].split('_')[0])
                    if key.endswith('_n_kept'):
                        if f'{layer_idx}_n_kept' not in layer_kept_ratios:
                            layer_kept_ratios[f'{layer_idx}_n_kept'] = []
                        layer_kept_ratios[f'{layer_idx}_n_kept'].append(value)
                    elif key == f'L{layer_idx}_kept':
                        if layer_idx not in layer_kept_ratios:
                            layer_kept_ratios[layer_idx] = []
                        layer_kept_ratios[layer_idx].append(value)
        else:
            prompt = f"USER: <image>\n{sample['question']}\nASSISTANT:"
            inputs = processor(
                text=prompt,
                images=sample['image'],
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        generated = processor.decode(output_ids[0], skip_special_tokens=True)

        if "ASSISTANT:" in generated:
            pred = generated.split("ASSISTANT:")[-1].strip()
        else:
            pred = generated.strip()

        predictions.append(pred)

        if 'answers' in sample:
            references.append(sample['answers'])
            gt = sample['answers']
        else:
            references.append(sample['answer'])
            gt = sample['answer']

        # 打印前 10 个样本的预测和 ground truth
        if step_idx <= 10 and is_main_process():
            print(f"[Eval {step_idx}] Pred: {pred!r} | GT: {gt!r}")

        # 聚合评估需要保留样本信息（只保留必要字段，不保留图像以避免显存累积）
        if requires_aggregate_eval:
            sample_info = {
                'answer': sample.get('answer'),
                'category': sample.get('category'),  # MME 需要
                'question_id': sample.get('question_id'),  # MME 需要（配对同图问题）
            }
            samples_for_aggregate.append(sample_info)

        # 每 local_log_interval 步打印中间统计
        if step_idx % local_log_interval == 0:
            if distributed and dist.is_initialized():
                # 分布式模式：收集所有 rank 的数据
                all_predictions = [None] * world_size
                all_references = [None] * world_size
                all_kept_ratios = [None] * world_size

                dist.all_gather_object(all_predictions, predictions)
                dist.all_gather_object(all_references, references)
                dist.all_gather_object(all_kept_ratios, kept_ratios)

                # 合并所有 rank 的数据
                merged_preds = []
                merged_refs = []
                merged_kept = []
                for p, r, k in zip(all_predictions, all_references, all_kept_ratios):
                    merged_preds.extend(p)
                    merged_refs.extend(r)
                    merged_kept.extend(k)

                if is_main_process():
                    interim_total = len(merged_preds)
                    if requires_aggregate_eval:
                        # 聚合评估模式：只打印进度和 kept ratio（无法增量计算 accuracy）
                        if merged_kept:
                            interim_kept = sum(merged_kept) / len(merged_kept)
                            print(f"\n[Step {interim_total}] Processed: {interim_total}, Kept: {interim_kept:.2%}")
                        else:
                            print(f"\n[Step {interim_total}] Processed: {interim_total}")
                    else:
                        # 普通模式：打印 accuracy
                        interim_result = judge(merged_preds, merged_refs)
                        interim_acc = interim_result['accuracy']
                        interim_correct = interim_result['correct']

                        if merged_kept:
                            interim_kept = sum(merged_kept) / len(merged_kept)
                            # 打印每层保留率
                            layer_str = ""
                            if layer_kept_ratios:
                                layer_parts = []
                                for layer_idx in sorted([k for k in layer_kept_ratios.keys() if isinstance(k, int)]):
                                    if layer_kept_ratios[layer_idx]:
                                        avg_ratio = sum(layer_kept_ratios[layer_idx]) / len(layer_kept_ratios[layer_idx])
                                        layer_parts.append(f"L{layer_idx}={avg_ratio:.2%}")
                                if layer_parts:
                                    layer_str = f" [{', '.join(layer_parts)}]"
                            print(f"\n[Step {interim_total}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total}), Kept: {interim_kept:.2%}{layer_str}")
                        else:
                            print(f"\n[Step {interim_total}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total})")
            else:
                # 单卡模式
                interim_total = len(predictions)
                if requires_aggregate_eval:
                    # 聚合评估模式：只打印进度和 kept ratio
                    if kept_ratios:
                        interim_kept = sum(kept_ratios) / len(kept_ratios)
                        print(f"\n[Step {interim_total}] Processed: {interim_total}, Kept: {interim_kept:.2%}")
                    else:
                        print(f"\n[Step {interim_total}] Processed: {interim_total}")
                else:
                    # 普通模式：打印 accuracy
                    interim_result = judge(predictions, references)
                    interim_acc = interim_result['accuracy']
                    interim_correct = interim_result['correct']

                    if kept_ratios:
                        interim_kept = sum(kept_ratios) / len(kept_ratios)
                        # 打印每层保留率
                        layer_str = ""
                        if layer_kept_ratios:
                            layer_parts = []
                            for layer_idx in sorted([k for k in layer_kept_ratios.keys() if isinstance(k, int)]):
                                if layer_kept_ratios[layer_idx]:
                                    avg_ratio = sum(layer_kept_ratios[layer_idx]) / len(layer_kept_ratios[layer_idx])
                                    layer_parts.append(f"L{layer_idx}={avg_ratio:.2%}")
                            if layer_parts:
                                layer_str = f" [{', '.join(layer_parts)}]"
                        print(f"\n[Step {step_idx}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total}), Kept: {interim_kept:.2%}{layer_str}")
                    else:
                        print(f"\n[Step {step_idx}] Acc: {interim_acc:.2%} ({interim_correct}/{interim_total})")

    # 分布式评估：收集所有 rank 的结果
    if distributed and dist.is_initialized():
        # 收集所有 rank 的 predictions 和 references
        all_predictions = [None] * dist.get_world_size()
        all_references = [None] * dist.get_world_size()
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_references, references)

        # 收集 kept_ratios
        all_kept_ratios = [None] * dist.get_world_size()
        dist.all_gather_object(all_kept_ratios, kept_ratios)

        # 收集 layer_kept_ratios
        all_layer_kept_ratios = [None] * dist.get_world_size()
        dist.all_gather_object(all_layer_kept_ratios, layer_kept_ratios)

        # 收集 samples_for_aggregate（如果需要聚合评估）
        if requires_aggregate_eval:
            all_samples = [None] * dist.get_world_size()
            dist.all_gather_object(all_samples, samples_for_aggregate)

        # 在所有 rank 上合并结果（保证一致性）
        predictions = []
        references = []
        kept_ratios = []
        merged_layer_kept_ratios = {}

        for preds, refs in zip(all_predictions, all_references):
            predictions.extend(preds)
            references.extend(refs)

        for ratios in all_kept_ratios:
            kept_ratios.extend(ratios)

        for layer_ratios in all_layer_kept_ratios:
            for key, values in layer_ratios.items():
                if key not in merged_layer_kept_ratios:
                    merged_layer_kept_ratios[key] = []
                merged_layer_kept_ratios[key].extend(values)

        layer_kept_ratios = merged_layer_kept_ratios

        # 合并 samples_for_aggregate
        if requires_aggregate_eval:
            samples_for_aggregate = []
            for samples in all_samples:
                samples_for_aggregate.extend(samples)

    # 根据是否需要聚合评估调用不同的 judge
    if requires_aggregate_eval and aggregate_judge is not None:
        result = aggregate_judge(predictions, references, samples_for_aggregate)
    else:
        result = judge(predictions, references)

    # 构建返回结果
    eval_result = {
        'mode': mode,
    }

    # 合并 judge 返回的所有字段
    eval_result.update(result)

    # 兼容旧接口：如果没有 accuracy 字段但有其他主指标，添加 accuracy 别名
    if 'accuracy' not in eval_result:
        if 'balanced_accuracy' in eval_result:
            eval_result['accuracy'] = eval_result['balanced_accuracy']
        elif 'total_score' in eval_result:
            # MME: 将 total_score 归一化为 0-1 范围作为 accuracy（假设满分 1400）
            eval_result['accuracy'] = eval_result['total_score'] / 1400.0

    if kept_ratios:
        eval_result['avg_kept_ratio'] = sum(kept_ratios) / len(kept_ratios)

    for key, values in layer_kept_ratios.items():
        if isinstance(key, int):
            eval_result[f'L{key}_kept'] = sum(values) / len(values)
        elif isinstance(key, str) and key.endswith('_n_kept'):
            layer_idx = key.split('_')[0]
            eval_result[f'L{layer_idx}_n_kept'] = sum(values) / len(values)

    return eval_result


# ============================================================
# 简单 Dataset 包装器
# ============================================================

class SimpleDataset(torch.utils.data.Dataset):
    """简单的 Dataset 包装器，用于支持 DataLoader"""
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """DataLoader 的 collate 函数，直接返回样本列表"""
    return batch


# ============================================================
# 主训练循环
# ============================================================

def train(config, rank: int, world_size: int, local_rank: int, device: torch.device):
    """主训练函数（分布式版本）"""
    logger = config.logger if is_main_process() else None
    method_cfg = config.method_settings

    pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])

    # 设置随机种子（每个进程使用不同的种子以获得不同的数据顺序）
    seed = config.global_settings.get('seed', 42)
    torch.manual_seed(seed + rank)

    # 加载模型
    model, processor = load_model(config, device, local_rank)

    # 广播模型参数，确保所有进程的初始参数一致
    broadcast_model_params(model, src=0)

    if is_main_process():
        logger.info("Model parameters broadcasted to all processes.")

    # 加载数据
    if is_main_process():
        logger.info("Loading dataset...")

    # 临时保存原始 logger，非主进程设置为 None 以避免重复日志
    original_logger = config.logger
    if not is_main_process():
        config.logger = None

    from engine.datas.loader import load_dataset
    data_bundle = load_dataset(config)

    # 恢复原始 logger
    config.logger = original_logger

    train_dataset = data_bundle['splits']['train']
    test_dataset = data_bundle['splits'].get('test', None)
    judge = data_bundle['judge']

    dataset_name = config.dataset_settings.get('name', 'unknown')
    if is_main_process():
        logger.info(f"Dataset: {dataset_name}")
        logger.info(f"Train samples: {len(train_dataset)}")
        if test_dataset:
            logger.info(f"Test samples: {len(test_dataset)}")

    # 创建 DistributedSampler 和 DataLoader
    train_wrapper = SimpleDataset(train_dataset)
    train_sampler = DistributedSampler(
        train_wrapper,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=seed,
    )

    trainer_cfg = config.trainer_settings.get('dl_settings', {})
    batch_size = trainer_cfg.get('batch_size', 4)

    train_loader = DataLoader(
        train_wrapper,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # 图像处理需要在主进程
        pin_memory=True,
    )

    # 创建优化器
    opt_cfg = trainer_cfg.get('optimizers', {})

    pruner_lr = opt_cfg.get('layer_pruners', {}).get('lr', 1e-4)
    pruner_weight_decay = opt_cfg.get('layer_pruners', {}).get('weight_decay', 0.0)
    disc_lr = opt_cfg.get('discriminator', {}).get('lr', 1.5e-4)

    from itertools import chain
    pruner_adapter_params = chain(model.get_pruner_parameters(), model.get_adapter_parameters())
    pruner_optimizer = torch.optim.Adam(pruner_adapter_params, lr=pruner_lr, weight_decay=pruner_weight_decay)
    disc_optimizer = torch.optim.Adam(model.get_discriminator_parameters(), lr=disc_lr)

    # 创建学习率调度器（余弦退火）
    # 计算总步数用于调度器
    epochs = trainer_cfg.get('epochs', 1)  # 提前获取 epochs
    total_batches_per_epoch = len(train_dataset) // batch_size
    total_steps_for_scheduler = epochs * total_batches_per_epoch

    lr_scheduler_cfg = opt_cfg.get('lr_scheduler', {})
    lr_scheduler_type = lr_scheduler_cfg.get('type', 'none')  # 'none', 'cosine', 'linear'
    warmup_ratio = lr_scheduler_cfg.get('warmup_ratio', 0.1)
    min_lr_ratio = lr_scheduler_cfg.get('min_lr_ratio', 0.1)  # 最小学习率 = 初始学习率 * min_lr_ratio

    pruner_scheduler = None
    disc_scheduler = None

    if lr_scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        warmup_steps = int(total_steps_for_scheduler * warmup_ratio)
        cosine_steps = total_steps_for_scheduler - warmup_steps

        # Pruner scheduler: warmup + cosine
        if warmup_steps > 0:
            pruner_warmup = LinearLR(pruner_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            pruner_cosine = CosineAnnealingLR(pruner_optimizer, T_max=cosine_steps, eta_min=pruner_lr * min_lr_ratio)
            pruner_scheduler = SequentialLR(pruner_optimizer, schedulers=[pruner_warmup, pruner_cosine], milestones=[warmup_steps])
        else:
            pruner_scheduler = CosineAnnealingLR(pruner_optimizer, T_max=total_steps_for_scheduler, eta_min=pruner_lr * min_lr_ratio)

        # Disc scheduler: warmup + cosine
        if warmup_steps > 0:
            disc_warmup = LinearLR(disc_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            disc_cosine = CosineAnnealingLR(disc_optimizer, T_max=cosine_steps, eta_min=disc_lr * min_lr_ratio)
            disc_scheduler = SequentialLR(disc_optimizer, schedulers=[disc_warmup, disc_cosine], milestones=[warmup_steps])
        else:
            disc_scheduler = CosineAnnealingLR(disc_optimizer, T_max=total_steps_for_scheduler, eta_min=disc_lr * min_lr_ratio)

        if is_main_process():
            logger.info(f"LR Scheduler: Cosine Annealing with warmup")
            logger.info(f"  Warmup steps: {warmup_steps} ({warmup_ratio:.0%})")
            logger.info(f"  Min LR ratio: {min_lr_ratio}")

    elif lr_scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LinearLR

        pruner_scheduler = LinearLR(pruner_optimizer, start_factor=1.0, end_factor=min_lr_ratio, total_iters=total_steps_for_scheduler)
        disc_scheduler = LinearLR(disc_optimizer, start_factor=1.0, end_factor=min_lr_ratio, total_iters=total_steps_for_scheduler)

        if is_main_process():
            logger.info(f"LR Scheduler: Linear Decay")
            logger.info(f"  Min LR ratio: {min_lr_ratio}")

    # 检查是否有 checkpoint 需要加载（用于恢复训练）
    checkpoint_path = config.global_settings.get('checkpoint', None)
    start_step = 0

    if checkpoint_path is not None:
        checkpoint_file = Path(checkpoint_path)
        if checkpoint_file.exists():
            if is_main_process():
                logger.info(f"Loading checkpoint from {checkpoint_path}...")

            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

            # 加载模型状态
            if 'pruner_state_dict' in checkpoint:
                model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
                if is_main_process():
                    logger.info("  Loaded pruner_manager state")

            if 'adapter_state_dict' in checkpoint and not model.use_separated_adapters:
                model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
                if is_main_process():
                    logger.info("  Loaded adapter_manager state")

            if 'separated_adapter_state_dict' in checkpoint and model.use_separated_adapters:
                model.separated_adapter_manager.load_state_dict(checkpoint['separated_adapter_state_dict'])
                if is_main_process():
                    logger.info("  Loaded separated_adapter_manager state")

            if 'disc_state_dict' in checkpoint:
                model.disc_manager.load_state_dict(checkpoint['disc_state_dict'])
                if is_main_process():
                    logger.info("  Loaded disc_manager state")

            # 加载优化器状态
            if 'pruner_optimizer' in checkpoint:
                pruner_optimizer.load_state_dict(checkpoint['pruner_optimizer'])
                if is_main_process():
                    logger.info("  Loaded pruner_optimizer state")

            if 'disc_optimizer' in checkpoint:
                disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
                if is_main_process():
                    logger.info("  Loaded disc_optimizer state")

            # 加载学习率调度器状态
            if 'pruner_scheduler' in checkpoint and pruner_scheduler is not None:
                pruner_scheduler.load_state_dict(checkpoint['pruner_scheduler'])
                if is_main_process():
                    logger.info("  Loaded pruner_scheduler state")

            if 'disc_scheduler' in checkpoint and disc_scheduler is not None:
                disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
                if is_main_process():
                    logger.info("  Loaded disc_scheduler state")

            # 恢复训练步数
            if 'step' in checkpoint:
                start_step = checkpoint['step']
                if is_main_process():
                    logger.info(f"  Resuming from step {start_step}")

            # 重新广播模型参数确保一致性
            broadcast_model_params(model, src=0)

            if is_main_process():
                logger.info("Checkpoint loaded successfully.")
        else:
            if is_main_process():
                logger.warning(f"Checkpoint file not found: {checkpoint_path}, starting from scratch")

    # 训练参数（epochs 已在上面获取）
    print_every = trainer_cfg.get('print_loss_every_batches', 50)
    eval_every = trainer_cfg.get('eval_every_batches', 1000)
    eval_max_samples = trainer_cfg.get('eval_max_samples', 500)
    save_every = trainer_cfg.get('save_every_batches', 3000)
    grad_clip = trainer_cfg.get('grad_clip_max_norm', None)

    # 计算总步数
    total_batches_per_epoch = len(train_loader)
    total_steps = epochs * total_batches_per_epoch

    if is_main_process():
        logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, "
                   f"batches_per_epoch={total_batches_per_epoch}")
        logger.info(f"Total steps: {total_steps}, Pruner LR: {pruner_lr}, Disc LR: {disc_lr}")
        logger.info(f"World size: {world_size}, Effective batch size: {batch_size * world_size}")
        # 显示 skip_phase1 状态
        skip_phase1 = method_cfg.get('skip_phase1', False)
        if skip_phase1:
            logger.info(f"[skip_phase1=True] Starting from Phase 2, skipping temperature and sparsity annealing")

    # 保存目录
    save_dir = Path(config.global_settings.get('save_dir', './outputs/checkpoints'))
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    global_step = start_step
    cached_origin_result = None

    # 统计每层的保留数量（用于推荐 topk_ks）
    layer_kept_counts = {idx: [] for idx in pruning_layers}  # {layer_idx: [n_kept_per_batch, ...]}

    # 阶段切换学习率缩放（用于 hybrid 模式）
    prev_phase = 0  # 跟踪上一步的阶段
    phase3_lr_scaled = False  # 是否已经缩放过阶段3的学习率
    phase3_lr_scale = lr_scheduler_cfg.get('phase3_lr_scale', 1.0)  # 阶段3学习率缩放系数

    for epoch in range(epochs):
        # 设置 epoch 以确保不同 epoch 的 shuffle 不同
        train_sampler.set_epoch(epoch)

        if is_main_process():
            logger.info(f"{'='*60}")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info(f"{'='*60}")

        epoch_losses = defaultdict(float)
        epoch_stats = defaultdict(float)
        n_batches = 0

        # 使用 tqdm 包装 DataLoader（只在主进程显示进度条）
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())

        for batch in pbar:
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

            # DEBUG: 打印 backward 前状态
            if os.environ.get('DEBUG_PRUNING', '0') == '1':
                print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
                      f"compute_losses_and_stats done, starting backward...")

            # === 先 backward 所有 loss，再同步梯度，再 step ===
            pruner_optimizer.zero_grad()
            pruner_total = sum(v for k, v in losses.items() if k != 'disc_loss')
            pruner_has_grad = pruner_total.requires_grad
            if pruner_has_grad:
                # DEBUG: 检查 loss 值
                if torch.isnan(pruner_total) or torch.isinf(pruner_total):
                    print(f"[DEBUG backward] pruner_total has NaN/Inf: {pruner_total.item()}")
                    for k, v in losses.items():
                        if k != 'disc_loss':
                            print(f"  {k}: {v.item()}, requires_grad={v.requires_grad}")
                if os.environ.get('DEBUG_PRUNING', '0') == '1':
                    print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
                          f"pruner_total.backward starting...")
                pruner_total.backward(retain_graph=True)
                if os.environ.get('DEBUG_PRUNING', '0') == '1':
                    print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
                          f"pruner_total.backward done")

            disc_optimizer.zero_grad()
            disc_has_grad = 'disc_loss' in losses and losses['disc_loss'].requires_grad
            if disc_has_grad:
                if os.environ.get('DEBUG_PRUNING', '0') == '1':
                    print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
                          f"disc_loss.backward starting...")
                losses['disc_loss'].backward()
                if os.environ.get('DEBUG_PRUNING', '0') == '1':
                    print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
                          f"disc_loss.backward done")

            # DEBUG: 打印梯度状态
            if os.environ.get('DEBUG_PRUNING', '0') == '1':
                n_pruner_grads = sum(1 for p in model.get_pruner_parameters() if p.grad is not None)
                n_adapter_grads = sum(1 for p in model.get_adapter_parameters() if p.grad is not None)
                n_disc_grads = sum(1 for p in model.get_discriminator_parameters() if p.grad is not None)
                print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] "
                      f"Before sync_gradients: pruner_grads={n_pruner_grads}, "
                      f"adapter_grads={n_adapter_grads}, disc_grads={n_disc_grads}")

            # === 同步梯度（关键步骤！）===
            sync_gradients(model)

            # DEBUG: 打印同步后状态
            if os.environ.get('DEBUG_PRUNING', '0') == '1':
                print(f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] sync_gradients done")

            if disc_has_grad:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.get_discriminator_parameters(), grad_clip)
                disc_optimizer.step()

            if pruner_has_grad:
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.get_pruner_parameters(), grad_clip)
                pruner_optimizer.step()

            # 学习率调度器步进
            if pruner_scheduler is not None:
                pruner_scheduler.step()
            if disc_scheduler is not None:
                disc_scheduler.step()

            # 阶段切换学习率缩放（hybrid 模式）
            current_phase = stats.get('hybrid_phase', 0)
            if current_phase == 3 and prev_phase != 3 and not phase3_lr_scaled and phase3_lr_scale < 1.0:
                # 进入阶段3，缩放学习率
                for param_group in pruner_optimizer.param_groups:
                    param_group['lr'] *= phase3_lr_scale
                for param_group in disc_optimizer.param_groups:
                    param_group['lr'] *= phase3_lr_scale
                phase3_lr_scaled = True
                if is_main_process():
                    new_pruner_lr = pruner_optimizer.param_groups[0]['lr']
                    new_disc_lr = disc_optimizer.param_groups[0]['lr']
                    logger.info(f"[Phase 3] LR scaled by {phase3_lr_scale}: pruner_lr={new_pruner_lr:.2e}, disc_lr={new_disc_lr:.2e}")
            prev_phase = current_phase

            # 判别器重新初始化
            # 注意：需要在所有进程间同步决策，避免死锁
            disc_reinit_enable = method_cfg.get('disc_reinit_enable', True)
            disc_reinit_mode = method_cfg.get('disc_reinit_mode', 'threshold')  # 'threshold' 或 'random'
            disc_reinit_threshold = method_cfg.get('disc_reinit_threshold', 0.85)
            disc_reinit_prob = method_cfg.get('disc_reinit_prob', 0.01)  # 随机模式下的概率

            # random 模式：在循环外部生成随机数，确保所有 rank 都参与 broadcast
            # 为每层生成一个随机数
            if disc_reinit_enable and disc_reinit_mode == 'random':
                rand_tensors = {}
                for layer_idx in pruning_layers:
                    rand_tensor = torch.tensor(0.0, device=device)
                    if is_main_process():
                        rand_tensor = torch.rand(1, device=device)[0]
                    if dist.is_initialized():
                        dist.broadcast(rand_tensor, src=0)
                    rand_tensors[layer_idx] = rand_tensor.item()

            if disc_reinit_enable and 'disc_per_layer' in stats:
                for layer_idx, (real_acc, fake_acc) in stats['disc_per_layer'].items():
                    should_reinit = False
                    reinit_reason = ""

                    if disc_reinit_mode == 'threshold':
                        # 阈值模式：准确率过高时重初始化
                        layer_acc = (real_acc + fake_acc) / 2
                        # 汇总所有 rank 的 accuracy（取平均），确保所有进程做出相同决策
                        layer_acc_tensor = torch.tensor(layer_acc, device=device)
                        if dist.is_initialized():
                            dist.all_reduce(layer_acc_tensor, op=dist.ReduceOp.SUM)
                            layer_acc_tensor /= dist.get_world_size()
                        layer_acc_global = layer_acc_tensor.item()

                        if layer_acc_global > disc_reinit_threshold:
                            should_reinit = True
                            reinit_reason = f"acc={layer_acc_global:.2%} > {disc_reinit_threshold:.0%}"

                    elif disc_reinit_mode == 'random':
                        # 随机模式：使用预先生成的随机数
                        if rand_tensors.get(layer_idx, 1.0) < disc_reinit_prob:
                            should_reinit = True
                            reinit_reason = f"random (prob={disc_reinit_prob})"

                    if should_reinit:
                        model.disc_manager.reinit_layer(layer_idx)
                        # 重新初始化后，从 rank 0 广播新参数
                        broadcast_model_params(model, src=0)
                        if is_main_process():
                            logger.info(f"  [REINIT] Discriminator L{layer_idx} reinited ({reinit_reason})")

            # 统计每层保留的 token 数量（用于训练结束后推荐 topk_ks）
            if result['pruning_infos']:
                for layer_idx, info in result['pruning_infos'].items():
                    if 'cumulative_mask' in info:
                        # cumulative_mask: (batch, n_vision), 计算每个样本保留的 token 数量
                        cumulative_mask = info['cumulative_mask']
                        n_kept_per_sample = cumulative_mask.sum(dim=-1)  # (batch,)
                        layer_kept_counts[layer_idx].extend(n_kept_per_sample.tolist())

            # 累计统计
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            for k, v in stats.items():
                if k == 'disc_per_layer':
                    if k not in epoch_stats:
                        epoch_stats[k] = {}
                    for layer_idx, (real_acc, fake_acc) in v.items():
                        if layer_idx not in epoch_stats[k]:
                            epoch_stats[k][layer_idx] = [0, 0]
                        epoch_stats[k][layer_idx][0] += real_acc
                        epoch_stats[k][layer_idx][1] += fake_acc
                else:
                    epoch_stats[k] += v
            n_batches += 1
            global_step += 1

            # 打印（只在主进程）
            if global_step % print_every == 0 and is_main_process():
                loss_str = ", ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                # 显示阶段信息
                phase_str = ""
                if 'hybrid_phase' in stats:
                    phase_str = f" [Phase {stats['hybrid_phase']}]"
                noise_str = "noise=ON" if stats.get('use_gumbel_noise', False) else "noise=OFF"
                logger.info(f"Step {global_step}{phase_str}: {loss_str} (temp={stats['temperature']:.2f}, {noise_str})")

                if 'avg_kept_ratio' in stats:
                    layer_ratios = []
                    for layer_idx in pruning_layers:
                        cumulative_key = f'L{layer_idx}_cumulative'
                        kept_key = f'L{layer_idx}_kept'
                        if cumulative_key in stats:
                            layer_ratios.append(f"L{layer_idx}={stats[cumulative_key]:.2%}")
                        elif kept_key in stats:
                            layer_ratios.append(f"L{layer_idx}={stats[kept_key]:.2%}")
                    layer_str = ", ".join(layer_ratios)
                    logger.info(f"  Kept ratio: {stats['avg_kept_ratio']:.2%} "
                               f"(target: {stats['target_kept_ratio']:.2%}) [{layer_str}]")

                    # 在 Phase 3 时，训练和推理的保留率应该一致
                    # 直接使用 stats 中的 L{layer_idx}_kept 即可，不需要重新计算
                    # （物理删除模式下，每层的 keep_logits 大小不同，不能直接相乘）

                if 'disc_per_layer' in stats:
                    per_layer_strs = []
                    for layer_idx in sorted(stats['disc_per_layer'].keys()):
                        real_acc, fake_acc = stats['disc_per_layer'][layer_idx]
                        layer_acc = (real_acc + fake_acc) / 2
                        per_layer_strs.append(f"L{layer_idx}={layer_acc:.0%}(R{real_acc:.0%}/F{fake_acc:.0%})")
                    logger.info(f"  Disc acc: {stats['disc_accuracy']:.2%} [{', '.join(per_layer_strs)}]")

            # 分布式评估：所有 rank 都参与
            if test_dataset and global_step % eval_every == 0:
                if is_main_process():
                    logger.info(f"Evaluating at step {global_step}...")

                eval_modes = config.evaluation_settings.get('eval_mode', ['origin', 'hard'])
                for eval_mode in eval_modes:
                    if eval_mode == 'origin':
                        if cached_origin_result is None:
                            eval_result = evaluate(
                                model, processor, test_dataset, judge, config, device,
                                max_samples=eval_max_samples,
                                mode=eval_mode,
                                distributed=True
                            )
                            cached_origin_result = eval_result
                            if is_main_process():
                                logger.info(f"  [{eval_mode}] Accuracy: {eval_result['accuracy']:.2%}")
                        else:
                            eval_result = cached_origin_result
                            if is_main_process():
                                logger.info(f"  [{eval_mode}] Accuracy: {eval_result['accuracy']:.2%} (cached)")
                    else:
                        eval_result = evaluate(
                            model, processor, test_dataset, judge, config, device,
                            max_samples=eval_max_samples,
                            mode=eval_mode,
                            distributed=True
                        )
                        if is_main_process():
                            logger.info(f"  [{eval_mode}] Accuracy: {eval_result['accuracy']:.2%}")
                            if 'avg_kept_ratio' in eval_result:
                                layer_ratios = []
                                for layer_idx in pruning_layers:
                                    kept_key = f'L{layer_idx}_kept'
                                    n_kept_key = f'L{layer_idx}_n_kept'
                                    if kept_key in eval_result:
                                        if n_kept_key in eval_result:
                                            layer_ratios.append(
                                                f"L{layer_idx}={eval_result[kept_key]:.2%}"
                                                f"({int(eval_result[n_kept_key])})"
                                            )
                                        else:
                                            layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}")
                                layer_str = ", ".join(layer_ratios)
                                logger.info(f"  [{eval_mode}] Avg kept ratio: "
                                           f"{eval_result['avg_kept_ratio']:.2%} [{layer_str}]")

                model.train()

            # 保存（只在主进程）
            if global_step % save_every == 0 and is_main_process():
                ckpt_path = save_dir / f"checkpoint_step{global_step}.pt"
                ckpt_data = {
                    'step': global_step,
                    'pruner_state_dict': model.pruner_manager.state_dict(),
                    'disc_state_dict': model.disc_manager.state_dict(),
                    'pruner_optimizer': pruner_optimizer.state_dict(),
                    'disc_optimizer': disc_optimizer.state_dict(),
                }
                # 根据 adapter 类型保存
                if model.use_separated_adapters:
                    ckpt_data['separated_adapter_state_dict'] = model.separated_adapter_manager.state_dict()
                else:
                    ckpt_data['adapter_state_dict'] = model.adapter_manager.state_dict()
                if pruner_scheduler is not None:
                    ckpt_data['pruner_scheduler'] = pruner_scheduler.state_dict()
                if disc_scheduler is not None:
                    ckpt_data['disc_scheduler'] = disc_scheduler.state_dict()
                torch.save(ckpt_data, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            # 同步所有进程
            if dist.is_initialized():
                dist.barrier()

        if is_main_process():
            logger.info(f"Epoch {epoch + 1} completed.")

    # 最终保存（只在主进程）
    if is_main_process():
        final_path = save_dir / "checkpoint_final.pt"
        final_ckpt = {
            'step': global_step,
            'pruner_state_dict': model.pruner_manager.state_dict(),
            'disc_state_dict': model.disc_manager.state_dict(),
        }
        if model.use_separated_adapters:
            final_ckpt['separated_adapter_state_dict'] = model.separated_adapter_manager.state_dict()
        else:
            final_ckpt['adapter_state_dict'] = model.adapter_manager.state_dict()
        torch.save(final_ckpt, final_path)
        logger.info(f"Training completed. Final checkpoint saved to {final_path}")

    # 汇总训练期间每层的保留数量统计（分布式聚合）
    if dist.is_initialized():
        # 收集所有 rank 的 layer_kept_counts
        all_kept_counts = [None] * dist.get_world_size()
        dist.all_gather_object(all_kept_counts, layer_kept_counts)
        # 合并所有 rank 的统计
        merged_kept_counts = {idx: [] for idx in pruning_layers}
        for counts_dict in all_kept_counts:
            for layer_idx, counts in counts_dict.items():
                merged_kept_counts[layer_idx].extend(counts)
        layer_kept_counts = merged_kept_counts

    # 输出推荐的 topk_ks（只在主进程）
    if is_main_process() and any(len(counts) > 0 for counts in layer_kept_counts.values()):
        logger.info("=" * 60)
        logger.info("Training Statistics - Recommended topk_ks for inference:")
        logger.info("=" * 60)
        recommended_topk_ks = {}
        for layer_idx in pruning_layers:
            counts = layer_kept_counts[layer_idx]
            if counts:
                mean_kept = sum(counts) / len(counts)
                std_kept = (sum((x - mean_kept) ** 2 for x in counts) / len(counts)) ** 0.5
                min_kept = min(counts)
                max_kept = max(counts)
                recommended_k = int(round(mean_kept))
                recommended_topk_ks[layer_idx] = recommended_k
                logger.info(f"  Layer {layer_idx}: mean={mean_kept:.1f}, std={std_kept:.1f}, "
                           f"min={min_kept:.0f}, max={max_kept:.0f} -> recommended k={recommended_k}")
        logger.info("")
        logger.info("Add to config (pruner_topk_ks):")
        for layer_idx, k in recommended_topk_ks.items():
            logger.info(f"    {layer_idx}: {k}")
        logger.info("=" * 60)

    # 最终评估（所有 rank 都需要参与分布式评估）
    if test_dataset:
        if is_main_process():
            logger.info("Final evaluation...")
        eval_modes = config.evaluation_settings.get('eval_mode', ['origin', 'hard'])
        for eval_mode in eval_modes:
            if eval_mode == 'origin' and cached_origin_result is not None:
                eval_result = cached_origin_result
                if is_main_process():
                    logger.info(f"[{eval_mode}] Final accuracy: {eval_result['accuracy']:.2%} (cached)")
            else:
                eval_result = evaluate(
                    model, processor, test_dataset, judge, config, device,
                    max_samples=eval_max_samples,
                    mode=eval_mode,
                    distributed=True
                )
                if is_main_process():
                    logger.info(f"[{eval_mode}] Final accuracy: {eval_result['accuracy']:.2%}")
                    if 'avg_kept_ratio' in eval_result:
                        layer_ratios = []
                        for layer_idx in pruning_layers:
                            kept_key = f'L{layer_idx}_kept'
                            n_kept_key = f'L{layer_idx}_n_kept'
                            if kept_key in eval_result:
                                if n_kept_key in eval_result:
                                    layer_ratios.append(
                                        f"L{layer_idx}={eval_result[kept_key]:.2%}"
                                        f"({int(eval_result[n_kept_key])})"
                                    )
                                else:
                                    layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}")
                        layer_str = ", ".join(layer_ratios)
                        logger.info(f"[{eval_mode}] Avg kept ratio: "
                                   f"{eval_result['avg_kept_ratio']:.2%} [{layer_str}]")


# ============================================================
# 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Consistency Pruning Training (DDP)")
    parser.add_argument('--config', type=str, default='configs/vision_token_pruning.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    # 初始化分布式环境
    rank, world_size, local_rank, device = setup_distributed()

    if rank == 0:
        print("=" * 60)
        print("Attention Consistency Pruning - DDP Training")
        print(f"World size: {world_size}")
        print("=" * 60)

    try:
        # 加载配置
        config = load_config(override_file=args.config)

        # 非主进程：禁用 logger 以避免重复日志
        if rank != 0:
            config.logger = None

        if rank == 0:
            logger = config.logger
            logger.info("Starting Attention Consistency Pruning training (DDP)...")
            logger.info(f"Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")

        # 训练
        train(config, rank, world_size, local_rank, device)

    finally:
        # 清理
        cleanup_distributed()


if __name__ == "__main__":
    main()
