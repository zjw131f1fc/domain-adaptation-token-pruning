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
from contextlib import nullcontext

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

# 导入拆分后的模块
from engine.distributed import (
    setup_distributed, cleanup_distributed, is_main_process,
    reduce_mean, sync_gradients, broadcast_model_params
)
from engine.data_utils import preprocess_batch, SimpleDataset, collate_fn
from engine.train_utils import compute_task_loss, train_step
from engine.eval_utils import evaluate


# ============================================================
# 模型加载与预处理
# ============================================================

def load_model(config, device: torch.device, local_rank: int):
    """加载可剪枝的 MLLM 模型（DDP 兼容版本）

    支持的模型类型：
    - llava: LLaVA 1.5 7B/13B
    - qwen2_vl: Qwen2-VL 2B/7B

    关键改动：
    1. 不使用 device_map='auto'，手动放置到指定 device
    2. 返回可训练模块列表用于 DDP 包装
    3. 根据 backbone_settings.model_type 自动路由到对应模型
    """
    from transformers import AutoProcessor

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
    use_adapter = method_cfg.get('use_adapter', True)
    adapter_type = method_cfg.get('adapter_type', 'lightweight')
    adapter_bottleneck = method_cfg.get('adapter_bottleneck', None)
    adapter_dropout = method_cfg.get('adapter_dropout', 0.15)  # Adapter dropout
    use_separated_adapters = method_cfg.get('use_separated_adapters', False)
    vision_adapter_bottleneck = method_cfg.get('vision_adapter_bottleneck', 256)
    text_adapter_bottleneck = method_cfg.get('text_adapter_bottleneck', 256)
    generator_adapter_bottleneck = method_cfg.get('generator_adapter_bottleneck', 512)

    # Pruner query dropout
    pruner_query_dropout = method_cfg.get('pruner_query_dropout', 0.0)

    # 剪枝阈值（sigmoid 后的阈值，用于训练第三阶段和推理）
    pruning_threshold = method_cfg.get('pruning_threshold', 0.5)

    # 模型路径和类型
    model_name = backbone_cfg.get('name', 'llava-1.5-7b')
    model_type = backbone_cfg.get('model_type', 'llava')  # 从配置加载器自动设置

    model_mapping = {
        'llava-1.5-7b': 'llava-hf/llava-1.5-7b-hf',
        'llava-1.5-13b': 'llava-hf/llava-1.5-13b-hf',
        'qwen2-vl-2b': 'Qwen/Qwen2-VL-2B-Instruct',
        'qwen2-vl-7b': 'Qwen/Qwen2-VL-7B-Instruct',
    }
    model_path = model_mapping.get(model_name, model_name)

    if logger:
        logger.info(f"Loading base model from {model_path} (type: {model_type})...")

    # 根据模型类型加载不同的基础模型
    if model_type == 'qwen2_vl':
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )
    else:  # llava
        from transformers import LlavaForConditionalGeneration, AutoProcessor
        base_model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
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

    # 创建可剪枝模型（根据模型类型选择）
    if model_type == 'qwen2_vl':
        from method.models.prunable_qwen2vl import PrunableQwen2VLForConditionalGeneration
        model = PrunableQwen2VLForConditionalGeneration(
            base_model=base_model,
            pruning_layers=pruning_layers,
            pruner_d_internal=pruner_d_internal,
            pruner_n_heads=pruner_n_heads,
            pruner_query_dropout=pruner_query_dropout,
            disc_d_hidden=disc_d_hidden,
            adapter_bottleneck=adapter_bottleneck,
            adapter_type=adapter_type,
            use_separated_adapters=use_separated_adapters,
            vision_adapter_bottleneck=vision_adapter_bottleneck,
            text_adapter_bottleneck=text_adapter_bottleneck,
            generator_adapter_bottleneck=generator_adapter_bottleneck,
            temperature=temperature,
            dropout=dropout,
            adapter_dropout=adapter_dropout,
            disc_use_spectral_norm=disc_spectral_norm,
            use_gumbel_noise=use_gumbel_noise,
            pruning_threshold=pruning_threshold,
        )
    else:  # llava
        from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
        model = PrunableLlavaForConditionalGeneration(
            base_model=base_model,
            pruning_layers=pruning_layers,
            pruner_d_internal=pruner_d_internal,
            pruner_n_heads=pruner_n_heads,
            pruner_query_dropout=pruner_query_dropout,
            disc_d_hidden=disc_d_hidden,
            use_adapter=use_adapter,
            adapter_bottleneck=adapter_bottleneck,
            adapter_type=adapter_type,
            use_separated_adapters=use_separated_adapters,
            vision_adapter_bottleneck=vision_adapter_bottleneck,
            text_adapter_bottleneck=text_adapter_bottleneck,
            generator_adapter_bottleneck=generator_adapter_bottleneck,
            temperature=temperature,
            dropout=dropout,
            adapter_dropout=adapter_dropout,
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

    # 创建 eval DataLoader 用于计算 eval loss（与 train 使用相同的 batch_size）
    eval_loss_loader = None
    eval_loss_iter = None
    if test_dataset:
        eval_wrapper = SimpleDataset(test_dataset)
        eval_sampler = DistributedSampler(
            eval_wrapper,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed + 1000,  # 使用不同的 seed
        )
        eval_loss_loader = DataLoader(
            eval_wrapper,
            batch_size=batch_size,
            sampler=eval_sampler,
            collate_fn=collate_fn,
            num_workers=0,
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

    # 判别器优化器（仅在 discriminator 模式下创建）
    adversarial_mode = method_cfg.get('adversarial_mode', 'discriminator')
    if adversarial_mode == 'discriminator':
        disc_optimizer = torch.optim.Adam(model.get_discriminator_parameters(), lr=disc_lr)
    else:
        disc_optimizer = None

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

        # Disc scheduler: warmup + cosine (仅在 discriminator 模式下创建)
        if disc_optimizer is not None:
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
        if disc_optimizer is not None:
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

            if 'adapter_state_dict' in checkpoint and model.use_adapter and not model.use_separated_adapters:
                model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
                if is_main_process():
                    logger.info("  Loaded adapter_manager state")

            if 'separated_adapter_state_dict' in checkpoint and model.use_adapter and model.use_separated_adapters:
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

            if 'disc_optimizer' in checkpoint and disc_optimizer is not None:
                disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
                if is_main_process():
                    logger.info("  Loaded disc_optimizer state")

            # 加载学习率调度器状态
            reset_step = config.global_settings.get('reset_step_on_load', False)
            if not reset_step:
                if 'pruner_scheduler' in checkpoint and pruner_scheduler is not None:
                    pruner_scheduler.load_state_dict(checkpoint['pruner_scheduler'])
                    if is_main_process():
                        logger.info("  Loaded pruner_scheduler state")

                if 'disc_scheduler' in checkpoint and disc_scheduler is not None:
                    disc_scheduler.load_state_dict(checkpoint['disc_scheduler'])
                    if is_main_process():
                        logger.info("  Loaded disc_scheduler state")

            # 恢复训练步数（如果 reset_step_on_load=True 则跳过）
            if 'step' in checkpoint and not reset_step:
                start_step = checkpoint['step']
                if is_main_process():
                    logger.info(f"  Resuming from step {start_step}")
            elif reset_step and is_main_process():
                logger.info("  Step counter reset to 0 (reset_step_on_load=True)")

            # 重新广播模型参数确保一致性
            broadcast_model_params(model, src=0)

            if is_main_process():
                logger.info("Checkpoint loaded successfully.")
        else:
            if is_main_process():
                logger.warning(f"Checkpoint file not found: {checkpoint_path}, starting from scratch")

    # 训练参数（epochs 已在上面获取）
    print_every = trainer_cfg.get('print_loss_every_batches', 50)
    eval_loss_every = trainer_cfg.get('eval_loss_every_batches', print_every)  # 计算 eval loss 的频率
    eval_every = trainer_cfg.get('eval_every_batches', 1000)
    eval_max_samples = trainer_cfg.get('eval_max_samples', 500)
    save_every = trainer_cfg.get('save_every_batches', 3000)
    grad_clip = trainer_cfg.get('grad_clip_max_norm', None)
    grad_accum_steps = trainer_cfg.get('gradient_accumulation_steps', 1)  # 梯度累积步数

    # 计算总步数（按优化器更新次数计算，不是 batch 数）
    total_batches_per_epoch = len(train_loader)
    total_steps = epochs * (total_batches_per_epoch // grad_accum_steps)

    if is_main_process():
        logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, "
                   f"batches_per_epoch={total_batches_per_epoch}")
        logger.info(f"Gradient accumulation: {grad_accum_steps} steps, "
                   f"Effective batch size: {batch_size * world_size * grad_accum_steps}")
        logger.info(f"Total optimizer steps: {total_steps}, Pruner LR: {pruner_lr}, Disc LR: {disc_lr}")
        # 显示 skip_phase1 状态
        skip_phase1 = method_cfg.get('skip_phase1', False)
        if skip_phase1:
            logger.info(f"[skip_phase1=True] Starting from Phase 2, skipping temperature and sparsity annealing")

    # 保存目录
    save_dir = Path(config.global_settings.get('save_dir', './outputs/checkpoints'))
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    global_step = start_step  # 优化器更新次数
    global_batch = start_step * grad_accum_steps  # 全局 batch 计数（用于 print/eval/save 判断）
    cached_origin_result = None

    # 统计每层的保留数量（用于推荐 topk_ks）
    layer_kept_counts = {idx: [] for idx in pruning_layers}  # {layer_idx: [n_kept_per_batch, ...]}

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
        accum_step = 0  # 累积计数器

        # 使用 tqdm 包装 DataLoader（只在主进程显示进度条）
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main_process())

        # 在 epoch 开始时清零梯度
        pruner_optimizer.zero_grad()
        if disc_optimizer is not None:
            disc_optimizer.zero_grad()

        for batch in pbar:
            accum_step += 1
            is_accum_step = (accum_step % grad_accum_steps != 0)  # 是否是累积中间步

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

            # === 梯度累积：loss 除以累积步数 ===
            pruner_total = sum(v for k, v in losses.items() if k != 'disc_loss') / grad_accum_steps
            pruner_has_grad = pruner_total.requires_grad
            disc_loss = losses.get('disc_loss', None)
            if disc_loss is not None:
                disc_loss = disc_loss / grad_accum_steps
            disc_has_grad = disc_loss is not None and disc_loss.requires_grad

            # === Backward（累积期间禁用 DDP 梯度同步）===
            # 使用 no_sync 上下文管理器在累积期间禁用自动同步
            sync_context = model.no_sync() if (is_accum_step and hasattr(model, 'no_sync')) else nullcontext()
            with sync_context:
                if pruner_has_grad:
                    pruner_total.backward(retain_graph=disc_has_grad)
                if disc_has_grad:
                    disc_loss.backward()

            # === 累积结束时：同步梯度并更新参数 ===
            if not is_accum_step:
                # 同步梯度（关键步骤！）
                sync_gradients(model)

                if disc_has_grad and disc_optimizer is not None:
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.get_discriminator_parameters(), grad_clip)
                    disc_optimizer.step()

                if pruner_has_grad:
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(model.get_pruner_parameters(), grad_clip)
                    pruner_optimizer.step()

                # 清零梯度，准备下一轮累积
                pruner_optimizer.zero_grad()
                if disc_optimizer is not None:
                    disc_optimizer.zero_grad()

                # 学习率调度器步进（按优化器更新次数）
                if pruner_scheduler is not None:
                    pruner_scheduler.step()
                if disc_scheduler is not None:
                    disc_scheduler.step()

                # 更新 global_step（按优化器更新次数）
                global_step += 1

            # 判别器重新初始化（只在累积结束时，且仅在 discriminator 模式下）
            # 注意：需要在所有进程间同步决策，避免死锁
            if not is_accum_step and adversarial_mode == 'discriminator':
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
                elif isinstance(v, (int, float)):
                    # 只累积数值类型的统计信息
                    epoch_stats[k] += v
            n_batches += 1
            global_batch += 1  # 每个 batch 都增加

            # 打印（只在主进程，按 batch 数判断）
            if global_batch % print_every == 0 and is_main_process():
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

            # 计算 eval loss（用于检测过拟合，按 batch 数判断）
            if eval_loss_loader is not None and global_batch % eval_loss_every == 0:
                # 获取一个 eval batch
                if eval_loss_iter is None:
                    eval_loss_iter = iter(eval_loss_loader)
                try:
                    eval_batch = next(eval_loss_iter)
                except StopIteration:
                    eval_loss_iter = iter(eval_loss_loader)
                    eval_batch = next(eval_loss_iter)

                # 计算 eval loss（不做 backward，保持 train 模式以公平比较）
                with torch.no_grad():
                    eval_result = train_step(
                        batch=eval_batch,
                        model=model,
                        processor=processor,
                        config=config,
                        current_step=global_step,
                        total_steps=total_steps,
                        device=device,
                    )

                # 汇总 eval loss（分布式平均）
                eval_losses = eval_result['losses']
                eval_task_loss = eval_losses.get('task_loss', torch.tensor(0.0)).item()
                eval_task_loss_tensor = torch.tensor(eval_task_loss, device=device)
                if dist.is_initialized():
                    dist.all_reduce(eval_task_loss_tensor, op=dist.ReduceOp.SUM)
                    eval_task_loss_tensor /= dist.get_world_size()
                eval_task_loss_avg = eval_task_loss_tensor.item()

                # 打印 train vs eval loss 对比
                if is_main_process():
                    train_task_loss = stats.get('raw_task_loss', 0)
                    diff = eval_task_loss_avg - train_task_loss
                    diff_str = f"+{diff:.4f}" if diff > 0 else f"{diff:.4f}"
                    logger.info(f"  [Loss] train={train_task_loss:.4f}, eval={eval_task_loss_avg:.4f} ({diff_str})")

            # 分布式评估：所有 rank 都参与（按 batch 数判断）
            if test_dataset and global_batch % eval_every == 0:
                if is_main_process():
                    logger.info(f"Evaluating at batch {global_batch} (step {global_step})...")

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

            # 保存（只在主进程，按 batch 数判断）
            if global_batch % save_every == 0 and is_main_process():
                ckpt_path = save_dir / f"checkpoint_batch{global_batch}.pt"
                ckpt_data = {
                    'step': global_step,
                    'batch': global_batch,
                    'pruner_state_dict': model.pruner_manager.state_dict(),
                    'disc_state_dict': model.disc_manager.state_dict(),
                    'pruner_optimizer': pruner_optimizer.state_dict(),
                }
                if disc_optimizer is not None:
                    ckpt_data['disc_optimizer'] = disc_optimizer.state_dict()
                # 根据 adapter 类型保存
                if model.use_adapter:
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
        if model.use_adapter:
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
