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
import gc
from contextlib import nullcontext
from copy import deepcopy

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
from engine.train_utils import compute_task_loss, train_step, _flatten_masked, _get_next_input_layernorm
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
    adapter_alpha_init = method_cfg.get('adapter_alpha_init', 0.1)
    adapter_delta_weight = method_cfg.get('adapter_delta_weight', 0.0)

    # Delayed repair adapter（语言侧，仅 gen_answer tokens）
    use_repair_adapter = method_cfg.get('use_repair_adapter', False)
    repair_layers = method_cfg.get('repair_layers', None)
    repair_source_layers = method_cfg.get('repair_source_layers', None)
    repair_bottleneck_dim = method_cfg.get('repair_bottleneck_dim', 512)
    repair_dropout = method_cfg.get('repair_dropout', adapter_dropout)
    repair_mask_encoder_type = method_cfg.get('repair_mask_encoder_type', method_cfg.get('mask_encoder_type', 'attention'))
    repair_use_pruned_info = method_cfg.get('repair_use_pruned_info', True)
    repair_alpha_init = method_cfg.get('repair_alpha_init', adapter_alpha_init)
    repair_detach_input = method_cfg.get('repair_detach_input', True)
    repair_adapter_type = method_cfg.get('repair_adapter_type', 'lightweight')
    repair_context_num_tokens = int(method_cfg.get('repair_context_num_tokens', 0))
    repair_context_dropout = float(method_cfg.get('repair_context_dropout', 0.0))
    repair_context_use_q2v_relevance = bool(method_cfg.get('repair_context_use_q2v_relevance', False))
    repair_apply_only_gen_tokens = bool(method_cfg.get('repair_apply_only_gen_tokens', True))

    # Subspace repair (low-rank): constrain repair deltas to a calibrated subspace.
    repair_subspace_enable = bool(method_cfg.get('repair_subspace_enable', False))
    repair_subspace_rank = int(method_cfg.get('repair_subspace_rank', 64))
    repair_subspace_orth_scale = float(method_cfg.get('repair_subspace_orth_scale', 0.0))

    # Pruner query dropout
    pruner_query_dropout = method_cfg.get('pruner_query_dropout', 0.0)

    # Pruner 额外配置
    pruner_n_queries = method_cfg.get('pruner_n_queries', 4)
    use_question_condition = method_cfg.get('use_question_condition', False)

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
            adapter_alpha_init=adapter_alpha_init,
            adapter_delta_weight=adapter_delta_weight,
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
            pruner_n_queries=pruner_n_queries,
            pruner_query_dropout=pruner_query_dropout,
            use_question_condition=use_question_condition,
            disc_d_hidden=disc_d_hidden,
            use_adapter=use_adapter,
            adapter_bottleneck=adapter_bottleneck,
            adapter_type=adapter_type,
            use_separated_adapters=use_separated_adapters,
            vision_adapter_bottleneck=vision_adapter_bottleneck,
            text_adapter_bottleneck=text_adapter_bottleneck,
            generator_adapter_bottleneck=generator_adapter_bottleneck,
            adapter_alpha_init=adapter_alpha_init,
            adapter_delta_weight=adapter_delta_weight,
            temperature=temperature,
            dropout=dropout,
            adapter_dropout=adapter_dropout,
            disc_use_spectral_norm=disc_spectral_norm,
            use_gumbel_noise=use_gumbel_noise,
            pruning_threshold=pruning_threshold,
            use_repair_adapter=use_repair_adapter,
            repair_layers=repair_layers,
            repair_source_layers=repair_source_layers,
            repair_bottleneck_dim=repair_bottleneck_dim,
            repair_dropout=repair_dropout,
            repair_mask_encoder_type=repair_mask_encoder_type,
            repair_use_pruned_info=repair_use_pruned_info,
            repair_alpha_init=repair_alpha_init,
            repair_adapter_type=repair_adapter_type,
            repair_context_num_tokens=repair_context_num_tokens,
            repair_context_dropout=repair_context_dropout,
            repair_context_use_q2v_relevance=repair_context_use_q2v_relevance,
            repair_apply_only_gen_tokens=repair_apply_only_gen_tokens,
            repair_detach_input=repair_detach_input,
            repair_subspace_enable=repair_subspace_enable,
            repair_subspace_rank=repair_subspace_rank,
            repair_subspace_orth_scale=repair_subspace_orth_scale,
        )

    # 冻结基础模型
    model.freeze_base_model()

    if logger:
        logger.info(f"Model loaded. Pruning layers: {pruning_layers}, gumbel_mode: {gumbel_mode}")
        logger.info(f"Trainable parameters: Pruners={sum(p.numel() for p in model.get_pruner_parameters()):,}, "
                   f"Adapters={sum(p.numel() for p in model.get_adapter_parameters()):,}, "
                   f"Discriminators={sum(p.numel() for p in model.get_discriminator_parameters()):,}")

    return model, processor


def _build_calib_loader(
    train_wrapper: torch.utils.data.Dataset,
    *,
    num_samples: int,
    batch_size: int,
    seed: int,
):
    """Build a small deterministic calibration DataLoader on rank0."""
    import random
    from torch.utils.data import DataLoader, Subset

    n = len(train_wrapper)
    if n <= 0:
        return None
    num_samples = max(1, min(int(num_samples), n))

    indices = list(range(n))
    rng = random.Random(int(seed))
    rng.shuffle(indices)
    indices = indices[:num_samples]

    subset = Subset(train_wrapper, indices)
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )


def maybe_calibrate_repair_subspace(model, processor, train_wrapper, config, device: torch.device):
    """Calibrate per-layer gap-PCA subspace bases and broadcast to all ranks.

    Runs only when `method_settings.repair_subspace_enable: true`.
    Intended for Stage 2 (adapter fine-tune) when the pruner is frozen and deterministic.
    """
    logger = config.logger if is_main_process() else None
    method_cfg = config.method_settings
    backbone_name = config.backbone_settings.get('name', 'llava-1.5-7b')
    if 'qwen2-vl' in backbone_name.lower():
        return

    if not bool(method_cfg.get("repair_subspace_enable", False)):
        return
    if not getattr(model, "use_repair_adapter", False):
        if logger:
            logger.warning("[repair_subspace_enable] is ON but use_repair_adapter is OFF; skipping subspace calibration.")
        return
    if not hasattr(model, "set_repair_subspace_basis"):
        if logger:
            logger.warning("[repair_subspace_enable] Model has no set_repair_subspace_basis(); skipping.")
        return

    repair_layers = list(method_cfg.get("repair_layers", []) or [])
    if not repair_layers:
        return

    # If already loaded from checkpoint, only recompute when explicitly requested.
    recompute = bool(method_cfg.get("repair_subspace_recompute", False))
    has_existing = bool(getattr(model, "_repair_subspace_basis_names", {}))
    if has_existing and (not recompute):
        if logger:
            logger.info("[repair_subspace] Basis already exists (likely loaded from checkpoint); skip calibration.")
        return

    # Calibration config
    calib_samples = int(method_cfg.get("repair_subspace_calib_samples", 128))
    calib_max_tokens = int(method_cfg.get("repair_subspace_calib_max_tokens_per_layer", 8192))
    subspace_rank = int(method_cfg.get("repair_subspace_rank", 64))
    calib_seed = int(method_cfg.get("repair_subspace_seed", config.global_settings.get("seed", 42)))
    apply_next_ln = bool(method_cfg.get("repair_subspace_apply_next_layernorm", method_cfg.get("repair_loss_apply_next_layernorm", False)))

    trainer_cfg = config.trainer_settings.get('dl_settings', {})
    batch_size = int(trainer_cfg.get('batch_size', 1))
    max_length = int(trainer_cfg.get('max_length', 1024))

    calib_loader = None
    if is_main_process():
        calib_loader = _build_calib_loader(
            train_wrapper,
            num_samples=calib_samples,
            batch_size=batch_size,
            seed=calib_seed,
        )
        if calib_loader is None:
            if logger:
                logger.warning("[repair_subspace] Calibration loader is None; skipping.")
            return

    basis_by_layer_cpu = None
    if is_main_process():
        from method.models.subspace import compute_gap_pca_basis

        gap_buf = {int(l): None for l in repair_layers}  # layer -> (N,D) CPU float32

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in calib_loader:
                    prep = preprocess_batch(batch, processor, device, max_length=max_length)
                    inputs = prep["inputs"]

                    common = {
                        "input_ids": inputs["input_ids"],
                        "pixel_values": inputs["pixel_values"],
                        "attention_mask": inputs["attention_mask"],
                        "vision_start": prep["vision_start"],
                        "vision_end": prep["vision_end"],
                        "question_starts": prep["question_starts"],
                        "question_ends": prep["question_ends"],
                        "answer_starts": prep.get("answer_starts", None),
                        "answer_ends": prep.get("answer_ends", None),
                        "return_pruning_info": False,
                        "detach_h_fake_for_adv": False,
                        "capture_layers": repair_layers,
                        # critical: pre-repair gap
                        "apply_repair": False,
                    }

                    student_out = model(pruning_mode="normal", **common)
                    teacher_out = model(pruning_mode=method_cfg.get("teacher_pruning_mode", "keep_all"), **common)

                    s_caps = student_out.captured or {}
                    t_caps = teacher_out.captured or {}
                    for layer_idx in repair_layers:
                        if layer_idx not in s_caps or layer_idx not in t_caps:
                            continue
                        s = s_caps[layer_idx]
                        t = t_caps[layer_idx]
                        m = (s["mask"] * t["mask"]).to(dtype=torch.bool)
                        s_h = s["h"]
                        t_h = t["h"]
                        if apply_next_ln:
                            ln = _get_next_input_layernorm(model, layer_idx)
                            s_h = ln(s_h)
                            t_h = ln(t_h)
                        gap = _flatten_masked(t_h - s_h, m).float().cpu()
                        if gap.numel() == 0:
                            continue
                        prev = gap_buf[int(layer_idx)]
                        cur = gap if prev is None else torch.cat([prev, gap], dim=0)
                        if int(cur.shape[0]) > calib_max_tokens:
                            perm = torch.randperm(int(cur.shape[0]))[:calib_max_tokens]
                            cur = cur[perm]
                        gap_buf[int(layer_idx)] = cur

            basis_by_layer_cpu = {}
            for layer_idx in repair_layers:
                X = gap_buf.get(int(layer_idx), None)
                if X is None or int(X.shape[0]) < 8:
                    continue
                try:
                    B = compute_gap_pca_basis(X, rank=subspace_rank, center=True, niter=2)  # (D,q)
                except Exception as e:
                    if logger:
                        logger.warning(f"[repair_subspace] PCA failed at layer {layer_idx}: {e}")
                    continue
                basis_by_layer_cpu[int(layer_idx)] = B  # CPU float32

            if logger:
                shapes = {k: tuple(v.shape) for k, v in basis_by_layer_cpu.items()}
                logger.info(f"[repair_subspace] Calibrated gap-PCA bases: {shapes} (apply_next_ln={apply_next_ln})")
        finally:
            if was_training:
                model.train()

    # Broadcast to all ranks: first broadcast effective rank q per layer, then basis values.
    basis_out = {}
    for layer_idx in repair_layers:
        layer_idx = int(layer_idx)
        q = 0
        if is_main_process() and basis_by_layer_cpu is not None and layer_idx in basis_by_layer_cpu:
            q = int(basis_by_layer_cpu[layer_idx].shape[1])
        q_t = torch.tensor([q], device=device, dtype=torch.int64)
        if dist.is_initialized():
            dist.broadcast(q_t, src=0)
        q = int(q_t.item())
        if q <= 0:
            continue

        if is_main_process() and basis_by_layer_cpu is not None and layer_idx in basis_by_layer_cpu:
            B = basis_by_layer_cpu[layer_idx].to(device=device, dtype=torch.float32)
        else:
            hidden = int(getattr(model, "hidden_size", 4096))
            B = torch.zeros(hidden, q, device=device, dtype=torch.float32)
        if dist.is_initialized():
            dist.broadcast(B, src=0)
        basis_out[layer_idx] = B

    if basis_out:
        model.set_repair_subspace_basis(basis_out)
        if dist.is_initialized():
            dist.barrier()
        if logger:
            logger.info(
                f"[repair_subspace] Enabled. rank={subspace_rank}, orth_scale={float(method_cfg.get('repair_subspace_orth_scale', 0.0))} "
                f"layers={sorted(list(basis_out.keys()))}"
            )



# ============================================================
# 主训练循环
# ============================================================

def train(config, rank: int, world_size: int, local_rank: int, device: torch.device):
    """主训练函数（分布式版本）"""
    logger = config.logger if is_main_process() else None
    method_cfg = config.method_settings
    backbone_name = config.backbone_settings.get('name', 'llava-1.5-7b')

    # ============================================================
    # Two-step training (LLaVA only):
    #   Stage 1: train pruner-only model (no repair adapter), save checkpoint
    #   Stage 2: instantiate model with repair adapter, load pruner weights, freeze pruner, finetune adapter
    # Scheduling/annealing (hybrid phases, sparsity anneal) should be computed within Stage 1 only.
    # ============================================================
    two_step_enable = bool(method_cfg.get("two_step_enable", False))
    if 'qwen2-vl' in backbone_name.lower():
        # 用户要求：qwen2-vl 先不动
        two_step_enable = False

    two_step_start_stage = int(method_cfg.get("two_step_start_stage", 1))
    if two_step_start_stage not in (1, 2):
        two_step_start_stage = 1

    two_step_stage1_ratio = float(
        method_cfg.get("two_step_stage1_ratio", method_cfg.get("hybrid_phase2_end", 0.8))
    )
    two_step_stage1_ratio = max(0.0, min(1.0, two_step_stage1_ratio))
    if two_step_enable and two_step_start_stage == 2 and two_step_stage1_ratio >= 1.0:
        raise ValueError("[two_step_start_stage=2] requires two_step_stage1_ratio < 1.0 (otherwise stage2 has 0 steps).")

    stage1_config = config
    stage2_config = config
    if two_step_enable:
        stage1_config = deepcopy(config)
        # Stage 1: build a pruner-only model (no delayed repair adapter modules)
        stage1_config.method_settings['use_repair_adapter'] = False
        stage1_config.method_settings['teacher_forward_enable'] = False
        stage1_config.method_settings['repair_loss_weight'] = 0.0

        stage2_config = deepcopy(config)
        # Stage 2: make pruning deterministic and stop sparsity anneal from drifting targets
        stage2_config.method_settings['gumbel_mode'] = stage2_config.method_settings.get("two_step_stage2_gumbel_mode", "never")
        stage2_config.method_settings['skip_phase1'] = True
        stage2_config.method_settings['sparsity_anneal_ratio'] = 0.0
        stage2_config.method_settings['sparsity_weight'] = float(stage2_config.method_settings.get("two_step_stage2_sparsity_weight", 0.0))
        # Stage 2 temperature: use explicit value if provided, else default to eval_temperature / hybrid low-temp.
        stage2_temp = stage2_config.method_settings.get("two_step_stage2_temperature", None)
        if stage2_temp is None:
            stage2_temp = stage2_config.method_settings.get("eval_temperature", stage2_config.method_settings.get("hybrid_phase1_temp_end", stage2_config.method_settings.get("temperature_min", 0.5)))
        stage2_config.method_settings['temperature_min'] = float(stage2_temp)

    pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])

    # 设置随机种子（每个进程使用不同的种子以获得不同的数据顺序）
    seed = config.global_settings.get('seed', 42)
    torch.manual_seed(seed + rank)

    # 加载模型
    two_step_stage2_only = bool(two_step_enable and two_step_start_stage == 2)
    if two_step_stage2_only and config.global_settings.get('checkpoint', None) is None:
        raise ValueError(
            "[two_step_start_stage=2] requires `global_settings.checkpoint` to point to a Stage 1 (pruner-only) checkpoint."
        )
    active_config = stage2_config if two_step_stage2_only else (stage1_config if two_step_enable else config)
    model, processor = load_model(active_config, device, local_rank)
    if two_step_stage2_only:
        # Stage 2 only: freeze pruner immediately; we'll train adapter on a fixed pruner.
        for p in model.get_pruner_parameters():
            p.requires_grad = False

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
    grad_accum_steps = trainer_cfg.get('gradient_accumulation_steps', 1)  # 梯度累积步数
    epochs = trainer_cfg.get('epochs', 1)

    train_loader = DataLoader(
        train_wrapper,
        batch_size=batch_size,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,  # 图像处理需要在主进程
        pin_memory=True,
    )

    # 计算总步数（按优化器更新次数计算，不是 batch 数）
    total_batches_per_epoch = len(train_loader)
    total_steps = epochs * (total_batches_per_epoch // grad_accum_steps)
    total_steps = max(1, int(total_steps))

    # two-step 切分步数：Stage1 负责所有退火/三阶段调度；Stage2 固定 pruning 行为训 adapter
    stage1_steps = total_steps
    stage2_steps = 0
    if two_step_enable:
        stage1_steps = int(round(total_steps * two_step_stage1_ratio))
        stage1_steps = max(1, min(stage1_steps, total_steps))
        if stage1_steps >= total_steps:
            # 没有 Stage 2
            two_step_enable = False
            active_config = config
            stage2_steps = 0
        else:
            stage2_steps = total_steps - stage1_steps
    if two_step_enable and two_step_start_stage == 2:
        # Stage 2 only: run for the remaining budget (stage2_steps) only.
        if stage2_steps <= 0:
            raise ValueError("[two_step_start_stage=2] stage2_steps computed as 0; check stage1_ratio / total_steps.")

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

    # Stage 1 optimizer:
    # - normal: pruner + adapters together (legacy)
    # - two-step: pruner only
    if two_step_enable and two_step_start_stage == 2:
        # Stage 2 only: adapter fine-tune optimizer (pruner is frozen).
        adapter_lr = float(method_cfg.get("two_step_stage2_lr", pruner_lr))
        adapter_params = list(model.get_repair_adapter_parameters()) if hasattr(model, "get_repair_adapter_parameters") else list(model.get_adapter_parameters())
        if len(adapter_params) == 0:
            raise RuntimeError(
                "[two_step_enable] Stage 2 start requested, but no adapter parameters found. "
                "Check `use_repair_adapter: true` and that repair modules are constructed."
            )
        pruner_optimizer = torch.optim.Adam(adapter_params, lr=adapter_lr, weight_decay=0.0)
    elif two_step_enable:
        pruner_optimizer = torch.optim.Adam(model.get_pruner_parameters(), lr=pruner_lr, weight_decay=pruner_weight_decay)
    else:
        from itertools import chain
        pruner_adapter_params = chain(model.get_pruner_parameters(), model.get_adapter_parameters())
        pruner_optimizer = torch.optim.Adam(pruner_adapter_params, lr=pruner_lr, weight_decay=pruner_weight_decay)

    # 判别器优化器（仅在 discriminator 模式下创建）
    adversarial_mode = method_cfg.get('adversarial_mode', 'discriminator')
    if adversarial_mode == 'discriminator' and not (two_step_enable and two_step_start_stage == 2):
        disc_optimizer = torch.optim.Adam(model.get_discriminator_parameters(), lr=disc_lr)
    else:
        disc_optimizer = None

    # 创建学习率调度器（余弦退火）

    lr_scheduler_cfg = opt_cfg.get('lr_scheduler', {})
    lr_scheduler_type = lr_scheduler_cfg.get('type', 'none')  # 'none', 'cosine', 'linear'
    warmup_ratio = lr_scheduler_cfg.get('warmup_ratio', 0.1)
    min_lr_ratio = lr_scheduler_cfg.get('min_lr_ratio', 0.1)  # 最小学习率 = 初始学习率 * min_lr_ratio

    pruner_scheduler = None
    disc_scheduler = None

    # Stage 2 only: keep constant LR (no scheduler) for a stable adapter environment.
    if (two_step_enable and two_step_start_stage == 2):
        lr_scheduler_type = 'none'

    if lr_scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # 重要：two-step 时，所有退火/调度统一按 Stage 1 的步数缩放（Stage 2 不再退火）
        steps_for_sched = stage1_steps if two_step_enable else total_steps
        steps_for_sched = max(1, int(steps_for_sched))
        warmup_steps = int(steps_for_sched * warmup_ratio)
        warmup_steps = max(0, min(warmup_steps, steps_for_sched - 1)) if steps_for_sched > 1 else 0
        cosine_steps = steps_for_sched - warmup_steps

        # Pruner scheduler: warmup + cosine
        if warmup_steps > 0 and cosine_steps > 0:
            pruner_warmup = LinearLR(pruner_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            pruner_cosine = CosineAnnealingLR(pruner_optimizer, T_max=cosine_steps, eta_min=pruner_lr * min_lr_ratio)
            pruner_scheduler = SequentialLR(pruner_optimizer, schedulers=[pruner_warmup, pruner_cosine], milestones=[warmup_steps])
        elif warmup_steps > 0:
            pruner_scheduler = LinearLR(pruner_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        else:
            pruner_scheduler = CosineAnnealingLR(pruner_optimizer, T_max=steps_for_sched, eta_min=pruner_lr * min_lr_ratio)

        # Disc scheduler: warmup + cosine (仅在 discriminator 模式下创建)
        if disc_optimizer is not None:
            if warmup_steps > 0 and cosine_steps > 0:
                disc_warmup = LinearLR(disc_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
                disc_cosine = CosineAnnealingLR(disc_optimizer, T_max=cosine_steps, eta_min=disc_lr * min_lr_ratio)
                disc_scheduler = SequentialLR(disc_optimizer, schedulers=[disc_warmup, disc_cosine], milestones=[warmup_steps])
            elif warmup_steps > 0:
                disc_scheduler = LinearLR(disc_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
            else:
                disc_scheduler = CosineAnnealingLR(disc_optimizer, T_max=steps_for_sched, eta_min=disc_lr * min_lr_ratio)

        if is_main_process():
            logger.info(f"LR Scheduler: Cosine Annealing with warmup")
            logger.info(f"  Warmup steps: {warmup_steps} ({warmup_ratio:.0%})")
            logger.info(f"  Scheduler steps: {steps_for_sched} (two_step={'ON' if two_step_enable else 'OFF'})")
            logger.info(f"  Min LR ratio: {min_lr_ratio}")

    elif lr_scheduler_type == 'linear':
        from torch.optim.lr_scheduler import LinearLR

        steps_for_sched = stage1_steps if two_step_enable else total_steps
        steps_for_sched = max(1, int(steps_for_sched))
        pruner_scheduler = LinearLR(pruner_optimizer, start_factor=1.0, end_factor=min_lr_ratio, total_iters=steps_for_sched)
        if disc_optimizer is not None:
            disc_scheduler = LinearLR(disc_optimizer, start_factor=1.0, end_factor=min_lr_ratio, total_iters=steps_for_sched)

        if is_main_process():
            logger.info(f"LR Scheduler: Linear Decay")
            logger.info(f"  Scheduler steps: {steps_for_sched} (two_step={'ON' if two_step_enable else 'OFF'})")
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

            # 新版 delayed repair adapter
            if 'repair_context_encoder_state_dict' in checkpoint and getattr(model, 'use_repair_adapter', False):
                if getattr(model, 'repair_context_encoder', None) is not None:
                    model.repair_context_encoder.load_state_dict(checkpoint['repair_context_encoder_state_dict'])
                    if is_main_process():
                        logger.info("  Loaded repair_context_encoder state")

            if 'repair_adapter_state_dict' in checkpoint and getattr(model, 'use_repair_adapter', False):
                if getattr(model, 'repair_adapter_manager', None) is not None:
                    model.repair_adapter_manager.load_state_dict(checkpoint['repair_adapter_state_dict'])
                    if is_main_process():
                        logger.info("  Loaded repair_adapter_manager state")

            # Subspace repair basis (optional)
            if 'repair_subspace_basis_state' in checkpoint and hasattr(model, "set_repair_subspace_basis"):
                try:
                    model.set_repair_subspace_basis(checkpoint['repair_subspace_basis_state'])
                    if is_main_process():
                        shapes = {k: tuple(v.shape) for k, v in checkpoint['repair_subspace_basis_state'].items()}
                        logger.info(f"  Loaded repair_subspace_basis_state: {shapes}")
                except Exception as e:
                    if is_main_process():
                        logger.warning(f"  Failed to load repair_subspace_basis_state: {e}")

            if 'disc_state_dict' in checkpoint:
                model.disc_manager.load_state_dict(checkpoint['disc_state_dict'])
                if is_main_process():
                    logger.info("  Loaded disc_manager state")

            # Stage 2 only: treat checkpoint as Stage 1 pruner-only (or a base checkpoint),
            # do not restore optimizer/scheduler/step state.
            stage2_only = bool(two_step_enable and two_step_start_stage == 2)
            if stage2_only:
                start_step = 0
                if is_main_process():
                    logger.info("  [two_step_start_stage=2] Skipped optimizer/scheduler/step restore; starting Stage 2 from step 0")
            else:
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

    if is_main_process():
        logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, "
                   f"batches_per_epoch={total_batches_per_epoch}")
        logger.info(f"Gradient accumulation: {grad_accum_steps} steps, "
                   f"Effective batch size: {batch_size * world_size * grad_accum_steps}")
        logger.info(f"Total optimizer steps: {total_steps}, Pruner LR: {pruner_lr}, Disc LR: {disc_lr}")
        # 显示 skip_phase1 状态
        skip_phase1 = active_config.method_settings.get('skip_phase1', False)
        if skip_phase1:
            logger.info(f"[skip_phase1=True] Starting from Phase 2, skipping temperature and sparsity annealing")
        if two_step_enable:
            logger.info(f"[two_step_enable] Stage1 steps={stage1_steps} ({two_step_stage1_ratio:.0%}), Stage2 steps={stage2_steps}.")

    # 保存目录
    save_dir = Path(config.global_settings.get('save_dir', './outputs/checkpoints'))
    if is_main_process():
        save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    global_step = start_step  # 优化器更新次数
    global_batch = start_step * grad_accum_steps  # 全局 batch 计数（用于 print/eval/save 判断）
    cached_origin_result = None
    training_stage = 2 if (two_step_enable and two_step_start_stage == 2) else 1
    stage2_switched = False
    if two_step_enable and two_step_start_stage == 2:
        stage2_switched = True
        # Stage 2 only: optionally calibrate low-rank subspace bases before training starts.
        maybe_calibrate_repair_subspace(model, processor, train_wrapper, active_config, device)

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

        should_stop_training = False
        for batch in pbar:
            accum_step += 1
            is_accum_step = (accum_step % grad_accum_steps != 0)  # 是否是累积中间步

            # 训练步骤
            if two_step_enable and two_step_start_stage == 2:
                # Stage 2 only: progress is simply global_step within Stage 2 budget.
                stage_step = global_step
                stage_total = max(1, stage2_steps)
            elif two_step_enable and training_stage == 1:
                stage_step = global_step
                stage_total = stage1_steps
            elif two_step_enable and training_stage == 2:
                stage_step = max(0, global_step - stage1_steps)
                stage_total = max(1, stage2_steps)
            else:
                stage_step = global_step
                stage_total = total_steps

            result = train_step(
                batch=batch,
                model=model,
                processor=processor,
                config=active_config,
                current_step=stage_step,
                total_steps=stage_total,
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
                        # Clip the parameters that the active optimizer is updating (stage-aware).
                        clip_params = []
                        for group in pruner_optimizer.param_groups:
                            clip_params.extend(group.get('params', []))
                        torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
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

                # === Two-step transition: Stage 1 -> Stage 2 (adapter fine-tune) ===
                if (
                    two_step_enable
                    and (not stage2_switched)
                    and training_stage == 1
                    and global_step >= stage1_steps
                ):
                    # Sync all ranks before switching models/optimizers.
                    if dist.is_initialized():
                        dist.barrier()

                    if is_main_process():
                        logger.info("=" * 60)
                        logger.info(f"[two_step_enable] Switching to Stage 2 at global_step={global_step} (batch={global_batch})")
                        logger.info("=" * 60)

                    # Save pruner-only checkpoint (Stage 1 output)
                    if is_main_process():
                        stage1_path = save_dir / "checkpoint_stage1_pruner_only.pt"
                        stage1_ckpt = {
                            'train_stage': 1,
                            'step': global_step,
                            'batch': global_batch,
                            'pruner_state_dict': model.pruner_manager.state_dict(),
                            'disc_state_dict': model.disc_manager.state_dict(),
                            'pruner_optimizer': pruner_optimizer.state_dict(),
                        }
                        if pruner_scheduler is not None:
                            stage1_ckpt['pruner_scheduler'] = pruner_scheduler.state_dict()
                        if disc_optimizer is not None:
                            stage1_ckpt['disc_optimizer'] = disc_optimizer.state_dict()
                        if disc_scheduler is not None:
                            stage1_ckpt['disc_scheduler'] = disc_scheduler.state_dict()
                        torch.save(stage1_ckpt, stage1_path)
                        logger.info(f"[two_step_enable] Saved Stage 1 (pruner-only) checkpoint to {stage1_path}")

                    # Keep pruner weights for Stage 2
                    pruner_state = model.pruner_manager.state_dict()
                    disc_state = model.disc_manager.state_dict()

                    # Release Stage 1 model/optimizer memory before loading Stage 2 model.
                    del model
                    pruner_optimizer = None
                    pruner_scheduler = None
                    disc_optimizer = None
                    disc_scheduler = None
                    gc.collect()
                    torch.cuda.empty_cache()

                    # Instantiate Stage 2 model (with repair adapter), load pruner weights, freeze pruner.
                    active_config = stage2_config
                    model, processor = load_model(active_config, device, local_rank)
                    model.pruner_manager.load_state_dict(pruner_state)
                    model.disc_manager.load_state_dict(disc_state)
                    for p in model.get_pruner_parameters():
                        p.requires_grad = False

                    # Stage 2 optimizer: repair adapter only (LLaVA).
                    adapter_lr = float(method_cfg.get("two_step_stage2_lr", pruner_lr))
                    if hasattr(model, "get_repair_adapter_parameters"):
                        adapter_params = list(model.get_repair_adapter_parameters())
                    else:
                        adapter_params = list(model.get_adapter_parameters())
                    if len(adapter_params) == 0:
                        raise RuntimeError(
                            "[two_step_enable] Stage 2 requested, but no adapter parameters found. "
                            "Check `use_repair_adapter: true` and that repair modules are constructed."
                        )
                    pruner_optimizer = torch.optim.Adam(adapter_params, lr=adapter_lr, weight_decay=0.0)
                    pruner_optimizer.zero_grad()

                    # Re-broadcast to ensure consistent params across ranks.
                    broadcast_model_params(model, src=0)
                    # Optional: calibrate subspace bases for low-rank repair (rank0 computes + broadcast).
                    maybe_calibrate_repair_subspace(model, processor, train_wrapper, active_config, device)

                    training_stage = 2
                    stage2_switched = True
                    # Restart gradient accumulation cycle for Stage 2.
                    accum_step = 0
                    if dist.is_initialized():
                        dist.barrier()

                # Stage 2 only: stop after running the allocated stage2_steps.
                if two_step_enable and two_step_start_stage == 2 and global_step >= int(stage2_steps):
                    if dist.is_initialized():
                        dist.barrier()
                    if is_main_process():
                        logger.info(f"[two_step_start_stage=2] Reached Stage 2 budget: steps={global_step}/{stage2_steps}. Stopping training loop.")
                    should_stop_training = True

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
                stage_str = f" [TrainStage {training_stage}]"
                if two_step_enable:
                    stage_str = f"{stage_str}({stage_step}/{stage_total})"
                noise_str = "noise=ON" if stats.get('use_gumbel_noise', False) else "noise=OFF"
                logger.info(f"Step {global_step}{stage_str}{phase_str}: {loss_str} (temp={stats['temperature']:.2f}, {noise_str})")

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

                # MSE 模式：显示对齐指标
                if 'mse_per_layer' in stats:
                    per_layer_strs = []
                    for layer_idx in sorted(stats['mse_per_layer'].keys()):
                        mse_val = stats['mse_per_layer'][layer_idx]
                        cosine_val = stats['cosine_per_layer'][layer_idx]
                        per_layer_strs.append(f"L{layer_idx}={mse_val:.4f}(cos={cosine_val:.3f})")
                    logger.info(f"  Alignment: MSE={stats['avg_mse']:.4f}, Cosine={stats['avg_cosine']:.3f} [{', '.join(per_layer_strs)}]")

                # Repair Adapter 详细指标
                if 'repair_per_layer_stats' in stats and stats['repair_per_layer_stats']:
                    per_layer_strs = []
                    avg_var_ratio = 0
                    avg_cosine = 0
                    n_layers = 0
                    for layer_idx in sorted(stats['repair_per_layer_stats'].keys()):
                        layer_stats = stats['repair_per_layer_stats'][layer_idx]
                        mean_l = layer_stats.get('mean_loss', 0)
                        var_l = layer_stats.get('var_loss', 0)
                        var_ratio = layer_stats.get('var_ratio', 1.0)
                        cosine = layer_stats.get('cosine_sim', 0)
                        per_layer_strs.append(f"L{layer_idx}(m={mean_l:.4f},v={var_l:.4f},vr={var_ratio:.2f},cos={cosine:.3f})")
                        avg_var_ratio += var_ratio
                        avg_cosine += cosine
                        n_layers += 1
                    if n_layers > 0:
                        avg_var_ratio /= n_layers
                        avg_cosine /= n_layers
                    logger.info(f"  Repair: var_ratio={avg_var_ratio:.2f}, cosine={avg_cosine:.3f} [{', '.join(per_layer_strs)}]")

                # Adapter 修复效果指标
                if 'adapter_stats_per_layer' in stats and stats['adapter_stats_per_layer']:
                    per_layer_strs = []
                    for layer_idx in sorted(stats['adapter_stats_per_layer'].keys()):
                        layer_stats = stats['adapter_stats_per_layer'][layer_idx]
                        dir_cos = layer_stats.get('direction_cosine', 0)
                        mag_ratio = layer_stats.get('magnitude_ratio', 0)
                        per_layer_strs.append(f"L{layer_idx}(dir={dir_cos:.3f},mag={mag_ratio:.2f})")
                    avg_dir = stats.get('adapter_avg_direction_cosine', 0)
                    avg_mag = stats.get('adapter_avg_magnitude_ratio', 0)
                    logger.info(f"  Adapter: dir_cos={avg_dir:.3f}, mag_ratio={avg_mag:.2f} [{', '.join(per_layer_strs)}]")

                # Repair delta decomposition diagnostics (optional regularizer)
                # Helps detect the "mostly orthogonal delta" failure mode early.
                if 'raw_repair_delta_reg_loss' in stats:
                    logger.info(
                        f"  RepairDeltaReg: loss={stats.get('raw_repair_delta_reg_loss', 0.0):.6f} "
                        f"(w_l2={method_cfg.get('repair_delta_l2_weight', 0.0)}, "
                        f"w_orth={method_cfg.get('repair_delta_orth_weight', 0.0)}, "
                        f"w_frac={method_cfg.get('repair_delta_frac_weight', 0.0)})"
                    )
                if 'repair_delta_diag_per_layer' in stats and stats['repair_delta_diag_per_layer']:
                    per_layer_strs = []
                    for layer_idx in sorted(stats['repair_delta_diag_per_layer'].keys()):
                        d = stats['repair_delta_diag_per_layer'][layer_idx]
                        per_layer_strs.append(
                            f"L{layer_idx}(frac={d.get('frac_mean', 0.0):.3f},"
                            f"orth={d.get('orth_mse', 0.0):.4f},"
                            f"dl2={d.get('delta_l2', 0.0):.4f})"
                        )
                    logger.info(f"  RepairDeltaDiag: [{', '.join(per_layer_strs)}]")

                # Direct teacher-closeness: token-wise MSE before vs after repair (per-layer + avg improvement%)
                if 'repair_mse_improve_per_layer' in stats and stats['repair_mse_improve_per_layer']:
                    per_layer_strs = []
                    for layer_idx in sorted(stats['repair_mse_improve_per_layer'].keys()):
                        imp = stats['repair_mse_improve_per_layer'][layer_idx]
                        mb = stats.get('repair_mse_before_per_layer', {}).get(layer_idx, None)
                        ma = stats.get('repair_mse_after_per_layer', {}).get(layer_idx, None)
                        if mb is None or ma is None:
                            per_layer_strs.append(f"L{layer_idx}(imp={imp:+.1f}%)")
                        else:
                            per_layer_strs.append(f"L{layer_idx}(b={mb:.4f},a={ma:.4f},imp={imp:+.1f}%)")
                    logger.info(
                        f"  RepairMSE: avg_imp={stats.get('repair_mse_improve_avg', 0.0):+.1f}% "
                        f"(avg_b={stats.get('repair_mse_before_avg', 0.0):.4f}, avg_a={stats.get('repair_mse_after_avg', 0.0):.4f}) "
                        f"[{', '.join(per_layer_strs)}]"
                    )

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
                eval_result = None
                with torch.no_grad():
                    eval_result = train_step(
                        batch=eval_batch,
                        model=model,
                        processor=processor,
                        config=active_config,
                        current_step=stage_step,
                        total_steps=stage_total,
                        device=device,
                    )
                if eval_result is None:
                    raise RuntimeError("eval_result is None: eval loss computation did not run as expected.")

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
                                model, processor, test_dataset, judge, active_config, device,
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
                            model, processor, test_dataset, judge, active_config, device,
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
                                logger.info(
                                    f"  [{eval_mode}] Avg kept ratio: "
                                    f"{eval_result['avg_kept_ratio']:.2%} [{layer_str}]"
                                )

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
                # 新版 delayed repair adapter
                if getattr(model, 'use_repair_adapter', False):
                    if getattr(model, 'repair_context_encoder', None) is not None:
                        ckpt_data['repair_context_encoder_state_dict'] = model.repair_context_encoder.state_dict()
                    if getattr(model, 'repair_adapter_manager', None) is not None:
                        ckpt_data['repair_adapter_state_dict'] = model.repair_adapter_manager.state_dict()
                    # Optional: calibrated low-rank subspace basis
                    if hasattr(model, "get_repair_subspace_state"):
                        subspace_state = model.get_repair_subspace_state()
                        if subspace_state:
                            ckpt_data["repair_subspace_basis_state"] = subspace_state
                if pruner_scheduler is not None:
                    ckpt_data['pruner_scheduler'] = pruner_scheduler.state_dict()
                if disc_scheduler is not None:
                    ckpt_data['disc_scheduler'] = disc_scheduler.state_dict()
                torch.save(ckpt_data, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")

            # 同步所有进程
            if dist.is_initialized():
                dist.barrier()

            if should_stop_training:
                break

        if should_stop_training:
            break

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
        if getattr(model, 'use_repair_adapter', False):
            if getattr(model, 'repair_context_encoder', None) is not None:
                final_ckpt['repair_context_encoder_state_dict'] = model.repair_context_encoder.state_dict()
            if getattr(model, 'repair_adapter_manager', None) is not None:
                final_ckpt['repair_adapter_state_dict'] = model.repair_adapter_manager.state_dict()
            if hasattr(model, "get_repair_subspace_state"):
                subspace_state = model.get_repair_subspace_state()
                if subspace_state:
                    final_ckpt["repair_subspace_basis_state"] = subspace_state
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
                        model, processor, test_dataset, judge, active_config, device,
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
