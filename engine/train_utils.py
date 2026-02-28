"""训练相关工具函数"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from engine.data_utils import preprocess_batch, preprocess_batch_qwen2vl


def _flatten_masked(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """将 (batch, L, D) 按 mask 展平为 (N, D)。mask 为 0/1 或 bool。"""
    if h is None or mask is None:
        raise ValueError("_flatten_masked requires both h and mask.")
    if h.dim() != 3 or mask.dim() != 2:
        raise ValueError(f"Expected h=(b,L,D) and mask=(b,L), got {tuple(h.shape)} / {tuple(mask.shape)}")
    b, L, d = h.shape
    h2 = h.reshape(b * L, d)
    m2 = mask.reshape(b * L).to(dtype=torch.bool)
    if m2.sum().item() == 0:
        # 兜底：返回一个 1xD 的零向量，避免下游 NaN
        return torch.zeros(1, d, device=h.device, dtype=h.dtype)
    return h2[m2]


def compute_distribution_alignment_loss(
    student_h: torch.Tensor,
    teacher_h: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mean_var",
    var_weight: float = 1.0,
    return_stats: bool = False,
):
    """分布级对齐损失（仅用于 gen_answer tokens）。

    目标不是逐 token 点到点对齐，而是让 student/teacher 在该区域的表示分布接近。

    Args:
        student_h/teacher_h: (batch, Lmax, hidden)
        mask: (batch, Lmax) 0/1，有效位置
        loss_type:
            - "mse": 点到点 MSE（作为对照）
            - "mean_var": 对齐均值 + 方差（更像"分布接近"）
        return_stats: 是否返回详细统计信息

    Returns:
        loss: 损失值
        stats (optional): 详细统计信息字典
    """
    Xs = _flatten_masked(student_h, mask).float()
    Xt = _flatten_masked(teacher_h, mask).float()

    if loss_type == "mse":
        # 点到点：要求 token 顺序一致（这里 pad 后是对齐的）
        n = min(Xs.shape[0], Xt.shape[0])
        loss = F.mse_loss(Xs[:n], Xt[:n])
        if return_stats:
            return loss, {'mse': float(loss.detach().item())}
        return loss

    # mean + diag(var) 对齐
    ms = Xs.mean(dim=0)
    mt = Xt.mean(dim=0)
    vs = Xs.var(dim=0, unbiased=False)
    vt = Xt.var(dim=0, unbiased=False)

    mean_loss = F.mse_loss(ms, mt)
    var_loss = F.mse_loss(vs, vt)
    loss = mean_loss + var_weight * var_loss

    if return_stats:
        # 计算更多诊断指标
        stats = {
            'mean_loss': float(mean_loss.detach().item()),
            'var_loss': float(var_loss.detach().item()),
            'student_var_mean': float(vs.mean().detach().item()),  # student 平均方差
            'teacher_var_mean': float(vt.mean().detach().item()),  # teacher 平均方差
            'var_ratio': float((vs.mean() / (vt.mean() + 1e-8)).detach().item()),  # 方差比
            'cosine_sim': float(F.cosine_similarity(ms.unsqueeze(0), mt.unsqueeze(0)).detach().item()),  # 均值余弦相似度
        }
        return loss, stats

    return loss


def _get_next_input_layernorm(model, layer_idx: int):
    """Get the LayerNorm/RMSNorm that the next decoder layer applies to its inputs.

    If layer_idx is the last layer, return the final norm.
    """
    base_model = model.module if hasattr(model, "module") else model
    llm = base_model.base_model.model.language_model
    next_layer_idx = int(layer_idx) + 1
    if next_layer_idx < len(llm.layers):
        next_layer = llm.layers[next_layer_idx]
        # PrunableLlamaDecoderLayer wraps the original layer under .original_layer
        if hasattr(next_layer, "original_layer"):
            return next_layer.original_layer.input_layernorm
        return next_layer.input_layernorm
    return llm.norm


def _compute_adapter_repair_stats(
    h_before: torch.Tensor,
    h_after: torch.Tensor,
    h_teacher: torch.Tensor,
    mask: torch.Tensor,
) -> dict:
    """计算 adapter 修复效果指标

    Args:
        h_before: (batch, Lmax, hidden) - repair 前的表征
        h_after: (batch, Lmax, hidden) - repair 后的表征
        h_teacher: (batch, Lmax, hidden) - teacher 的表征
        mask: (batch, Lmax) - 有效位置 mask

    Returns:
        dict: {
            'direction_cosine': 方向正确性（越接近 1 越好）
            'magnitude_ratio': 大小比例（越接近 1 越好）
        }
    """
    # 展平有效位置
    delta = _flatten_masked(h_after - h_before, mask).float()  # adapter 实际修正
    target = _flatten_masked(h_teacher - h_before, mask).float()  # 理想修正

    # 1. 方向正确性：cosine similarity
    delta_flat = delta.reshape(-1)
    target_flat = target.reshape(-1)
    delta_norm = delta_flat.norm()
    target_norm = target_flat.norm()

    if delta_norm > 1e-8 and target_norm > 1e-8:
        direction_cosine = F.cosine_similarity(
            delta_flat.unsqueeze(0), target_flat.unsqueeze(0)
        ).item()
    else:
        direction_cosine = 0.0

    # 2. 大小比例：|delta| / |target|
    if target_norm > 1e-8:
        magnitude_ratio = (delta_norm / target_norm).item()
    else:
        magnitude_ratio = 0.0

    return {
        'direction_cosine': direction_cosine,
        'magnitude_ratio': magnitude_ratio,
    }


def _compute_delta_decomposition_losses(
    h_before: torch.Tensor,
    h_after: torch.Tensor,
    h_teacher: torch.Tensor,
    mask: torch.Tensor,
):
    """Decompose adapter delta into components parallel/orthogonal to the ideal correction.

    delta  = h_after - h_before
    target = h_teacher - h_before

    Returns:
        losses: dict with keys:
            - delta_l2: mean(||delta||^2)
            - orth_mse: mean(||delta_orth||^2)
            - frac_mse: mean((frac-1)^2), where frac is the amount along target direction
        stats: lightweight diagnostics (means only)
    """
    delta = _flatten_masked(h_after - h_before, mask).float()
    target = _flatten_masked(h_teacher - h_before, mask).float()

    # Per-token dot products
    dot = (delta * target).sum(dim=1)
    tgt_norm_sq = (target * target).sum(dim=1).clamp(min=1e-8)
    frac = dot / tgt_norm_sq  # how much of the ideal correction is applied along target direction

    delta_parallel = frac.unsqueeze(1) * target
    delta_orth = delta - delta_parallel

    delta_l2 = (delta * delta).mean()
    orth_mse = (delta_orth * delta_orth).mean()
    frac_mse = ((frac - 1.0) ** 2).mean()

    stats = {
        "frac_mean": float(frac.detach().mean().item()),
        "orth_mse": float(orth_mse.detach().item()),
        "delta_l2": float(delta_l2.detach().item()),
    }
    return {"delta_l2": delta_l2, "orth_mse": orth_mse, "frac_mse": frac_mse}, stats


def compute_task_loss(
    logits: torch.Tensor,
    answer_starts: List[int],
    answers: List[str],
    tokenizer,
    device: torch.device,
) -> torch.Tensor:
    """计算 task loss (cross entropy)

    Args:
        logits: 模型输出的 logits (batch, seq_len, vocab_size)
        answer_starts: 每个样本的答案起始位置
        answers: 答案文本列表
        tokenizer: tokenizer
        device: 设备

    Returns:
        平均 cross entropy loss
    """
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

        loss = F.cross_entropy(pred_logits, targets)
        total_loss = total_loss + loss

    return total_loss / batch_size if batch_size > 0 else total_loss


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

    Args:
        batch: 样本列表
        model: 可剪枝模型
        processor: LLaVA processor
        config: 配置对象
        current_step: 当前步数
        total_steps: 总步数
        device: 设备

    Returns:
        包含 losses, stats, pruning_infos 的字典
    """
    method_cfg = config.method_settings
    adversarial_mode = method_cfg.get('adversarial_mode', 'none')  # 'none' | 'discriminator' | 'mse'（旧逻辑）

    # === Gumbel Mode 两阶段调度 ===
    gumbel_mode = method_cfg.get('gumbel_mode', 'never')
    skip_phase1 = method_cfg.get('skip_phase1', False)
    progress = current_step / total_steps if total_steps > 0 else 0

    if gumbel_mode == 'hybrid':
        # 混合三阶段策略
        phase1_end = method_cfg.get('hybrid_phase1_end', 0.5)
        phase2_end = method_cfg.get('hybrid_phase2_end', 1.0)  # 第三阶段起始点
        phase1_temp_start = method_cfg.get('hybrid_phase1_temp_start', 1.5)
        phase1_temp_end = method_cfg.get('hybrid_phase1_temp_end', 0.1)

        if skip_phase1:
            # 跳过阶段1，直接从阶段2开始
            current_temp = phase1_temp_end
            use_gumbel_noise = True
            current_phase = 2
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
                # 阶段3：对齐期 - 低温 + 关闭 noise（训练推理一致）
                current_temp = phase1_temp_end
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

    # 根据模型类型选择预处理函数
    backbone_name = config.backbone_settings.get('name', 'llava-1.5-7b')
    if 'qwen2-vl' in backbone_name.lower():
        prep = preprocess_batch_qwen2vl(batch, processor, device, max_length=max_length)
    else:
        prep = preprocess_batch(batch, processor, device, max_length=max_length)
    inputs = prep['inputs']

    # === Forward ===
    model.train()

    # 是否阻止 adv_loss 梯度流向 pruner
    detach_adv_from_pruner = method_cfg.get('detach_adv_from_pruner', False)

    # repair / teacher-forward 配置
    repair_layers = method_cfg.get('repair_layers', [])
    repair_loss_weight = method_cfg.get('repair_loss_weight', 0.0)
    repair_loss_type = method_cfg.get('repair_loss_type', 'mean_var')
    repair_var_weight = method_cfg.get('repair_var_weight', 1.0)
    teacher_forward_enable = method_cfg.get('teacher_forward_enable', False)
    repair_delta_l2_weight = method_cfg.get('repair_delta_l2_weight', 0.0)
    repair_delta_orth_weight = method_cfg.get('repair_delta_orth_weight', 0.0)
    repair_delta_frac_weight = method_cfg.get('repair_delta_frac_weight', 0.0)

    capture_layers = repair_layers if (teacher_forward_enable and repair_loss_weight > 0 and repair_layers) else []

    # 构建 student forward 参数
    forward_kwargs = {
        'input_ids': inputs['input_ids'],
        'pixel_values': inputs['pixel_values'],
        'attention_mask': inputs['attention_mask'],
        'vision_start': prep['vision_start'],
        'vision_end': prep['vision_end'],
        'question_starts': prep['question_starts'],
        'question_ends': prep['question_ends'],
        'answer_starts': prep['answer_starts'],
        'answer_ends': prep['answer_ends'],
        'return_pruning_info': True,  # student 需要 pruning_infos 来算 sparsity
        'detach_h_fake_for_adv': detach_adv_from_pruner,
        'pruning_mode': 'normal',
        'apply_repair': True,
        'capture_layers': capture_layers,
    }

    # Qwen2-VL 需要 image_grid_thw
    if 'image_grid_thw' in inputs:
        forward_kwargs['image_grid_thw'] = inputs['image_grid_thw']

    output = model(**forward_kwargs)

    # teacher forward（keep_all，无修复，用于定义“source 表示分布”）
    teacher_output = None
    if teacher_forward_enable and (repair_loss_weight > 0) and capture_layers:
        teacher_kwargs = dict(forward_kwargs)
        teacher_kwargs['return_pruning_info'] = False
        teacher_kwargs['pruning_mode'] = method_cfg.get('teacher_pruning_mode', 'keep_all')
        teacher_kwargs['apply_repair'] = False
        with torch.no_grad():
            teacher_output = model(**teacher_kwargs)

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
    )
    losses['task_loss'] = task_loss
    stats['raw_task_loss'] = task_loss.item()

    # 2. Repair loss（teacher-forcing, gen_answer region, distribution alignment）
    if (teacher_output is not None) and output.captured and teacher_output.captured:
        student_caps = output.captured_for_repair or output.captured
        teacher_caps = teacher_output.captured
        layer_losses = []
        per_layer = {}
        per_layer_stats = {}
        apply_next_ln = bool(method_cfg.get("repair_loss_apply_next_layernorm", False))
        for layer_idx in capture_layers:
            if layer_idx not in student_caps or layer_idx not in teacher_caps:
                continue
            s = student_caps[layer_idx]
            t = teacher_caps[layer_idx]
            m = s["mask"] * t["mask"]
            s_h = s["h"]
            t_h = t["h"]
            if apply_next_ln:
                ln = _get_next_input_layernorm(model, layer_idx)
                s_h = ln(s_h)
                t_h = ln(t_h)
            layer_loss, layer_stats = compute_distribution_alignment_loss(
                s_h, t_h, m,
                loss_type=repair_loss_type,
                var_weight=repair_var_weight,
                return_stats=True,
            )
            layer_losses.append(layer_loss)
            per_layer[layer_idx] = float(layer_loss.detach().item())
            per_layer_stats[layer_idx] = layer_stats

        if layer_losses:
            repair_loss = torch.stack(layer_losses).mean()
        else:
            repair_loss = torch.tensor(0.0, device=device)

        losses['repair_loss'] = repair_loss
        stats['raw_repair_loss'] = float(repair_loss.detach().item())
        stats['repair_loss_type'] = repair_loss_type
        stats['repair_per_layer'] = per_layer
        stats['repair_per_layer_stats'] = per_layer_stats  # 详细统计

        # === Adapter 修复效果指标 ===
        # 计算 adapter 的 delta 与理想 delta 的对比
        # delta = h_after_repair - h_before_repair（adapter 实际修正）
        # target = h_teacher - h_before_repair（理想修正）
        before_caps = output.captured_before_repair
        if before_caps:
            adapter_stats_per_layer = {}
            delta_loss_per_layer = {}
            delta_diag_per_layer = {}
            delta_losses_accum = []
            for layer_idx in capture_layers:
                if layer_idx not in before_caps or layer_idx not in student_caps or layer_idx not in teacher_caps:
                    continue
                h_before = before_caps[layer_idx]["h"]  # (batch, Lmax, hidden)
                h_after = student_caps[layer_idx]["h"]   # (batch, Lmax, hidden)
                h_teacher = teacher_caps[layer_idx]["h"] # (batch, Lmax, hidden)
                m = before_caps[layer_idx]["mask"] * student_caps[layer_idx]["mask"] * teacher_caps[layer_idx]["mask"]

                # 计算 adapter 修复效果指标
                layer_adapter_stats = _compute_adapter_repair_stats(h_before, h_after, h_teacher, m)
                adapter_stats_per_layer[layer_idx] = layer_adapter_stats

                # 可选：对 delta 做更强的约束（鼓励沿 target 方向、抑制正交分量）
                if (repair_delta_l2_weight > 0) or (repair_delta_orth_weight > 0) or (repair_delta_frac_weight > 0):
                    dl, diag = _compute_delta_decomposition_losses(h_before, h_after, h_teacher, m)
                    delta_diag_per_layer[layer_idx] = diag
                    # Weighted per-layer delta regularizers
                    layer_delta_loss = torch.tensor(0.0, device=h_after.device)
                    if repair_delta_l2_weight > 0:
                        layer_delta_loss = layer_delta_loss + repair_delta_l2_weight * dl["delta_l2"]
                    if repair_delta_orth_weight > 0:
                        layer_delta_loss = layer_delta_loss + repair_delta_orth_weight * dl["orth_mse"]
                    if repair_delta_frac_weight > 0:
                        layer_delta_loss = layer_delta_loss + repair_delta_frac_weight * dl["frac_mse"]
                    delta_loss_per_layer[layer_idx] = float(layer_delta_loss.detach().item())
                    delta_losses_accum.append(layer_delta_loss)

            if adapter_stats_per_layer:
                stats['adapter_stats_per_layer'] = adapter_stats_per_layer
                # 计算平均值
                avg_direction = sum(s['direction_cosine'] for s in adapter_stats_per_layer.values()) / len(adapter_stats_per_layer)
                avg_magnitude = sum(s['magnitude_ratio'] for s in adapter_stats_per_layer.values()) / len(adapter_stats_per_layer)
                stats['adapter_avg_direction_cosine'] = avg_direction
                stats['adapter_avg_magnitude_ratio'] = avg_magnitude
            if delta_losses_accum:
                losses["repair_delta_reg_loss"] = torch.stack(delta_losses_accum).mean()
                stats["raw_repair_delta_reg_loss"] = float(losses["repair_delta_reg_loss"].detach().item())
                stats["repair_delta_reg_per_layer"] = delta_loss_per_layer
                stats["repair_delta_diag_per_layer"] = delta_diag_per_layer
    else:
        losses['repair_loss'] = torch.tensor(0.0, device=device)
        stats['raw_repair_loss'] = 0.0

    # 3. Sparsity Loss
    if output.pruning_infos and len(output.pruning_infos) > 0:
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

        # 收集各层的累积保留率
        cumulative_ratios = []
        for i, layer_idx in enumerate(pruning_layers):
            cumulative_mask = output.pruning_infos[layer_idx]['cumulative_mask']
            cumulative_ratio = cumulative_mask.float().mean()
            cumulative_ratios.append(cumulative_ratio)
            stats[f'L{layer_idx}_kept'] = cumulative_ratio.item()
        # 计算独立保留率 p_i = 当前层的 current_mask 的平均值
        # 不再使用除法 cumulative_r_i / cumulative_r_{i-1}，避免梯度计算不稳定
        independent_ratios = []
        for i, layer_idx in enumerate(pruning_layers):
            info = output.pruning_infos[layer_idx]
            current_mask = info.get('current_mask')
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

        # 计算各段的层数：[n0, n1, ..., nK]，长度 = n_pruning_layers + 1
        # n0 = 第一层剪枝前的层数（这些层没有被剪）
        # n1 = 从 pruning_layers[0] 到 pruning_layers[1] 之间的层数（都受第一个 pruner 影响）
        # ...
        segment_lengths = []
        segment_lengths.append(pruning_layers[0])  # 0..L0-1
        for i in range(n_pruning_layers - 1):
            segment_lengths.append(pruning_layers[i + 1] - pruning_layers[i])
        segment_lengths.append(total_layers - pruning_layers[-1])

        if sparsity_loss_mode == 'exact':
            # avg = (n0*1 + n1*p1 + n2*p1*p2 + ...) / total_layers
            avg_kept = torch.tensor(0.0, device=device)
            avg_kept = avg_kept + segment_lengths[0] * 1.0
            cumulative_product = torch.tensor(1.0, device=device)
            for i in range(n_pruning_layers):
                cumulative_product = cumulative_product * independent_ratios[i]
                avg_kept = avg_kept + segment_lengths[i + 1] * cumulative_product
            avg_kept = avg_kept / total_layers
            sparsity_loss = torch.abs(avg_kept - target_ratio)
            stats['avg_kept_ratio'] = avg_kept.item()
        else:
            # harmonic mean 近似
            inv_sum = sum(1.0 / p for p in independent_ratios)
            hm = n_pruning_layers / inv_sum
            avg_approx = torch.tensor(0.0, device=device)
            avg_approx = avg_approx + segment_lengths[0] * 1.0
            hm_power = hm
            for i in range(1, len(segment_lengths)):
                avg_approx = avg_approx + segment_lengths[i] * hm_power
                hm_power = hm_power * hm
            avg_approx = avg_approx / total_layers
            sparsity_loss = torch.abs(avg_approx - target_ratio)
            stats['harmonic_mean'] = hm.item()
            stats['avg_kept_ratio'] = avg_approx.item()

        losses['sparsity_loss'] = sparsity_loss
        stats['raw_sparsity_loss'] = sparsity_loss.item()
        stats['final_kept_ratio'] = cumulative_ratios[-1].item()
        stats['target_kept_ratio'] = target_ratio
        stats['total_layers'] = total_layers
    else:
        losses['sparsity_loss'] = torch.tensor(0.0, device=device)
        stats['raw_sparsity_loss'] = 0.0
        stats['avg_kept_ratio'] = 1.0
        stats['final_kept_ratio'] = 1.0
        stats['target_kept_ratio'] = 1.0
        stats['total_layers'] = 0

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

    # === Adapter Delta 正则：限制修正幅度 ===
    adapter_delta_weight = method_cfg.get('adapter_delta_weight', 0.0)
    if adapter_delta_weight > 0:
        base_model = model.module if hasattr(model, 'module') else model
        delta_loss = None
        if getattr(base_model, 'use_adapter', False):
            if base_model.use_separated_adapters and base_model.separated_adapter_manager is not None:
                delta_loss = base_model.separated_adapter_manager.collect_delta_loss()
            elif base_model.adapter_manager is not None:
                delta_loss = base_model.adapter_manager.collect_delta_loss()
        if delta_loss is not None:
            losses['adapter_delta_loss'] = delta_loss
            stats['adapter_delta_loss'] = delta_loss.item()
            stats['adapter_delta_weight'] = adapter_delta_weight

    # === 应用权重 ===
    task_weight = method_cfg.get('task_loss_weight', 1.0)
    repair_weight = method_cfg.get('repair_loss_weight', 0.0)
    sparsity_weight = method_cfg.get('sparsity_weight', 0.2)

    warmup_ratio = method_cfg.get('loss_weight_warmup_ratio', 0.0)
    if warmup_ratio > 0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2

        task_weight_start = method_cfg.get('task_loss_weight_start', task_weight)
        task_weight = task_weight_start + (task_weight - task_weight_start) * cosine_factor.item()

    if method_cfg.get('sparsity_warmup_enable', False):
        sparsity_warmup_ratio = method_cfg.get('sparsity_warmup_ratio', 0.2)
        sparsity_weight_max = method_cfg.get('sparsity_weight_max', sparsity_weight)
        if progress < sparsity_warmup_ratio:
            sparsity_weight = sparsity_weight + (sparsity_weight_max - sparsity_weight) * (progress / sparsity_warmup_ratio)
        else:
            sparsity_weight = sparsity_weight_max

    stats['task_weight'] = task_weight
    stats['repair_weight'] = repair_weight
    stats['sparsity_weight'] = sparsity_weight

    weighted_losses = {
        'task_loss': losses['task_loss'] * task_weight,
    }
    if 'repair_loss' in losses:
        weighted_losses['repair_loss'] = losses['repair_loss'] * repair_weight
    if 'repair_delta_reg_loss' in losses:
        weighted_losses['repair_delta_reg_loss'] = losses['repair_delta_reg_loss']
    if 'sparsity_loss' in losses:
        weighted_losses['sparsity_loss'] = losses['sparsity_loss'] * sparsity_weight
    if 'entropy_loss' in losses:
        weighted_losses['entropy_loss'] = losses['entropy_loss'] * entropy_weight
    if 'adapter_delta_loss' in losses:
        weighted_losses['adapter_delta_loss'] = losses['adapter_delta_loss'] * adapter_delta_weight
    if 'tightening_loss' in losses:
        weighted_losses['tightening_loss'] = losses['tightening_loss']  # 权重已在计算时应用
    # 不再使用判别器相关损失（如需可恢复旧逻辑）

    return {
        'losses': weighted_losses,
        'stats': stats,
        'pruning_infos': {idx: info for idx, info in output.pruning_infos.items()} if output.pruning_infos else None,
    }
