"""训练相关工具函数"""

import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
from typing import Dict, Any, List

from engine.data_utils import preprocess_batch


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
) -> torch.Tensor:
    """分布级对齐损失（仅用于 gen_answer tokens）。

    目标不是逐 token 点到点对齐，而是让 student/teacher 在该区域的表示分布接近。

    Args:
        student_h/teacher_h: (batch, Lmax, hidden)
        mask: (batch, Lmax) 0/1，有效位置
        loss_type:
            - "mse": 点到点 MSE（作为对照）
            - "mean_var": 对齐均值 + 方差（旧口径）
            - "w2": diagonal-Gaussian W2^2 surrogate（对齐均值 + 标准差；建议使用）
    """
    details = compute_distribution_alignment_details(
        student_h=student_h,
        teacher_h=teacher_h,
        mask=mask,
        loss_type=loss_type,
        var_weight=var_weight,
    )
    return details["total"]


def compute_distribution_alignment_details(
    student_h: torch.Tensor,
    teacher_h: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mean_var",
    var_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """返回对齐损失及其可解释分解项（用于训练时打印诊断指标）。"""
    if student_h is None or teacher_h is None or mask is None:
        raise ValueError("compute_distribution_alignment_details requires student_h, teacher_h and mask.")
    if student_h.dim() != 3 or teacher_h.dim() != 3 or mask.dim() != 2:
        raise ValueError(
            f"Expected student_h/teacher_h=(b,L,D) and mask=(b,L), got "
            f"{tuple(student_h.shape)} / {tuple(teacher_h.shape)} / {tuple(mask.shape)}"
        )
    if student_h.shape != teacher_h.shape:
        raise ValueError(f"student_h and teacher_h must have same shape, got {tuple(student_h.shape)} vs {tuple(teacher_h.shape)}")

    # 仅保留“多卡全局统计”路径：
    # - 这里的目标是让对齐损失使用 *全局 batch*（跨 rank）的 masked moments
    # - 因此强制要求分布式已初始化，并用可导的 all_reduce 聚合统计量
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "Distribution alignment loss requires torch.distributed to be initialized. "
            "Please launch with torchrun (world_size>=1) so global masked moments are well-defined."
        )

    h_s = student_h.float()
    h_t = teacher_h.float()
    m = mask.to(dtype=h_s.dtype)
    m_exp = m.unsqueeze(-1)  # (b, L, 1)

    sum_s = (h_s * m_exp).sum(dim=(0, 1))  # (D,)
    sum_s2 = ((h_s * h_s) * m_exp).sum(dim=(0, 1))  # (D,)
    sum_t = (h_t * m_exp).sum(dim=(0, 1))  # (D,)
    sum_t2 = ((h_t * h_t) * m_exp).sum(dim=(0, 1))  # (D,)

    # token-wise MSE (diagnostic + loss_type="mse")
    diff2_sum = (((h_s - h_t) * m_exp) ** 2).sum()  # scalar

    count = m.sum()  # scalar (token count)
    d = float(h_s.shape[-1])

    # Fuse multiple reductions into one collective to reduce latency.
    D = sum_s.shape[0]
    pack = torch.cat(
        [
            sum_s,
            sum_s2,
            sum_t,
            sum_t2,
            count.reshape(1),
            diff2_sum.reshape(1),
        ],
        dim=0,
    )
    pack = dist_nn_f.all_reduce(pack, op=dist.ReduceOp.SUM)
    sum_s = pack[0:D]
    sum_s2 = pack[D:2 * D]
    sum_t = pack[2 * D:3 * D]
    sum_t2 = pack[3 * D:4 * D]
    count = pack[4 * D]
    diff2_sum = pack[4 * D + 1]

    denom = count.clamp(min=1.0)
    inv = 1.0 / denom
    ms = sum_s * inv
    mt = sum_t * inv
    # var = E[x^2] - (E[x])^2  (matches unbiased=False)
    vs = sum_s2 * inv - ms * ms
    vt = sum_t2 * inv - mt * mt

    # Small numeric negatives can happen due to fp32 rounding.
    vs = vs.clamp(min=0.0)
    vt = vt.clamp(min=0.0)

    mean_mse = F.mse_loss(ms, mt)
    var_mse = F.mse_loss(vs, vt)
    # W2^2 (diag-Gaussian) uses std (sqrt variance), not variance.
    # Add epsilon to avoid unstable gradients when var is very small.
    std_s = torch.sqrt(vs + 1e-8)
    std_t = torch.sqrt(vt + 1e-8)
    std_mse = F.mse_loss(std_s, std_t)
    token_mse = diff2_sum / (denom * d)

    loss_type = str(loss_type or "mean_var").strip().lower()
    if loss_type == "mse":
        total = token_mse
    elif loss_type == "mean_var":
        total = mean_mse + var_weight * var_mse
    elif loss_type in {"w2", "mean_std"}:
        # In diagonal-Gaussian W2^2, the scale term is std_mse with coefficient 1.0.
        # We keep `var_weight` as the generic "scale weight" for ablations (e.g., mean-only: set it to 0).
        total = mean_mse + var_weight * std_mse
    else:
        raise ValueError(f"Unknown distribution alignment loss_type={loss_type!r} (expected 'mse', 'mean_var', 'w2').")

    return {
        "total": total,
        "mean_mse": mean_mse,
        "var_mse": var_mse,
        "std_mse": std_mse,
        "w2_sq": mean_mse + std_mse,
        "token_mse": token_mse,
    }


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

    # ==================== Ablations（论文消融开关，尽量一键启用）====================
    # - w/o pruner: top-k attention baseline
    # - w/o adapter: 关闭 delayed repair 注入（同时关闭 repair loss，避免变成“无 adapter 的对齐训练”）
    # - w/o repair loss: 保留 delayed repair 注入，但关掉 teacher-student 对齐损失
    # - mean-only: soft repair 对齐只对齐均值（α=0）
    ab_w_o_pruner_topk = bool(method_cfg.get("ablation_w_o_pruner_topk_attn", False))
    ab_w_o_adapter = bool(method_cfg.get("ablation_w_o_adapter", False))
    ab_w_o_repair_loss = bool(method_cfg.get("ablation_w_o_repair_loss", False))
    ab_mean_only = bool(method_cfg.get("ablation_repair_mean_only", False))

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

    if ab_w_o_adapter:
        # 纯 pruning baseline（不做 delayed repair，也不做 repair loss）
        teacher_forward_enable = False
        repair_loss_weight = 0.0
    if ab_w_o_repair_loss:
        # 保留 delayed repair 注入，但不使用 repair 对齐监督
        teacher_forward_enable = False
        repair_loss_weight = 0.0
    if ab_mean_only:
        repair_var_weight = 0.0

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
        'pruning_mode': ('topk_attn' if ab_w_o_pruner_topk else 'normal'),
        'target_token_num': method_cfg.get('target_token_num', None),
        'apply_repair': (not ab_w_o_adapter),
        'capture_layers': capture_layers,
    }

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
        student_caps_pre = getattr(output, "captured_pre_repair", None) or output.captured
        teacher_caps = teacher_output.captured
        layer_losses = []
        layer_mean_mse = []
        layer_var_mse = []
        layer_std_mse = []
        layer_token_mse = []
        layer_pre_losses = []
        per_layer = {}
        per_layer_mean = {}
        per_layer_var = {}
        per_layer_std = {}
        per_layer_token = {}
        per_layer_pre = {}
        per_layer_gain = {}
        for layer_idx in capture_layers:
            if layer_idx not in student_caps or layer_idx not in teacher_caps:
                continue
            s = student_caps[layer_idx]
            s_pre = student_caps_pre.get(layer_idx, None) if isinstance(student_caps_pre, dict) else None
            t = teacher_caps[layer_idx]
            m = s["mask"] * t["mask"]
            details = compute_distribution_alignment_details(
                s["h"],
                t["h"],
                m,
                loss_type=repair_loss_type,
                var_weight=repair_var_weight,
            )
            layer_loss = details["total"]
            layer_losses.append(layer_loss)
            layer_mean_mse.append(details["mean_mse"])
            layer_var_mse.append(details["var_mse"])
            if "std_mse" in details:
                layer_std_mse.append(details["std_mse"])
            layer_token_mse.append(details["token_mse"])

            per_layer[layer_idx] = float(details["total"].detach().item())
            per_layer_mean[layer_idx] = float(details["mean_mse"].detach().item())
            per_layer_var[layer_idx] = float(details["var_mse"].detach().item())
            if "std_mse" in details:
                per_layer_std[layer_idx] = float(details["std_mse"].detach().item())
            per_layer_token[layer_idx] = float(details["token_mse"].detach().item())

            # 诊断：修复前的 gap（不参与反传）
            if s_pre is not None:
                with torch.no_grad():
                    pre_details = compute_distribution_alignment_details(
                        s_pre["h"], t["h"], s_pre["mask"] * t["mask"],
                        loss_type=repair_loss_type,
                        var_weight=repair_var_weight,
                    )
                pre_total = pre_details["total"]
                layer_pre_losses.append(pre_total)
                pre_val = float(pre_total.detach().item())
                post_val = float(details["total"].detach().item())
                per_layer_pre[layer_idx] = pre_val
                per_layer_gain[layer_idx] = pre_val - post_val

        if layer_losses:
            repair_loss = torch.stack(layer_losses).mean()
            mean_mse = torch.stack(layer_mean_mse).mean()
            var_mse = torch.stack(layer_var_mse).mean()
            std_mse = torch.stack(layer_std_mse).mean() if layer_std_mse else torch.tensor(0.0, device=device)
            token_mse = torch.stack(layer_token_mse).mean()
            pre_repair = torch.stack(layer_pre_losses).mean() if layer_pre_losses else torch.tensor(0.0, device=device)
        else:
            repair_loss = torch.tensor(0.0, device=device)
            mean_mse = torch.tensor(0.0, device=device)
            var_mse = torch.tensor(0.0, device=device)
            std_mse = torch.tensor(0.0, device=device)
            token_mse = torch.tensor(0.0, device=device)
            pre_repair = torch.tensor(0.0, device=device)

        losses['repair_loss'] = repair_loss
        stats['raw_repair_loss'] = float(repair_loss.detach().item())
        stats['repair_loss_type'] = repair_loss_type
        stats['raw_repair_mean_mse'] = float(mean_mse.detach().item())
        stats['raw_repair_var_mse'] = float(var_mse.detach().item())
        stats['raw_repair_std_mse'] = float(std_mse.detach().item())
        stats['raw_repair_w2_sq'] = float((mean_mse + std_mse).detach().item())
        stats['raw_repair_token_mse'] = float(token_mse.detach().item())
        stats['raw_repair_pre'] = float(pre_repair.detach().item())
        stats['raw_repair_gain'] = float((pre_repair - repair_loss).detach().item())
        stats['repair_var_weight'] = float(repair_var_weight)
        stats['repair_per_layer'] = per_layer
        stats['repair_mean_per_layer'] = per_layer_mean
        stats['repair_var_per_layer'] = per_layer_var
        stats['repair_std_per_layer'] = per_layer_std
        stats['repair_token_per_layer'] = per_layer_token
        stats['repair_pre_per_layer'] = per_layer_pre
        stats['repair_gain_per_layer'] = per_layer_gain
    else:
        losses['repair_loss'] = torch.tensor(0.0, device=device)
        stats['raw_repair_loss'] = 0.0
        stats['raw_repair_mean_mse'] = 0.0
        stats['raw_repair_var_mse'] = 0.0
        stats['raw_repair_std_mse'] = 0.0
        stats['raw_repair_w2_sq'] = 0.0
        stats['raw_repair_token_mse'] = 0.0
        stats['raw_repair_pre'] = 0.0
        stats['raw_repair_gain'] = 0.0

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
            # 直接使用 cumulative_ratio 计算平均算力（最贴近“全层平均算力”的定义）
            # avg = (n0*1 + n1*r1 + n2*r2 + ... ) / total_layers
            # 其中 r_i 为 pruning_layers[i] 处的 cumulative_mask mean
            avg_kept = torch.tensor(0.0, device=device)
            avg_kept = avg_kept + segment_lengths[0] * 1.0
            for i in range(n_pruning_layers):
                avg_kept = avg_kept + segment_lengths[i + 1] * cumulative_ratios[i]
            avg_kept = avg_kept / total_layers
            sparsity_loss = torch.abs(avg_kept - target_ratio)
            stats['avg_kept_ratio'] = avg_kept.item()
        else:
            # harmonic mean 近似
            # 先从 cumulative_ratios 恢复每层的条件独立保留率 p_i = r_i / r_{i-1}
            independent_ratios = []
            for i in range(n_pruning_layers):
                if i == 0:
                    p_i = cumulative_ratios[i]
                else:
                    p_i = cumulative_ratios[i] / cumulative_ratios[i - 1].clamp(min=1e-6)
                independent_ratios.append(p_i.clamp(min=1e-6, max=1.0))
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
        # repair_loss 使用的是“跨 rank 的全局统计”（global masked moments）。
        # 由于本仓库的梯度同步在 sync_gradients() 中做的是 *平均*（SUM / world_size），
        # 这里把 repair_loss 乘以 world_size，使最终等效梯度与“全局 batch 上计算一次 loss”一致。
        world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        weighted_losses['repair_loss'] = (losses['repair_loss'] * float(world_size)) * repair_weight
    if 'sparsity_loss' in losses:
        weighted_losses['sparsity_loss'] = losses['sparsity_loss'] * sparsity_weight
    if 'entropy_loss' in losses:
        weighted_losses['entropy_loss'] = losses['entropy_loss'] * entropy_weight
    if 'tightening_loss' in losses:
        weighted_losses['tightening_loss'] = losses['tightening_loss']  # 权重已在计算时应用
    # 不再使用判别器相关损失（如需可恢复旧逻辑）

    return {
        'losses': weighted_losses,
        'stats': stats,
        'pruning_infos': {idx: info for idx, info in output.pruning_infos.items()} if output.pruning_infos else None,
    }
