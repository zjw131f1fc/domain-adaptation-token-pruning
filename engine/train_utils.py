"""训练相关工具函数"""

import math
import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from engine.data_utils import preprocess_batch, preprocess_batch_qwen2vl


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
    adversarial_mode = method_cfg.get('adversarial_mode', 'discriminator')

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

    # 构建 forward 参数
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
        'return_pruning_info': True,
        'detach_h_fake_for_adv': detach_adv_from_pruner,
    }

    # Qwen2-VL 需要 image_grid_thw
    if 'image_grid_thw' in inputs:
        forward_kwargs['image_grid_thw'] = inputs['image_grid_thw']

    output = model(**forward_kwargs)

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

    # 2. 如果有剪枝信息，计算对抗损失（Discriminator 或 MSE）
    if output.pruning_infos and len(output.pruning_infos) > 0:
        h_real_dict = {idx: info['h_real'] for idx, info in output.pruning_infos.items()}
        h_fake_dict = {idx: info['h_fake'] for idx, info in output.pruning_infos.items()}

        warmup_ratio = method_cfg.get('pruner_warmup_ratio', 0.0)
        in_warmup = current_step < total_steps * warmup_ratio
        gan_weight = 0.0 if in_warmup else 1.0
        stats['in_warmup'] = in_warmup

        if adversarial_mode == 'mse':
            # === MSE 模式：直接约束 h_real 和 h_fake 的一致性 ===
            mse_loss_type = method_cfg.get('mse_loss_type', 'mse')
            mse_normalize = method_cfg.get('mse_normalize', False)
            has_h_corrected = all(info.get('h_corrected') is not None for info in output.pruning_infos.values())
            h_corrected_dict = {idx: info['h_corrected'] for idx, info in output.pruning_infos.items()} if has_h_corrected else None
            h_align_dict = h_corrected_dict if has_h_corrected else h_fake_dict
            stats['align_source'] = 'h_corrected' if has_h_corrected else 'h_fake'

            alignment_loss_total = torch.tensor(0.0, device=device)
            n_samples = 0

            # 每层的对齐指标
            mse_per_layer = {}
            cosine_per_layer = {}
            l1_per_layer = {}

            for layer_idx in h_real_dict:
                h_real_list = h_real_dict[layer_idx]
                h_fake_list = h_align_dict[layer_idx]

                layer_mse = 0.0
                layer_cosine = 0.0
                layer_l1 = 0.0
                layer_samples = 0

                # 逐样本计算
                for h_real, h_fake in zip(h_real_list, h_fake_list):
                    # h_real, h_fake: (heads, n_ans, head_dim)
                    h_real_flat = h_real.reshape(-1)
                    h_fake_flat = h_fake.reshape(-1)

                    # 可选：归一化
                    if mse_normalize:
                        h_real_norm = F.normalize(h_real_flat, dim=0)
                        h_fake_norm = F.normalize(h_fake_flat, dim=0)
                    else:
                        h_real_norm = h_real_flat
                        h_fake_norm = h_fake_flat

                    # 计算损失
                    if mse_loss_type == 'l1':
                        sample_loss = F.l1_loss(h_fake_norm, h_real_norm)
                    elif mse_loss_type == 'smooth_l1':
                        sample_loss = F.smooth_l1_loss(h_fake_norm, h_real_norm)
                    elif mse_loss_type == 'cosine':
                        # 余弦相似度损失: 1 - cosine_similarity
                        cosine_sim = F.cosine_similarity(h_fake_flat.unsqueeze(0), h_real_flat.unsqueeze(0))
                        sample_loss = 1.0 - cosine_sim.mean()
                    else:  # mse
                        sample_loss = F.mse_loss(h_fake_norm, h_real_norm)

                    alignment_loss_total = alignment_loss_total + sample_loss

                    # 计算指标（用于监控，不参与梯度）
                    with torch.no_grad():
                        layer_mse += F.mse_loss(h_fake_flat, h_real_flat).item()
                        cosine_sim = F.cosine_similarity(h_fake_flat.unsqueeze(0), h_real_flat.unsqueeze(0))
                        layer_cosine += cosine_sim.mean().item()
                        layer_l1 += F.l1_loss(h_fake_flat, h_real_flat).item()
                        layer_samples += 1

                n_samples = len(h_real_list)

                # 记录每层指标
                if layer_samples > 0:
                    mse_per_layer[layer_idx] = layer_mse / layer_samples
                    cosine_per_layer[layer_idx] = layer_cosine / layer_samples
                    l1_per_layer[layer_idx] = layer_l1 / layer_samples

            # 除以样本数和层数
            n_layers = len(h_real_dict)
            alignment_loss = alignment_loss_total / (n_samples * n_layers)

            # 使用专门的 mse_loss_weight，如果没有则使用 adv_loss_weight
            losses['adv_loss'] = alignment_loss * gan_weight
            stats['raw_adv_loss'] = alignment_loss.item()
            stats['adversarial_mode'] = 'mse'
            stats['mse_loss_type'] = mse_loss_type

            # 记录对齐指标
            stats['mse_per_layer'] = mse_per_layer
            stats['cosine_per_layer'] = cosine_per_layer
            stats['l1_per_layer'] = l1_per_layer

            # 计算整体指标
            if mse_per_layer:
                stats['avg_mse'] = sum(mse_per_layer.values()) / len(mse_per_layer)
                stats['avg_cosine'] = sum(cosine_per_layer.values()) / len(cosine_per_layer)
                stats['avg_l1'] = sum(l1_per_layer.values()) / len(l1_per_layer)

            # MSE 模式下没有判别器损失
            losses['disc_loss'] = torch.tensor(0.0, device=device)
            stats['raw_disc_loss'] = 0.0
        else:
            # === Discriminator 模式：使用判别器进行对抗训练 ===
            loss_type = method_cfg.get('disc_loss_type', 'bce')
            gp_weight = method_cfg.get('disc_gp_weight', 10.0)
            has_attn = all(('h_real_attn' in info and 'h_fake_attn' in info) for info in output.pruning_infos.values())
            if not has_attn:
                raise ValueError("Discriminator expects attn features, but h_real_attn/h_fake_attn not found in pruning_infos.")
            h_disc_real_dict = {idx: info['h_real_attn'] for idx, info in output.pruning_infos.items()}
            h_disc_fake_dict = {idx: info['h_fake_attn'] for idx, info in output.pruning_infos.items()}
            stats['disc_source'] = 'attn'

            # 获取 disc_manager（可能被 DDP 包装）
            disc_manager = model.disc_manager.module if hasattr(model.disc_manager, 'module') else model.disc_manager

            adv_loss = disc_manager.compute_adv_loss(h_disc_fake_dict, loss_type=loss_type)
            losses['adv_loss'] = adv_loss * gan_weight
            stats['raw_adv_loss'] = adv_loss.item()

            disc_loss = disc_manager.compute_disc_loss(h_disc_real_dict, h_disc_fake_dict, loss_type=loss_type, gp_weight=gp_weight)
            losses['disc_loss'] = disc_loss * gan_weight
            stats['raw_disc_loss'] = disc_loss.item()

            acc_info = disc_manager.compute_accuracy(h_disc_real_dict, h_disc_fake_dict)
            stats['disc_accuracy'] = acc_info['overall']
            stats['disc_real_acc'] = acc_info['real_acc']
            stats['disc_fake_acc'] = acc_info['fake_acc']
            stats['disc_per_layer'] = acc_info['per_layer']
            stats['adversarial_mode'] = 'discriminator'

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

        # 收集各层的累积保留率
        cumulative_ratios = []
        for i, layer_idx in enumerate(pruning_layers):
            cumulative_mask = output.pruning_infos[layer_idx]['cumulative_mask']
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
            cumulative_product = torch.tensor(1.0, device=device, requires_grad=True)
            for i in range(n_pruning_layers):
                cumulative_product = cumulative_product * independent_ratios[i]
                avg_kept = avg_kept + n_segments[i + 1] * cumulative_product
            avg_kept = avg_kept / total_layers
            sparsity_loss = torch.abs(avg_kept - target_ratio)

            # 调试：检查梯度是否能传导
            # 只在训练模式下检查梯度（eval 时在 torch.no_grad() 上下文中，没有梯度是正常的）
            if torch.is_grad_enabled() and not sparsity_loss.requires_grad:
                print(f"[WARNING] sparsity_loss has no grad! step={current_step}")
                print(f"  avg_kept.requires_grad: {avg_kept.requires_grad}")
                for i, p in enumerate(independent_ratios):
                    print(f"  independent_ratios[{i}].requires_grad: {p.requires_grad}, grad_fn: {p.grad_fn}")

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

            stats['harmonic_mean'] = hm.item()

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
    adv_weight = method_cfg.get('adv_loss_weight', 0.5)
    if adversarial_mode == 'mse':
        adv_weight = method_cfg.get('mse_loss_weight', adv_weight)
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
    if 'adapter_delta_loss' in losses:
        weighted_losses['adapter_delta_loss'] = losses['adapter_delta_loss'] * adapter_delta_weight
    if 'tightening_loss' in losses:
        weighted_losses['tightening_loss'] = losses['tightening_loss']  # 权重已在计算时应用
    if 'disc_loss' in losses:
        weighted_losses['disc_loss'] = losses['disc_loss']

    return {
        'losses': weighted_losses,
        'stats': stats,
        'pruning_infos': {idx: info for idx, info in output.pruning_infos.items()} if output.pruning_infos else None,
    }
