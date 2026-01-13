"""Vision Token Pruning - 批量训练函数

实现Layer-wise Pruning的批量训练。
通过将图像resize到统一大小，实现真正的批量处理，大幅提升训练速度。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List
from collections import defaultdict

from .utils import (
    extract_target_hidden_states_batch,
    compute_task_loss_batch,
    register_multi_layer_hooks_batch,
    remove_hooks,
    get_current_sparsity_weight
)


# === DEBUG 工具函数 ===
def debug_check_tensor(tensor, name, step=None):
    """检查tensor是否包含NaN或Inf"""
    if tensor is None:
        return False

    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    step_str = f"[Step {step}] " if step is not None else ""

    if has_nan:
        nan_count = torch.isnan(tensor).sum().item()
        valid_vals = tensor[~torch.isnan(tensor)]
        if valid_vals.numel() > 0:
            print(f"{step_str}[DEBUG NaN] {name}: nan_count={nan_count}, "
                  f"valid_min={valid_vals.min().item():.4f}, valid_max={valid_vals.max().item():.4f}")
        else:
            print(f"{step_str}[DEBUG NaN] {name}: ALL VALUES ARE NaN!")
        return True

    if has_inf:
        inf_count = torch.isinf(tensor).sum().item()
        print(f"{step_str}[DEBUG Inf] {name}: inf_count={inf_count}")
        return True

    return False


def train_step(batch: List[Any], device: torch.device, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """批量版本的训练step函数

    通过将图像resize到统一大小，实现真正的批量处理，大幅提升训练速度。

    架构：
    1. 批量预处理：将所有图像resize到336x336，统一vision token数量
    2. Layer-wise Pruning（LLM内部）：在Layer 5/15/25分别剪枝
    3. GAN对抗训练：Discriminator判别real/fake

    参数:
        batch: 数据batch，每个元素包含 image, question, answer
        device: 设备
        info: 包含config, models等的字典

    返回:
        损失字典，包含优化器组：layer_pruners, discriminator
    """
    config = info["config"]
    backbone = info["models"]["backbone"]
    layer_pruners = info["models"]["layer_pruners"]
    discriminator = info["models"]["discriminator"]
    current_step = info["global_batch_index"]

    # === 配置参数 ===
    disc_target_layers = config["method_settings"]["disc_target_layers"]
    disc_reinit_prob = config["method_settings"]["disc_reinit_prob"]
    total_steps = config["trainer_settings"]["dl_settings"]["epochs"] * info.get("total_planned_batches", 1000)

    # === Discriminator随机重初始化 ===
    if torch.rand(1).item() < disc_reinit_prob:
        config["logger"].info(f"[Step {current_step}] Discriminator reinit triggered")
        discriminator._init_weights()

    # === Temperature Annealing ===
    temperature = config['method_settings'].get('temperature', 1.0)
    temperature_min = config['method_settings'].get('temperature_min', 0.1)
    anneal_rate = config['method_settings'].get('temperature_anneal_rate', 0.5)

    progress = current_step / total_steps
    if progress < anneal_rate:
        current_temp = temperature - (progress / anneal_rate) * (temperature - temperature_min)
    else:
        current_temp = temperature_min

    for layer_idx in layer_pruners.get_all_layers():
        pruner = layer_pruners.get_pruner(layer_idx)
        pruner.set_temperature(current_temp)

    # === 初始化损失累加器 ===
    layer_pruners_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    disc_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    stats = defaultdict(float)

    batch_size = len(batch)

    # ========== Phase 1: 批量预处理 ==========
    images = [sample["image"] for sample in batch]
    questions = [sample["question"] for sample in batch]
    answers = [sample["answer"] for sample in batch]

    with torch.no_grad():
        emb_info = backbone.preprocess_batch(images, questions, answers)
        original_embeddings = emb_info['embeddings']  # (batch, seq_len, 4096)
        original_vision_pos = emb_info['vision_token_positions']  # (start, end) 所有样本相同
        answer_positions = emb_info['answer_token_positions']  # List of (start, end)
        attention_mask = emb_info['attention_mask']  # (batch, seq_len)

        # 获取未投影的vision features
        vision_features_raw = emb_info['raw_vision_features']  # (batch, num_vision, 1024)

        if vision_features_raw is None:
            raise ValueError("backbone未返回raw_vision_features，请检查backbone实现")

        # 提取question embeddings（用于layer pruners的cross-attention）
        v_start, v_end = original_vision_pos

        # 投影vision features到LLM维度
        vision_features_projected = backbone.model.multi_modal_projector(vision_features_raw)  # (batch, num_vision, 4096)

    # 构建embeddings（vision部分已经在preprocess_batch中处理好了）
    embeddings_for_forward = original_embeddings
    new_vision_pos = original_vision_pos
    new_attention_mask = attention_mask

    # 提取question embeddings（用于layer pruners）
    # 注意：只取question部分，排除answer token
    # answer_positions是List of (start, end)，取最小的answer_start作为截止点
    min_answer_start = min(pos[0] for pos in answer_positions)
    question_embeddings = embeddings_for_forward[:, v_end+1:min_answer_start, :]  # (batch, question_len, 4096)

    # ========== Phase 2: Layer-wise Pruning Forward（带hooks） ==========

    # 2.1 创建mask收集器
    pruning_masks = []

    # 2.2 注册hooks
    use_attn_residual = config["method_settings"].get("use_attn_residual", False)
    handles = register_multi_layer_hooks_batch(
        backbone,
        layer_pruners,
        new_vision_pos,
        question_embeddings,
        mask_collector=pruning_masks,
        use_attn_residual=use_attn_residual
    )

    try:
        # 2.3 Forward（fake sample - 带剪枝）
        layer_pruners.train()
        result_fake = backbone.forward(
            embeddings=embeddings_for_forward,
            attention_mask=new_attention_mask,
            output_hidden_states=True
        )

        # === DEBUG: 检查forward输出 ===
        debug_check_tensor(result_fake['logits'], "result_fake['logits']", current_step)

        # 2.4 提取hidden states from target layers
        fake_hidden_list = extract_target_hidden_states_batch(
            result_fake['all_hidden_states'],
            answer_positions,
            disc_target_layers,
            batch_size=batch_size,
            attention_mask=new_attention_mask
        )

        # === DEBUG: 检查fake hidden states ===
        for i, fh in enumerate(fake_hidden_list):
            debug_check_tensor(fh, f"fake_hidden_list[{i}]", current_step)

    finally:
        remove_hooks(handles)

    # 2.5 Forward（real sample - 无剪枝）
    with torch.no_grad():
        result_real = backbone.forward(
            embeddings=original_embeddings,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        real_hidden_list = extract_target_hidden_states_batch(
            result_real['all_hidden_states'],
            answer_positions,
            disc_target_layers,
            batch_size=batch_size,
            attention_mask=attention_mask
        )

    # ========== Phase 3: Discriminator Judgment ==========

    discriminator.eval()

    # 3.1 判别fake（用于generator loss）
    for p in discriminator.parameters():
        p.requires_grad = False

    fake_pred_for_gen = discriminator(fake_hidden_list)  # (batch, seq_len)

    # === DEBUG: 检查discriminator输出 ===
    debug_check_tensor(fake_pred_for_gen, "fake_pred_for_gen", current_step)

    for p in discriminator.parameters():
        p.requires_grad = True

    # 3.2 判别real
    real_pred = discriminator(real_hidden_list)

    # === DEBUG: 检查real_pred ===
    debug_check_tensor(real_pred, "real_pred", current_step)

    # ========== Phase 4: Loss Computation ==========

    # --- Layer Pruners Loss ---

    # 1. Task Loss: 保持任务性能
    task_loss = compute_task_loss_batch(
        result_fake['logits'],
        answer_positions,
        answers,
        backbone.processor,
        attention_mask=new_attention_mask
    )
    layer_pruners_losses["task_loss"] = task_loss

    # === DEBUG: 检查task_loss ===
    debug_check_tensor(task_loss, "task_loss (before weight)", current_step)

    # 2. Adversarial loss (使用 logits，数值更稳定)
    adv_loss = F.binary_cross_entropy_with_logits(
        fake_pred_for_gen,
        torch.ones_like(fake_pred_for_gen),
        reduction='mean'
    )
    layer_pruners_losses["adv_loss"] = adv_loss

    # === DEBUG: 检查adv_loss ===
    debug_check_tensor(adv_loss, "adv_loss (before weight)", current_step)

    # 3. Sparsity Loss
    if len(pruning_masks) > 0:
        # === DEBUG: 检查pruning_masks ===
        for i, mask in enumerate(pruning_masks):
            if debug_check_tensor(mask, f"pruning_masks[{i}]", current_step):
                print(f"[Step {current_step}] [DEBUG] pruning_masks[{i}] stats: "
                      f"shape={mask.shape}, min={mask.min().item():.4f}, max={mask.max().item():.4f}, "
                      f"mean={mask.mean().item():.4f}")

        target_sparsity = config['method_settings'].get('target_sparsity')
        use_token_num_target = config['method_settings'].get('use_token_num_target')
        sparsity_loss_only_on_excess = config['method_settings'].get('sparsity_loss_only_on_excess')

        n_vision = pruning_masks[0].shape[1]

        if use_token_num_target:
            target_token_num = config['method_settings'].get('target_token_num', 128)
            target_kept_ratio = target_token_num / n_vision
        else:
            target_kept_ratio = 1.0 - target_sparsity

        # 计算每层的保留率
        kept_ratios = [mask.mean().to(device) for mask in pruning_masks]
        avg_kept_ratio = torch.stack(kept_ratios).mean()
        final_mask = pruning_masks[-1]
        final_kept_ratio = final_mask.mean()

        # === DEBUG: 检查kept_ratios ===
        debug_check_tensor(avg_kept_ratio, "avg_kept_ratio", current_step)

        if sparsity_loss_only_on_excess:
            excess = torch.relu(avg_kept_ratio - target_kept_ratio)
            sparsity_constraint_loss = excess.to(device).pow(2)
        else:
            sparsity_constraint_loss = (avg_kept_ratio - target_kept_ratio).to(device).pow(2)

        layer_pruners_losses["sparsity_loss"] = sparsity_constraint_loss

        # === DEBUG: 检查sparsity_loss ===
        debug_check_tensor(sparsity_constraint_loss, "sparsity_loss (before weight)", current_step)

        # Token Count Loss
        token_count_loss = avg_kept_ratio.to(device)
        layer_pruners_losses["token_count_loss"] = token_count_loss

        # Binarization Loss: 鼓励 mask 接近 0 或 1
        # binary_term = mask * (1 - mask) 在 mask=0 或 1 时为 0，在 mask=0.5 时最大 (0.25)
        binarization_loss = torch.tensor(0.0, device=device)
        for mask in pruning_masks:
            binary_term = (mask * (1 - mask)).mean()
            binarization_loss = binarization_loss + binary_term.to(device)
        binarization_loss = binarization_loss / len(pruning_masks)
        layer_pruners_losses["binarization_loss"] = binarization_loss

        # 统计信息
        pruning_layers = layer_pruners.get_all_layers()
        for idx, mask in enumerate(pruning_masks):
            layer_num = pruning_layers[idx]
            stats[f"L{layer_num}_kept"] = mask.mean().item()
        stats["avg_kept_ratio"] = avg_kept_ratio.item()
        stats["final_kept_ratio"] = final_kept_ratio.item()
        stats["final_token_count"] = final_mask.sum().item() / batch_size
        stats["target_kept_ratio"] = target_kept_ratio

    # --- Discriminator Loss ---

    discriminator.train()

    # Real loss (使用 logits)
    disc_losses["real_loss"] = F.binary_cross_entropy_with_logits(
        real_pred,
        torch.ones_like(real_pred),
        reduction='mean'
    )

    # === DEBUG: 检查disc real_loss ===
    debug_check_tensor(disc_losses["real_loss"], "disc_real_loss", current_step)

    # Fake loss (使用 logits)
    fake_hidden_detached = [h.detach() for h in fake_hidden_list]
    fake_pred_for_disc = discriminator(fake_hidden_detached)
    disc_losses["fake_loss"] = F.binary_cross_entropy_with_logits(
        fake_pred_for_disc,
        torch.zeros_like(fake_pred_for_disc),
        reduction='mean'
    )

    # === DEBUG: 检查disc fake_loss ===
    debug_check_tensor(disc_losses["fake_loss"], "disc_fake_loss", current_step)

    # 判别器准确率 (对 logits 应用 sigmoid 后判断)
    real_prob = torch.sigmoid(real_pred)
    fake_prob = torch.sigmoid(fake_pred_for_disc)
    real_correct = (real_prob > 0.5).float().mean()
    fake_correct = (fake_prob < 0.5).float().mean()
    disc_accuracy = (real_correct + fake_correct) / 2.0
    stats["disc_accuracy"] = disc_accuracy.item()
    stats["disc_real_acc"] = real_correct.item()
    stats["disc_fake_acc"] = fake_correct.item()

    # 清理
    del embeddings_for_forward, result_fake, result_real
    del fake_hidden_list, real_hidden_list, fake_hidden_detached
    del fake_pred_for_gen, real_pred, fake_pred_for_disc, pruning_masks

    # ========== Phase 5: 应用权重 ==========

    # Dynamic Loss Weight Scheduling
    progress = current_step / total_steps

    task_weight_start = config['method_settings'].get('task_loss_weight_start', None)
    task_weight_end = config['method_settings'].get('task_loss_weight')
    adv_weight_start = config['method_settings'].get('adv_loss_weight_start', None)
    adv_weight_end = config['method_settings'].get('adv_loss_weight')
    warmup_ratio = config['method_settings'].get('loss_weight_warmup_ratio', 0.0)

    if warmup_ratio > 0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2

        if task_weight_start is not None:
            task_weight = task_weight_start + (task_weight_end - task_weight_start) * cosine_factor
        else:
            task_weight = task_weight_end

        if adv_weight_start is not None:
            adv_weight = adv_weight_start + (adv_weight_end - adv_weight_start) * cosine_factor
        else:
            adv_weight = adv_weight_end
    else:
        task_weight = task_weight_end
        adv_weight = adv_weight_end

    sparsity_weight = get_current_sparsity_weight(config, current_step, total_steps)
    token_count_weight = config['method_settings'].get('token_count_loss_weight')
    binarization_weight = config['method_settings'].get('binarization_loss_weight', 0.0)

    stats["current_task_weight"] = float(task_weight)
    stats["current_adv_weight"] = float(adv_weight)
    stats["current_sparsity_weight"] = float(sparsity_weight)

    # 应用权重
    layer_pruners_losses["adv_loss"] = layer_pruners_losses["adv_loss"] * adv_weight
    layer_pruners_losses["task_loss"] = layer_pruners_losses["task_loss"] * task_weight
    if "sparsity_loss" in layer_pruners_losses:
        layer_pruners_losses["sparsity_loss"] = layer_pruners_losses["sparsity_loss"] * sparsity_weight
    if "token_count_loss" in layer_pruners_losses:
        layer_pruners_losses["token_count_loss"] = layer_pruners_losses["token_count_loss"] * token_count_weight
    if "binarization_loss" in layer_pruners_losses:
        layer_pruners_losses["binarization_loss"] = layer_pruners_losses["binarization_loss"] * binarization_weight

    # 确保tensor在正确设备上
    target_device = next(layer_pruners.parameters()).device
    for losses_dict in [layer_pruners_losses, disc_losses]:
        for k in losses_dict:
            if isinstance(losses_dict[k], torch.Tensor):
                losses_dict[k] = losses_dict[k].to(target_device)

    torch.cuda.empty_cache()

    return {
        "discriminator": dict(disc_losses),
        "layer_pruners": dict(layer_pruners_losses),
        "metrics": stats
    }
