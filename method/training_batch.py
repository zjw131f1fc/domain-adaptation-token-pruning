"""Vision Token Pruning - Batch化训练函数

实现真正的batch化训练（当enable_true_batch=True时启用）
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from collections import defaultdict
from .utils import (
    extract_text_hidden_states,
    weighted_pool_text_hidden_states,
    add_position_aware_noise_to_pooled,
    compute_task_loss,
    register_multi_layer_hooks_batch,
    remove_hooks,
    replace_vision_tokens_in_embeddings_batch,
    update_temperature_for_all,
    get_current_sparsity_weight
)


def train_step_batch(batch: List[Any], device: torch.device, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """两阶段剪枝的训练step函数（Batch化版本）

    与train_step的主要区别：
    - 使用 backbone.preprocess_batch 一次性处理整个batch
    - Backbone forward一次性处理所有样本
    - 要求启用 enable_true_batch 配置

    架构：
    1. Token Merge（LLM输入前）：576 tokens → ~288 tokens
    2. Layer-wise Pruning（LLM内部）：在指定层分别剪枝
    3. GAN对抗训练：Discriminator判别real/fake

    参数:
        batch: 数据batch（List of samples）
        device: 设备
        info: 包含config, models等的字典

    返回:
        损失字典，包含三个optimizer组：discriminator, token_merger, layer_pruners
    """
    config = info["config"]
    backbone = info["models"]["backbone"]
    token_merger = info["models"].get("token_merger", None)
    layer_pruners = info["models"]["layer_pruners"]
    discriminator = info["models"]["discriminator"]
    current_step = info["global_batch_index"]

    # 检查是否启用batch化
    enable_true_batch = config["backbone_settings"]["mllm_settings"].get("enable_true_batch", False)
    if not enable_true_batch:
        raise ValueError(
            "train_step_batch requires enable_true_batch=True. "
            "Please set backbone_settings.mllm_settings.enable_true_batch=true in config"
        )

    # === 配置参数 ===
    enable_token_merger = config["method_settings"].get("enable_token_merger", True)
    disc_target_layers = config["method_settings"]["disc_target_layers"]
    disc_reinit_prob = config["method_settings"]["disc_reinit_prob"]
    total_steps = config["trainer_settings"]["dl_settings"]["epochs"] * info.get("total_planned_batches", 1000)

    # === Discriminator随机重初始化 ===
    if torch.rand(1).item() < disc_reinit_prob:
        config["logger"].info(f"[Step {current_step}] Discriminator reinit triggered")
        discriminator._init_weights()

    # === Temperature Annealing ===
    if token_merger is not None and enable_token_merger:
        update_temperature_for_all(token_merger, layer_pruners, config, current_step, total_steps)
    else:
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
    token_merger_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    layer_pruners_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    disc_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    stats = defaultdict(float)

    batch_size = len(batch)

    # === 提取batch数据 ===
    images = [sample["image"] for sample in batch]
    questions = [sample["question"] for sample in batch]
    answers = [sample["answer"] for sample in batch]

    # ========== Phase 1: Batch Preprocess ==========
    with torch.no_grad():
        # 一次性预处理整个batch
        emb_info = backbone.preprocess_batch(images, questions, answers)

        original_embeddings = emb_info['embeddings']  # (batch_size, seq_len, hidden_dim)
        original_vision_pos = emb_info['vision_token_positions']  # (batch_size, 2)
        answer_pos = emb_info['answer_token_positions']  # (batch_size, 2)
        vision_features_raw = emb_info['raw_vision_features']  # (batch_size, n_vision, 1024)

        if vision_features_raw is None:
            raise ValueError("backbone未返回raw_vision_features")

        # 统计vision token数（batch维度相同）
        n_vision_tokens = vision_features_raw.shape[1]
        stats["vision_tokens_mean"] = n_vision_tokens
        stats["vision_tokens_max"] = n_vision_tokens
        stats["vision_tokens_min"] = n_vision_tokens

        # 提取question embeddings
        # 所有样本的vision位置相同（batch化保证）
        v_start, v_end = original_vision_pos[0, 0].item(), original_vision_pos[0, 1].item()

        # 计算answer_start（batch维度可能不同，需要逐样本处理）
        # 但在batch化模式下，通常answer位置也对齐
        # 这里简化处理：假设使用负索引且对齐
        answer_start_abs = answer_pos[0, 0].item()  # 负索引
        if answer_start_abs < 0:
            answer_start_abs = original_embeddings.shape[1] + answer_start_abs

        # 提取question部分（所有样本相同位置）
        question_embeddings_for_merger = original_embeddings[:, v_end+1:answer_start_abs, :]  # (batch_size, q_len, dim)

    # ========== Phase 2: Token Merge（如果启用） ==========
    if enable_token_merger and token_merger is not None:
        token_merger.train()
        if config.method_settings.merger_type in ["question_aware", "fixed_pooling"]:
            merge_result = token_merger(vision_features_raw, question_embeddings_for_merger, use_gumbel=True)
        else:
            merge_result = token_merger(vision_features_raw, use_gumbel=True)
        merged_vision = merge_result['merged_features']  # (batch_size, M, 1024)

        # 投影到LLM维度
        merged_vision = backbone.model.multi_modal_projector(merged_vision)  # (batch_size, M, 4096)

        # 替换vision部分（batch版本）
        embeddings_merged, new_vision_pos, new_attention_mask = replace_vision_tokens_in_embeddings_batch(
            original_embeddings,
            original_vision_pos,
            merged_vision,
            emb_info['attention_mask']
        )
    else:
        # 禁用token merger
        vision_features_projected = backbone.model.multi_modal_projector(vision_features_raw)
        embeddings_merged, new_vision_pos, new_attention_mask = replace_vision_tokens_in_embeddings_batch(
            original_embeddings,
            original_vision_pos,
            vision_features_projected,
            emb_info['attention_mask']
        )

    # 提取question embeddings（用于layer pruners）
    num_removed_tokens = (original_vision_pos[0, 1] - original_vision_pos[0, 0] + 1) - (new_vision_pos[0, 1] - new_vision_pos[0, 0] + 1)
    answer_start_merged = answer_start_abs - num_removed_tokens.item()
    question_embeddings = embeddings_merged[:, new_vision_pos[0, 1].item()+1:answer_start_merged, :]  # (batch_size, q_len, dim)

    # ========== Phase 3: Layer-wise Pruning Forward（带hooks） ==========
    pruning_masks = []

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
        # Forward（fake sample - 带剪枝）
        layer_pruners.train()
        result_fake = backbone.forward(
            embeddings=embeddings_merged,
            attention_mask=new_attention_mask,
            output_hidden_states=True
        )

        # 提取text hidden states（batch维度）
        fake_hidden_list = []
        for layer_idx in disc_target_layers:
            hidden = result_fake['all_hidden_states'][layer_idx]  # (batch_size, seq_len, dim)
            # 对每个样本提取text部分
            batch_text_hiddens = []
            for b in range(batch_size):
                v_start_b = new_vision_pos[b, 0].item()
                v_end_b = new_vision_pos[b, 1].item()
                text_before = hidden[b:b+1, :v_start_b, :]
                text_after = hidden[b:b+1, v_end_b+1:, :]
                text_hidden_b = torch.cat([text_before, text_after], dim=1)
                batch_text_hiddens.append(text_hidden_b)
            fake_hidden_list.append(torch.cat(batch_text_hiddens, dim=0))  # (batch_size, text_len, dim)

    finally:
        remove_hooks(handles)

    # Forward（real sample - 无剪枝）
    with torch.no_grad():
        result_real = backbone.forward(
            embeddings=original_embeddings,
            attention_mask=emb_info['attention_mask'],
            output_hidden_states=True
        )

        real_hidden_list = []
        for layer_idx in disc_target_layers:
            hidden = result_real['all_hidden_states'][layer_idx]
            batch_text_hiddens = []
            for b in range(batch_size):
                v_start_b = original_vision_pos[b, 0].item()
                v_end_b = original_vision_pos[b, 1].item()
                text_before = hidden[b:b+1, :v_start_b, :]
                text_after = hidden[b:b+1, v_end_b+1:, :]
                text_hidden_b = torch.cat([text_before, text_after], dim=1)
                batch_text_hiddens.append(text_hidden_b)
            real_hidden_list.append(torch.cat(batch_text_hiddens, dim=0))

    # ========== Phase 4: Discriminator Judgment ==========
    start_weight = config["method_settings"].get("disc_pool_start_weight", 0.4)
    end_weight = config["method_settings"].get("disc_pool_end_weight", 1.0)
    noise_scale_start = config["method_settings"].get("disc_noise_scale_start", 0.05)
    noise_scale_end = config["method_settings"].get("disc_noise_scale_end", 0.01)

    fake_hidden_pooled = weighted_pool_text_hidden_states(
        fake_hidden_list,
        start_weight=start_weight,
        end_weight=end_weight,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        training=True
    )

    real_hidden_pooled = weighted_pool_text_hidden_states(
        real_hidden_list,
        start_weight=start_weight,
        end_weight=end_weight,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        training=True
    )

    disc_noise_scale = config["method_settings"].get("disc_noise_scale", 0.0)
    fake_hidden_pooled = add_position_aware_noise_to_pooled(fake_hidden_pooled, noise_scale=disc_noise_scale, training=True)
    real_hidden_pooled = add_position_aware_noise_to_pooled(real_hidden_pooled, noise_scale=disc_noise_scale, training=True)

    fake_hidden_for_disc = [h.unsqueeze(1) for h in fake_hidden_pooled]
    real_hidden_for_disc = [h.unsqueeze(1) for h in real_hidden_pooled]

    discriminator.eval()

    # 判别fake（用于generator loss）
    for p in discriminator.parameters():
        p.requires_grad = False
    fake_pred_for_gen = discriminator(fake_hidden_for_disc)  # (batch_size, 1)
    for p in discriminator.parameters():
        p.requires_grad = True

    # 判别real
    real_pred = discriminator(real_hidden_for_disc)  # (batch_size, 1)

    # ========== Phase 5: Loss Computation ==========
    if enable_token_merger:
        adv_loss = F.binary_cross_entropy(fake_pred_for_gen, torch.ones_like(fake_pred_for_gen), reduction='mean')
        token_merger_losses["adv_loss"] = adv_loss

        # Task loss（batch化版本）
        task_loss_sum = torch.tensor(0.0, device=device)
        for b in range(batch_size):
            task_loss_b = compute_task_loss(
                result_fake['logits'][b:b+1],
                (answer_pos[b, 0].item(), answer_pos[b, 1].item()),
                batch[b]["answer"],
                backbone.processor
            )
            task_loss_sum = task_loss_sum + task_loss_b
        task_loss = task_loss_sum / batch_size
        token_merger_losses["task_loss"] = task_loss

    # Layer Pruners Loss
    if not enable_token_merger:
        task_loss_sum = torch.tensor(0.0, device=device)
        for b in range(batch_size):
            task_loss_b = compute_task_loss(
                result_fake['logits'][b:b+1],
                (answer_pos[b, 0].item(), answer_pos[b, 1].item()),
                batch[b]["answer"],
                backbone.processor
            )
            task_loss_sum = task_loss_sum + task_loss_b
        task_loss = task_loss_sum / batch_size

    adv_loss = F.binary_cross_entropy(fake_pred_for_gen, torch.ones_like(fake_pred_for_gen), reduction='mean')
    layer_pruners_losses["adv_loss"] = adv_loss
    layer_pruners_losses["task_loss"] = task_loss

    # Sparsity Loss
    if len(pruning_masks) > 0:
        target_sparsity = config['method_settings'].get('target_sparsity')
        use_token_num_target = config['method_settings'].get('use_token_num_target')
        sparsity_loss_only_on_excess = config['method_settings'].get('sparsity_loss_only_on_excess')

        n_vision = pruning_masks[0].shape[1]

        if use_token_num_target:
            target_token_num = config['method_settings'].get('target_token_num', 128)
            target_kept_ratio = target_token_num / n_vision
        else:
            target_kept_ratio = 1.0 - target_sparsity

        kept_ratios = [mask.mean().to(device) for mask in pruning_masks]
        avg_kept_ratio = torch.stack(kept_ratios).mean()
        final_mask = pruning_masks[-1]
        final_kept_ratio = final_mask.mean()

        if sparsity_loss_only_on_excess:
            excess = torch.relu(avg_kept_ratio - target_kept_ratio)
            sparsity_constraint_loss = excess.to(device).pow(2)
        else:
            sparsity_constraint_loss = (avg_kept_ratio - target_kept_ratio).to(device).pow(2)

        layer_pruners_losses["sparsity_loss"] = sparsity_constraint_loss
        layer_pruners_losses["token_count_loss"] = avg_kept_ratio.to(device)

        # Bimodal loss
        binarization_loss = torch.tensor(0.0, device=device)
        for mask in pruning_masks:
            binary_term = (mask * (1 - mask)).mean()
            variance_term = mask.var()
            binarization_loss = binarization_loss + (binary_term - 0.5 * variance_term).to(device)
        binarization_loss = binarization_loss / len(pruning_masks)
        layer_pruners_losses["binarization_loss"] = binarization_loss

        # Stats
        pruning_layers = layer_pruners.get_all_layers()
        for idx, mask in enumerate(pruning_masks):
            layer_num = pruning_layers[idx]
            stats[f"L{layer_num}_kept"] = mask.mean().item()
        stats["avg_kept_ratio"] = avg_kept_ratio.item()
        stats["final_kept_ratio"] = final_kept_ratio.item()

        if use_attn_residual and config["method_settings"].get("learnable_attn_weight", False):
            for idx in pruning_layers:
                pruner = layer_pruners.get_pruner(idx)
                if hasattr(pruner, 'attn_residual_weight'):
                    weight_val = pruner.attn_residual_weight.item()
                    stats[f"L{idx}_attn_weight"] = weight_val

    # Discriminator Loss
    discriminator.train()

    disc_losses["real_loss"] = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred), reduction='mean')

    fake_hidden_detached = [h.detach() for h in fake_hidden_list]
    fake_pred_for_disc = discriminator(fake_hidden_detached)
    disc_losses["fake_loss"] = F.binary_cross_entropy(fake_pred_for_disc, torch.zeros_like(fake_pred_for_disc), reduction='mean')

    # Stats
    real_correct = (real_pred > 0.5).float().mean()
    fake_correct = (fake_pred_for_disc < 0.5).float().mean()
    stats["disc_real_acc"] = real_correct.item()
    stats["disc_fake_acc"] = fake_correct.item()

    # Cleanup
    del embeddings_merged, result_fake, result_real
    del fake_hidden_list, real_hidden_list, fake_hidden_detached
    del fake_pred_for_gen, real_pred, fake_pred_for_disc, pruning_masks

    for sample in batch:
        if 'image' in sample and hasattr(sample['image'], 'close'):
            sample['image'].close()

    torch.cuda.empty_cache()

    # ========== Phase 6: Apply Loss Weights ==========
    progress = current_step / total_steps
    task_weight_start = config['method_settings'].get('task_loss_weight_start', None)
    task_weight_end = config['method_settings'].get('task_loss_weight')
    adv_weight_start = config['method_settings'].get('adv_loss_weight_start', None)
    adv_weight_end = config['method_settings'].get('adv_loss_weight')
    warmup_ratio = config['method_settings'].get('loss_weight_warmup_ratio', 0.0)

    if warmup_ratio > 0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2
        task_weight = task_weight_start + (task_weight_end - task_weight_start) * cosine_factor if task_weight_start is not None else task_weight_end
        adv_weight = adv_weight_start + (adv_weight_end - adv_weight_start) * cosine_factor if adv_weight_start is not None else adv_weight_end
    else:
        task_weight = task_weight_end
        adv_weight = adv_weight_end

    sparsity_weight = get_current_sparsity_weight(config, current_step, total_steps)
    token_count_weight = config['method_settings'].get('token_count_loss_weight')
    binarization_weight = config['method_settings'].get('binarization_loss_weight', 0.0)

    stats["current_task_weight"] = float(task_weight)
    stats["current_adv_weight"] = float(adv_weight)
    stats["current_sparsity_weight"] = float(sparsity_weight)

    if enable_token_merger:
        token_merger_losses["adv_loss"] = token_merger_losses["adv_loss"] * adv_weight
        token_merger_losses["task_loss"] = token_merger_losses["task_loss"] * task_weight

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
    for losses_dict in [token_merger_losses, layer_pruners_losses, disc_losses]:
        for k in losses_dict:
            if isinstance(losses_dict[k], torch.Tensor):
                losses_dict[k] = losses_dict[k].to(target_device)

    return {
        "discriminator": dict(disc_losses),
        "token_merger": dict(token_merger_losses),
        "layer_pruners": dict(layer_pruners_losses),
        "metrics": stats
    }
