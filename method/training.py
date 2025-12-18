"""Vision Token Pruning - 两阶段剪枝训练函数

实现Token Merge + Layer-wise Pruning的两阶段剪枝训练。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List

from collections import defaultdict
from .utils import (
    extract_target_hidden_states,
    compute_task_loss,
    register_multi_layer_hooks,
    remove_hooks,
    replace_vision_tokens_in_embeddings,
    update_temperature_for_all
)


def train_step(batch: List[Any], device: torch.device, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """两阶段剪枝的训练step函数

    架构：
    1. Token Merge（LLM输入前）：576 tokens → ~288 tokens
    2. Layer-wise Pruning（LLM内部）：在Layer 10/20/31分别剪枝
    3. GAN对抗训练：Discriminator判别real/fake

    Generator = Token Merger + Layer Pruners（统一优化）

    参数:
        batch: 数据batch
        device: 设备
        info: 包含config, models等的字典
            - models:
                - "backbone": LLaVA backbone
                - "token_merger": LearnableTokenMerger实例
                - "layer_pruners": LayerSpecificPruner实例
                - "discriminator": Discriminator实例

    返回:
        损失字典，包含两个optimizer组：generator, discriminator
    """
    config = info["config"]
    backbone = info["models"]["backbone"]
    token_merger = info["models"].get("token_merger", None)  # 可能为None
    layer_pruners = info["models"]["layer_pruners"]
    discriminator = info["models"]["discriminator"]
    current_step = info["global_batch_index"]

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
    # 只在token_merger存在且启用时更新其temperature
    if token_merger is not None and enable_token_merger:
        update_temperature_for_all(token_merger, layer_pruners, config, current_step, total_steps)
    else:
        # 只更新layer_pruners的temperature
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
    # 拆分为3个独立损失组（支持不同学习率）
    token_merger_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    layer_pruners_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    disc_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    stats = defaultdict(float)

    valid_samples = 0

    # === 遍历Batch ===
    for sample in batch:
        valid_samples += 1

        # ========== Phase 1: Token Merge（LLM输入前） ==========

        # 1.1 获取原始embeddings和raw vision features
        with torch.no_grad():
            emb_info = backbone.preprocess(sample["image"], sample["question"], sample["answer"])
            original_embeddings = emb_info['embeddings']  # (1, seq_len, 4096)
            original_vision_pos = emb_info['vision_token_positions']  # (start, end)
            answer_pos = emb_info['answer_token_positions']

            # 获取未投影的vision features (1024维，CLIP输出)
            vision_features_raw = emb_info['raw_vision_features']  # (1, 576, 1024)

            if vision_features_raw is None:
                raise ValueError("backbone未返回raw_vision_features，请检查backbone实现")

            # 提取question embeddings（用于question-aware merger和pruner）
            # 结构: [0:v_start] = "USER: "
            #       [v_start:v_end+1] = vision tokens
            #       [v_end+1:answer_start] = "\n{question}\nASSISTANT: "
            #       [answer_start:] = "{answer}"
            # 我们只需要提取真正的question部分: [v_end+1:answer_start]
            v_start, v_end = original_vision_pos
            if answer_pos[0] < 0:
                # 负索引转正索引
                answer_start_abs = original_embeddings.shape[1] + answer_pos[0]
            else:
                answer_start_abs = answer_pos[0]

            # 提取question部分（只包含"\n{question}\nASSISTANT: "）
            question_embeddings_for_merger = original_embeddings[:, v_end+1:answer_start_abs, :]

        # 1.2 Token Merge（可训练，在1024维空间操作）
        if enable_token_merger and token_merger is not None:
            token_merger.train()
            # V2和V3都需要question_embeddings，V1不需要
            if config.method_settings.merger_type in ["question_aware", "fixed_pooling"]:
                merge_result = token_merger(vision_features_raw, question_embeddings_for_merger, use_gumbel=True)
            else:
                merge_result = token_merger(vision_features_raw, use_gumbel=True)
            merged_vision = merge_result['merged_features']  # (1, M, 1024)
            # Note: merge_result['importance_logits'] 可用于额外的 merge sparsity loss，但当前未启用

            # 1.3 投影到LLM维度 (1024 → 4096)
            # Token merge输出是1024维，需要投影到4096维
            merged_vision = backbone.model.multi_modal_projector(merged_vision)  # (1, M, 4096)

            # 1.4 替换vision部分
            embeddings_merged, new_vision_pos, new_attention_mask = replace_vision_tokens_in_embeddings(
                original_embeddings,
                original_vision_pos,
                merged_vision,
                emb_info['attention_mask']
            )
        else:
            # 禁用token merger，直接使用原始vision features
            # 需要先投影到LLM维度
            vision_features_projected = backbone.model.multi_modal_projector(vision_features_raw)  # (1, 576, 4096)
            embeddings_merged, new_vision_pos, new_attention_mask = replace_vision_tokens_in_embeddings(
                original_embeddings,
                original_vision_pos,
                vision_features_projected,
                emb_info['attention_mask']
            )

        # 1.5 提取question embeddings（用于layer pruners的cross-attention）
        # 结构: [0:new_vision_pos[0]] = "USER: "
        #       [new_vision_pos[0]:new_vision_pos[1]+1] = merged vision tokens
        #       [new_vision_pos[1]+1:answer_start] = "\n{question}\nASSISTANT: "
        #       [answer_start:] = "{answer}"
        # 注意: answer_pos是相对于original_embeddings的，需要调整到embeddings_merged
        num_removed_tokens = (original_vision_pos[1] - original_vision_pos[0] + 1) - (new_vision_pos[1] - new_vision_pos[0] + 1)
        answer_start_merged = answer_start_abs - num_removed_tokens

        # 只提取question部分（不包含"USER: "和answer）
        question_embeddings = embeddings_merged[:, new_vision_pos[1]+1:answer_start_merged, :]

        # ========== Phase 2: Layer-wise Pruning Forward（带hooks） ==========

        # 2.1 创建mask收集器（用于sparsity loss）
        pruning_masks = []

        # 2.2 注册hooks到Layer 10/20/31
        handles = register_multi_layer_hooks(
            backbone,
            layer_pruners,
            new_vision_pos,
            question_embeddings,
            mask_collector=pruning_masks  # 收集每层的soft_mask
        )

        try:
            # 2.2 Forward（fake sample - 带剪枝）
            layer_pruners.train()
            result_fake = backbone.forward(
                embeddings=embeddings_merged,
                attention_mask=new_attention_mask,
                output_hidden_states=True
            )

            # 2.3 提取hidden states from target layers
            fake_hidden_list = extract_target_hidden_states(
                result_fake['all_hidden_states'],
                answer_pos,
                disc_target_layers,
                batch_size=1
            )

        finally:
            # 清理hooks
            remove_hooks(handles)

        # 2.4 Forward（real sample - 无剪枝）
        with torch.no_grad():
            result_real = backbone.forward(
                embeddings=original_embeddings,
                attention_mask=emb_info['attention_mask'],
                output_hidden_states=True
            )

            real_hidden_list = extract_target_hidden_states(
                result_real['all_hidden_states'],
                answer_pos,
                disc_target_layers,
                batch_size=1
            )

        # ========== Phase 3: Discriminator Judgment ==========

        discriminator.eval()

        # 3.1 判别fake（用于generator loss）
        for p in discriminator.parameters():
            p.requires_grad = False

        fake_pred_for_gen = discriminator(fake_hidden_list)  # (batch, seq_len)

        for p in discriminator.parameters():
            p.requires_grad = True

        # 3.2 判别real
        real_pred = discriminator(real_hidden_list)

        # ========== Phase 4: Loss Computation ==========

        # --- Token Merger Loss ---
        # Merger只需要优化adv_loss和task_loss（保证合并后的tokens质量）
        # 只有在启用token merger时才计算相关loss
        if enable_token_merger:
            # 1. Adversarial Loss: 骗过discriminator
            adv_loss = F.binary_cross_entropy(
                fake_pred_for_gen,
                torch.ones_like(fake_pred_for_gen),
                reduction='mean'
            )
            token_merger_losses["adv_loss"] = token_merger_losses["adv_loss"] + adv_loss

            # 2. Task Loss: 保持任务性能
            task_loss = compute_task_loss(
                result_fake['logits'],
                answer_pos,
                sample["answer"],
                backbone.processor
            )
            token_merger_losses["task_loss"] = token_merger_losses["task_loss"] + task_loss

        # --- Layer Pruners Loss ---
        # Pruners需要优化adv_loss、task_loss、以及sparsity相关的loss

        # 如果未启用token merger，layer pruners也需要计算task loss
        if not enable_token_merger:
            task_loss = compute_task_loss(
                result_fake['logits'],
                answer_pos,
                sample["answer"],
                backbone.processor
            )

        # Adversarial loss (总是需要)
        adv_loss = F.binary_cross_entropy(
            fake_pred_for_gen,
            torch.ones_like(fake_pred_for_gen),
            reduction='mean'
        )
        layer_pruners_losses["adv_loss"] = layer_pruners_losses["adv_loss"] + adv_loss
        layer_pruners_losses["task_loss"] = layer_pruners_losses["task_loss"] + task_loss

        # 3. Sparsity Loss: 只在最后一层约束token保留率
        # pruning_masks: list of (batch, n_vision) 每层的soft_mask
        # 由于mask是累积乘法的，最后一层的约束会通过梯度传播到前面的层
        if len(pruning_masks) > 0:
            # 获取配置
            target_sparsity = config['method_settings'].get('target_sparsity')
            use_token_num_target = config['method_settings'].get('use_token_num_target')
            sparsity_loss_only_on_excess = config['method_settings'].get('sparsity_loss_only_on_excess')

            # 获取原始vision token数（第一层的输入）
            n_vision = pruning_masks[0].shape[1]

            # 计算目标保留率
            if use_token_num_target:
                # 基于绝对token数
                target_token_num = config['method_settings'].get('target_token_num', 128)
                target_kept_ratio = target_token_num / n_vision
            else:
                # 基于稀疏度比例
                target_kept_ratio = 1.0 - target_sparsity

            # === Sparsity Loss: 只在最后一层约束 ===
            # 最后一层的mask反映了累积剪枝后的效果
            # 梯度会通过hidden states传播到前面的层
            final_mask = pruning_masks[-1]  # 最后一层的soft_mask
            final_kept_ratio = final_mask.mean()  # 最后一层的保留比例

            if sparsity_loss_only_on_excess:
                # 只在保留率超过目标时惩罚
                excess = torch.relu(final_kept_ratio - target_kept_ratio)
                sparsity_constraint_loss = excess.to(device).pow(2)
            else:
                # 双向惩罚（过多或过少都惩罚）
                sparsity_constraint_loss = (final_kept_ratio - target_kept_ratio).to(device).pow(2)

            layer_pruners_losses["sparsity_loss"] = layer_pruners_losses["sparsity_loss"] + sparsity_constraint_loss

            # === Token Count Loss: 只在最后一层计算 ===
            # 最后一层保留的token总数
            token_count_loss = final_mask.sum().to(device) / n_vision
            layer_pruners_losses["token_count_loss"] = layer_pruners_losses["token_count_loss"] + token_count_loss

            # 统计信息：记录每层的保留率
            for idx, mask in enumerate(pruning_masks):
                stats[f"layer_{idx}_kept_ratio"] += mask.mean().item()
            stats["final_kept_ratio"] += final_kept_ratio.item()
            stats["final_token_count"] += final_mask.sum().item()
            stats["target_kept_ratio"] += target_kept_ratio

        # --- Discriminator Loss ---

        discriminator.train()

        # Real loss
        disc_losses["real_loss"] = disc_losses["real_loss"] + F.binary_cross_entropy(
            real_pred,
            torch.ones_like(real_pred),
            reduction='mean'
        )

        # Fake loss（detach hidden states）
        fake_hidden_detached = [h.detach() for h in fake_hidden_list]
        fake_pred_for_disc = discriminator(fake_hidden_detached)
        disc_losses["fake_loss"] = disc_losses["fake_loss"] + F.binary_cross_entropy(
            fake_pred_for_disc,
            torch.zeros_like(fake_pred_for_disc),
            reduction='mean'
        )

        # 清理
        del embeddings_merged, result_fake, result_real
        del fake_hidden_list, real_hidden_list, fake_hidden_detached
        del fake_pred_for_gen, real_pred, fake_pred_for_disc, pruning_masks

    # ========== Phase 5: 归一化Loss并应用权重 ==========

    if valid_samples > 0:
        # Normalize (非 inplace 操作)
        for k in token_merger_losses:
            token_merger_losses[k] = token_merger_losses[k] / valid_samples
        for k in layer_pruners_losses:
            layer_pruners_losses[k] = layer_pruners_losses[k] / valid_samples
        for k in disc_losses:
            disc_losses[k] = disc_losses[k] / valid_samples
        for k in stats:
            stats[k] = stats[k] / valid_samples

        # Apply weights (非 inplace 操作)
        adv_weight = config['method_settings'].get('adv_loss_weight')
        task_weight = config['method_settings'].get('task_loss_weight')
        sparsity_weight = config['method_settings'].get('sparsity_weight')
        token_count_weight = config['method_settings'].get('token_count_loss_weight')

        # Token Merger权重（只有在启用时才应用）
        if enable_token_merger:
            token_merger_losses["adv_loss"] = token_merger_losses["adv_loss"] * adv_weight
            token_merger_losses["task_loss"] = token_merger_losses["task_loss"] * task_weight

        # Layer Pruners权重
        layer_pruners_losses["adv_loss"] = layer_pruners_losses["adv_loss"] * adv_weight
        layer_pruners_losses["task_loss"] = layer_pruners_losses["task_loss"] * task_weight
        if "sparsity_loss" in layer_pruners_losses:
            layer_pruners_losses["sparsity_loss"] = layer_pruners_losses["sparsity_loss"] * sparsity_weight
        if "token_count_loss" in layer_pruners_losses:
            layer_pruners_losses["token_count_loss"] = layer_pruners_losses["token_count_loss"] * token_count_weight

    # 确保tensor在正确设备上
    # 使用layer_pruners而不是token_merger来获取设备（因为token_merger可能为None）
    target_device = next(layer_pruners.parameters()).device
    for losses_dict in [token_merger_losses, layer_pruners_losses, disc_losses]:
        for k in losses_dict:
            if isinstance(losses_dict[k], torch.Tensor):
                losses_dict[k] = losses_dict[k].to(target_device)

    torch.cuda.empty_cache()

    # 返回3个独立的优化器组，按顺序: discriminator → token_merger → layer_pruners
    # 这样discriminator先释放计算图，减少显存峰值
    return {
        "discriminator": dict(disc_losses),      # 第1个：独立计算图，先backward先释放
        "token_merger": dict(token_merger_losses),  # 第2个：共享计算图
        "layer_pruners": dict(layer_pruners_losses) # 第3个：共享计算图，最后释放
    }
