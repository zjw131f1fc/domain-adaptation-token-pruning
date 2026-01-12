"""Vision Token Pruning - 两阶段剪枝训练函数

实现Token Merge + Layer-wise Pruning的两阶段剪枝训练。
支持批量处理（通过将图像resize到统一大小）。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

from collections import defaultdict
from .utils import (
    extract_target_hidden_states,
    extract_target_hidden_states_batch,
    compute_task_loss,
    compute_task_loss_batch,
    register_multi_layer_hooks,
    register_multi_layer_hooks_batch,
    remove_hooks,
    replace_vision_tokens_in_embeddings,
    update_temperature_for_all,
    get_current_sparsity_weight
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
        use_attn_residual = config["method_settings"].get("use_attn_residual", False)
        handles = register_multi_layer_hooks(
            backbone,
            layer_pruners,
            new_vision_pos,
            question_embeddings,
            mask_collector=pruning_masks,  # 收集每层的soft_mask
            use_attn_residual=use_attn_residual  # 是否启用attention residual
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

            # === Sparsity Loss: 基于所有层的平均保留率 ===
            # 计算每层的保留率，然后取平均
            # 这样训练目标与评估指标（hard_avg_keep_ratio）一致
            kept_ratios = [mask.mean().to(device) for mask in pruning_masks]  # 每层的保留率，确保在同一设备
            avg_kept_ratio = torch.stack(kept_ratios).mean()  # 所有层的平均保留率
            final_mask = pruning_masks[-1]  # 保留最后一层用于统计
            final_kept_ratio = final_mask.mean()

            if sparsity_loss_only_on_excess:
                # 只在保留率超过目标时惩罚
                excess = torch.relu(avg_kept_ratio - target_kept_ratio)
                sparsity_constraint_loss = excess.to(device).pow(2)
            else:
                # 双向惩罚（过多或过少都惩罚）
                sparsity_constraint_loss = (avg_kept_ratio - target_kept_ratio).to(device).pow(2)

            layer_pruners_losses["sparsity_loss"] = layer_pruners_losses["sparsity_loss"] + sparsity_constraint_loss

            # === Token Count Loss: 基于所有层的平均保留率 ===
            # avg_kept_ratio 已经是归一化的比例（0-1范围）
            token_count_loss = avg_kept_ratio.to(device)
            layer_pruners_losses["token_count_loss"] = layer_pruners_losses["token_count_loss"] + token_count_loss

            # === Bimodal Loss: 鼓励 soft_mask 接近 0 或 1，但不能全是同一个值 ===
            # 两部分组成:
            # 1. binarization: mask * (1 - mask) 在 0.5 时最大，鼓励输出接近 0 或 1
            # 2. variance: 鼓励 token 之间有差异，避免全部输出相同值（全剪或全留）
            binarization_loss = torch.tensor(0.0, device=device)
            for mask in pruning_masks:
                # 鼓励接近 0 或 1
                binary_term = (mask * (1 - mask)).mean()
                # 鼓励 token 之间有差异（variance 越大越好，所以取负）
                variance_term = mask.var()
                # 组合：最小化 binary_term，最大化 variance（所以减去）
                binarization_loss = binarization_loss + (binary_term - 0.5 * variance_term).to(device)
            binarization_loss = binarization_loss / len(pruning_masks)
            layer_pruners_losses["binarization_loss"] = layer_pruners_losses["binarization_loss"] + binarization_loss

            # 统计信息：记录每层的保留率
            pruning_layers = layer_pruners.get_all_layers()
            for idx, mask in enumerate(pruning_masks):
                layer_num = pruning_layers[idx]
                stats[f"L{layer_num}_kept"] += mask.mean().item()
            stats["avg_kept_ratio"] += avg_kept_ratio.item()  # 新增：所有层平均保留率
            stats["final_kept_ratio"] += final_kept_ratio.item()  # 保留：最后一层保留率
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

        # 计算判别器胜率（正确分类的比例）
        # real_pred 应该接近1，fake_pred_for_disc 应该接近0
        real_correct = (real_pred > 0.5).float().mean()
        fake_correct = (fake_pred_for_disc < 0.5).float().mean()
        disc_accuracy = (real_correct + fake_correct) / 2.0
        stats["disc_accuracy"] = stats.get("disc_accuracy", 0.0) + disc_accuracy.item()
        stats["disc_real_acc"] = stats.get("disc_real_acc", 0.0) + real_correct.item()
        stats["disc_fake_acc"] = stats.get("disc_fake_acc", 0.0) + fake_correct.item()

        # 清理张量和图像对象
        del embeddings_merged, result_fake, result_real
        del fake_hidden_list, real_hidden_list, fake_hidden_detached
        del fake_pred_for_gen, real_pred, fake_pred_for_disc, pruning_masks

        # 显式释放图像对象（PIL Image），避免内存泄露
        if 'image' in sample and hasattr(sample['image'], 'close'):
            sample['image'].close()
        del sample

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

        # === Dynamic Loss Weight Scheduling (余弦调度) ===
        # 训练初期: task_weight高，adv_weight低（优先学习保留信息）
        # 训练后期: task_weight降低，adv_weight升高（强化对抗训练）

        # 1. 计算训练进度
        progress = current_step / total_steps  # 0.0 → 1.0

        # 2. 读取配置
        task_weight_start = config['method_settings'].get('task_loss_weight_start', None)
        task_weight_end = config['method_settings'].get('task_loss_weight')
        adv_weight_start = config['method_settings'].get('adv_loss_weight_start', None)
        adv_weight_end = config['method_settings'].get('adv_loss_weight')
        warmup_ratio = config['method_settings'].get('loss_weight_warmup_ratio', 0.0)

        # 3. 余弦调度计算
        if warmup_ratio > 0 and progress < warmup_ratio:
            # Warmup阶段：平滑过渡
            warmup_progress = progress / warmup_ratio  # 0.0 → 1.0
            # 余弦插值: cos从1→0，映射为0→1的平滑曲线
            cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2

            # Task weight: start → end (递减)
            if task_weight_start is not None:
                task_weight = task_weight_start + (task_weight_end - task_weight_start) * cosine_factor
            else:
                task_weight = task_weight_end  # 未配置start，直接使用end

            # Adv weight: start → end (递增)
            if adv_weight_start is not None:
                adv_weight = adv_weight_start + (adv_weight_end - adv_weight_start) * cosine_factor
            else:
                adv_weight = adv_weight_end
        else:
            # Warmup后：使用目标权重
            task_weight = task_weight_end
            adv_weight = adv_weight_end

        # 4. Sparsity weight (使用warmup机制，前期弱约束→后期强约束，防止token数反弹)
        sparsity_weight = get_current_sparsity_weight(config, current_step, total_steps)
        token_count_weight = config['method_settings'].get('token_count_loss_weight')
        binarization_weight = config['method_settings'].get('binarization_loss_weight', 0.0)

        # 5. 记录当前权重（用于日志）
        stats["current_task_weight"] = float(task_weight)
        stats["current_adv_weight"] = float(adv_weight)
        stats["current_sparsity_weight"] = float(sparsity_weight)

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
        if "binarization_loss" in layer_pruners_losses:
            layer_pruners_losses["binarization_loss"] = layer_pruners_losses["binarization_loss"] * binarization_weight

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
        "layer_pruners": dict(layer_pruners_losses), # 第3个：共享计算图，最后释放
        "metrics": stats  # 训练指标：保留率、判别器胜率等
    }


def train_step_batch(batch: List[Any], device: torch.device, info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
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
        # 计算answer起始位置（使用第一个样本的位置，因为vision token数量相同）
        # 注意：由于padding，不同样本的answer位置可能不同
        # 我们需要找到每个样本的实际answer位置

        # 投影vision features到LLM维度
        vision_features_projected = backbone.model.multi_modal_projector(vision_features_raw)  # (batch, num_vision, 4096)

    # 构建embeddings（vision部分已经在preprocess_batch中处理好了）
    embeddings_for_forward = original_embeddings
    new_vision_pos = original_vision_pos
    new_attention_mask = attention_mask

    # 提取question embeddings（用于layer pruners）
    # 结构: [0:v_start] = "USER: "
    #       [v_start:v_end+1] = vision tokens
    #       [v_end+1:answer_start] = "\n{question}\nASSISTANT: "
    #       [answer_start:] = "{answer}"
    # 由于不同样本的answer位置可能不同，我们使用最短的question部分
    # 或者使用attention mask来处理

    # 简化处理：使用固定的question区域（从vision结束到序列末尾的一部分）
    # 这里我们取vision结束后的固定长度作为question embeddings
    question_embeddings = embeddings_for_forward[:, v_end+1:, :]  # (batch, remaining_len, 4096)

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

        # 2.4 提取hidden states from target layers
        fake_hidden_list = extract_target_hidden_states_batch(
            result_fake['all_hidden_states'],
            answer_positions,
            disc_target_layers,
            batch_size=batch_size,
            attention_mask=new_attention_mask
        )

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

    for p in discriminator.parameters():
        p.requires_grad = True

    # 3.2 判别real
    real_pred = discriminator(real_hidden_list)

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

    # 2. Adversarial loss
    adv_loss = F.binary_cross_entropy(
        fake_pred_for_gen,
        torch.ones_like(fake_pred_for_gen),
        reduction='mean'
    )
    layer_pruners_losses["adv_loss"] = adv_loss

    # 3. Sparsity Loss
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

        # 计算每层的保留率
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

        # Token Count Loss
        token_count_loss = avg_kept_ratio.to(device)
        layer_pruners_losses["token_count_loss"] = token_count_loss

        # Binarization Loss
        binarization_loss = torch.tensor(0.0, device=device)
        for mask in pruning_masks:
            binary_term = (mask * (1 - mask)).mean()
            variance_term = mask.var()
            binarization_loss = binarization_loss + (binary_term - 0.5 * variance_term).to(device)
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

    # Real loss
    disc_losses["real_loss"] = F.binary_cross_entropy(
        real_pred,
        torch.ones_like(real_pred),
        reduction='mean'
    )

    # Fake loss
    fake_hidden_detached = [h.detach() for h in fake_hidden_list]
    fake_pred_for_disc = discriminator(fake_hidden_detached)
    disc_losses["fake_loss"] = F.binary_cross_entropy(
        fake_pred_for_disc,
        torch.zeros_like(fake_pred_for_disc),
        reduction='mean'
    )

    # 判别器准确率
    real_correct = (real_pred > 0.5).float().mean()
    fake_correct = (fake_pred_for_disc < 0.5).float().mean()
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
