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
    # 每个样本的question长度可能不同，需要分别提取并padding
    hidden_dim = embeddings_for_forward.shape[2]

    # 计算每个样本的实际长度（用于转换负索引）
    actual_lengths = attention_mask.sum(dim=1).tolist()  # List[int]

    # 计算每个样本的question范围
    question_ranges = []
    seq_len = embeddings_for_forward.shape[1]
    for i, (ans_start, _) in enumerate(answer_positions):
        # 转换负索引
        sample_len = int(actual_lengths[i])
        if ans_start < 0:
            ans_start = sample_len + ans_start
        q_start = v_end + 1
        q_end = ans_start - 1  # question结束于answer开始前一个位置

        # 边界检查
        if q_end < q_start:
            print(f"[WARNING] 样本 {i}: question范围无效 (q_start={q_start}, q_end={q_end}, "
                  f"ans_start={ans_start}, v_end={v_end}, sample_len={sample_len}, seq_len={seq_len})")
            print(f"  原始 answer_positions[{i}] = {answer_positions[i]}")
            print(f"  questions[{i}] = {questions[i][:100]}...")
            # 使用最小有效范围（至少1个token）
            q_end = q_start
        if q_start >= seq_len:
            print(f"[WARNING] 样本 {i}: q_start={q_start} >= seq_len={seq_len}")
            q_start = seq_len - 1
            q_end = seq_len - 1

        question_ranges.append((q_start, q_end))

    # 找到最大question长度
    max_question_len = max(q_end - q_start + 1 for q_start, q_end in question_ranges)

    # 为每个样本提取question并padding
    question_embeddings = torch.zeros(batch_size, max_question_len, hidden_dim,
                                      device=embeddings_for_forward.device,
                                      dtype=embeddings_for_forward.dtype)
    question_mask = torch.zeros(batch_size, max_question_len,
                                device=embeddings_for_forward.device,
                                dtype=torch.bool)

    for i, (q_start, q_end) in enumerate(question_ranges):
        q_len = q_end - q_start + 1
        question_embeddings[i, :q_len, :] = embeddings_for_forward[i, q_start:q_end+1, :]
        question_mask[i, :q_len] = True

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
        use_attn_residual=use_attn_residual,
        question_mask=question_mask
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

        # 立即释放 all_hidden_states（32层的hidden states占用大量显存）
        del result_fake['all_hidden_states']

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

        # 立即释放 all_hidden_states（32层的hidden states占用大量显存）
        # del result_real['all_hidden_states']
        # del result_real

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

    # 1. Task Loss: 保持任务性能（使用fake sample的logits，梯度传导到pruner）
    task_loss = compute_task_loss_batch(
        result_fake['logits'],
        answer_positions,
        answers,
        backbone.processor,
        attention_mask=new_attention_mask
    )
    layer_pruners_losses["task_loss"] = task_loss
    del task_loss  # 已存入字典，删除局部引用

    # 2. Adversarial loss (使用 logits，数值更稳定)
    adv_loss = F.binary_cross_entropy_with_logits(
        fake_pred_for_gen,
        torch.ones_like(fake_pred_for_gen),
        reduction='mean'
    )

    # === DEBUG: 检查 adv_loss 梯度传导 ===
    debug_adv_grad = config.get("debug_adv_grad", False)
    if debug_adv_grad:
        print(f"\n[DEBUG adv_loss] === 梯度传导检查 ===")
        print(f"[DEBUG adv_loss] adv_loss.requires_grad: {adv_loss.requires_grad}")
        print(f"[DEBUG adv_loss] adv_loss.grad_fn: {adv_loss.grad_fn}")
        print(f"[DEBUG adv_loss] fake_pred_for_gen.requires_grad: {fake_pred_for_gen.requires_grad}")
        print(f"[DEBUG adv_loss] fake_pred_for_gen.grad_fn: {fake_pred_for_gen.grad_fn}")

        # 检查 fake_hidden_list 的梯度
        for i, h in enumerate(fake_hidden_list):
            print(f"[DEBUG adv_loss] fake_hidden_list[{i}].requires_grad: {h.requires_grad}, grad_fn: {h.grad_fn}")

        # 检查 pruning_masks 的梯度
        for i, mask in enumerate(pruning_masks):
            print(f"[DEBUG adv_loss] pruning_masks[{i}].requires_grad: {mask.requires_grad}, grad_fn: {mask.grad_fn}")

        # 关键检查：disc_target_layers vs pruning_layers
        print(f"\n[DEBUG adv_loss] === 层配置检查 ===")
        print(f"[DEBUG adv_loss] pruning_layers: {layer_pruners.get_all_layers()}")
        print(f"[DEBUG adv_loss] disc_target_layers: {disc_target_layers}")
        print(f"[DEBUG adv_loss] 警告: 如果 disc_target_layers 不包含 pruning_layers，")
        print(f"[DEBUG adv_loss]        adv_loss 的梯度可能无法传导到 pruner！")
        print(f"[DEBUG adv_loss]        因为 LLM 是 frozen 的，梯度无法通过 LLM 层传导。")

        # 关键测试：单独对 adv_loss 做 backward，检查 pruner 是否收到梯度
        print(f"\n[DEBUG adv_loss] === 梯度传导测试 ===")
        # 保存当前梯度
        for p in layer_pruners.parameters():
            if p.grad is not None:
                p.grad.zero_()

        # 单独 backward adv_loss
        adv_loss.backward(retain_graph=True)

        # 检查 pruner 是否收到梯度
        pruner_grads = []
        for name, p in layer_pruners.named_parameters():
            if p.grad is not None:
                grad_norm = p.grad.norm().item()
                pruner_grads.append((name, grad_norm))
                if grad_norm > 1e-10:
                    print(f"[DEBUG adv_loss] ✓ {name}: grad_norm = {grad_norm:.6f}")
            else:
                print(f"[DEBUG adv_loss] ✗ {name}: NO GRAD!")

        if not pruner_grads or all(g < 1e-10 for _, g in pruner_grads):
            print(f"\n[DEBUG adv_loss] ❌ 严重问题: adv_loss 的梯度没有传导到 layer_pruners!")
            print(f"[DEBUG adv_loss] 原因: disc_target_layers ({disc_target_layers}) 和 pruning_layers ({layer_pruners.get_all_layers()}) 不匹配")
            print(f"[DEBUG adv_loss] 解决方案: 将 disc_target_layers 设置为 pruning_layers 或其后一层")
        else:
            print(f"\n[DEBUG adv_loss] ✓ adv_loss 的梯度成功传导到 layer_pruners")

        # 清零梯度，让后续正常训练
        for p in layer_pruners.parameters():
            if p.grad is not None:
                p.grad.zero_()

    layer_pruners_losses["adv_loss"] = adv_loss
    del adv_loss  # 已存入字典，删除局部引用

    # 3. Sparsity Loss
    if len(pruning_masks) > 0:
        target_sparsity = config['method_settings'].get('target_sparsity')
        use_token_num_target = config['method_settings'].get('use_token_num_target')
        sparsity_loss_only_on_excess = config['method_settings'].get('sparsity_loss_only_on_excess')
        min_layer_keep_ratio = config['method_settings'].get('min_layer_keep_ratio', 0.02)

        n_vision = pruning_masks[0].shape[1]

        if use_token_num_target:
            target_token_num = config['method_settings'].get('target_token_num', 128)
            target_kept_ratio = target_token_num / n_vision
        else:
            target_kept_ratio = 1.0 - target_sparsity

        # 计算每层的保留率（用于统计）
        kept_ratios = [mask.mean().to(device) for mask in pruning_masks]

        # 计算累积 mask（所有层 mask 相乘）- 反映真实的最终保留率
        cumulative_mask = pruning_masks[0]
        for mask in pruning_masks[1:]:
            cumulative_mask = cumulative_mask * mask
        final_kept_ratio = cumulative_mask.mean()

        # === 新增：Per-layer minimum constraint ===
        # 防止任何单层剪枝过度导致mode collapse
        per_layer_penalty = torch.tensor(0.0, device=device)
        for ratio in kept_ratios:
            # 当保留率低于最小值时，施加强惩罚
            violation = torch.relu(min_layer_keep_ratio - ratio)
            per_layer_penalty = per_layer_penalty + violation.pow(2) * 100.0  # 强惩罚

        # 计算 LLM 所有层的加权平均保留率（考虑 KV cache）
        # 假设 LLaMA-7B 有 32 层，剪枝层是 [5, 15, 25]
        # Layer 0-4: 保留率=1.0, Layer 5-14: 保留率=L5, Layer 15-24: 保留率=L5*L15, Layer 25-31: 保留率=L5*L15*L25
        pruning_layers = layer_pruners.get_all_layers()  # e.g., [5, 15, 25]
        total_llm_layers = 32  # LLaMA-7B

        # 计算每个区间的累积保留率
        cumulative_ratios = []
        cum_ratio = torch.tensor(1.0, device=device)
        for mask in pruning_masks:
            cum_ratio = cum_ratio * mask.mean().to(device)
            cumulative_ratios.append(cum_ratio)

        # 计算加权平均：每个区间的层数 × 该区间的保留率
        weighted_sum = torch.tensor(0.0, device=device)
        prev_layer = 0
        for i, layer_idx in enumerate(pruning_layers):
            # 剪枝层之前的区间：保留率是上一个累积值（或1.0）
            n_layers_before = layer_idx - prev_layer
            ratio_before = cumulative_ratios[i-1] if i > 0 else torch.tensor(1.0, device=device)
            weighted_sum = weighted_sum + n_layers_before * ratio_before
            prev_layer = layer_idx

        # 最后一个剪枝层之后的区间
        n_layers_after = total_llm_layers - prev_layer
        weighted_sum = weighted_sum + n_layers_after * cumulative_ratios[-1]

        avg_kept_ratio = weighted_sum / total_llm_layers

        if sparsity_loss_only_on_excess:
            excess = torch.relu(avg_kept_ratio - target_kept_ratio)
            sparsity_constraint_loss = excess.to(device).pow(2)
        else:
            sparsity_constraint_loss = (avg_kept_ratio - target_kept_ratio).to(device).pow(2)

        # 加入per-layer penalty防止mode collapse
        sparsity_constraint_loss = sparsity_constraint_loss + per_layer_penalty

        layer_pruners_losses["sparsity_loss"] = sparsity_constraint_loss

        # Token Count Loss
        token_count_loss = avg_kept_ratio.to(device)
        layer_pruners_losses["token_count_loss"] = token_count_loss

        # Binarization Loss: 禁用
        binarization_loss = torch.tensor(0.0, device=device)
        layer_pruners_losses["binarization_loss"] = binarization_loss

        # 统计信息
        for idx, mask in enumerate(pruning_masks):
            layer_num = pruning_layers[idx]
            stats[f"L{layer_num}_kept"] = mask.mean().item()
        stats["avg_kept_ratio"] = avg_kept_ratio.item()
        stats["final_kept_ratio"] = final_kept_ratio.item()
        stats["final_token_count"] = cumulative_mask.sum().item() / batch_size
        stats["target_kept_ratio"] = target_kept_ratio

        # 立即清理sparsity计算的中间变量
        del kept_ratios, avg_kept_ratio, cumulative_mask, final_kept_ratio, cumulative_ratios
        del sparsity_constraint_loss, token_count_loss, binarization_loss, per_layer_penalty
        if sparsity_loss_only_on_excess:
            del excess

    # --- Discriminator Loss ---

    discriminator.train()

    # Real loss (使用 logits)
    disc_losses["real_loss"] = F.binary_cross_entropy_with_logits(
        real_pred,
        torch.ones_like(real_pred),
        reduction='mean'
    )

    # Fake loss (使用 logits)
    fake_hidden_detached = [h.detach() for h in fake_hidden_list]
    fake_pred_for_disc = discriminator(fake_hidden_detached)
    disc_losses["fake_loss"] = F.binary_cross_entropy_with_logits(
        fake_pred_for_disc,
        torch.zeros_like(fake_pred_for_disc),
        reduction='mean'
    )

    # 判别器准确率 (对 logits 应用 sigmoid 后判断)
    real_prob = torch.sigmoid(real_pred)
    fake_prob = torch.sigmoid(fake_pred_for_disc)
    real_correct = (real_prob > 0.5).float().mean()
    fake_correct = (fake_prob < 0.5).float().mean()
    disc_accuracy = (real_correct + fake_correct) / 2.0
    stats["disc_accuracy"] = disc_accuracy.item()
    stats["disc_real_acc"] = real_correct.item()
    stats["disc_fake_acc"] = fake_correct.item()

    # 清理中间变量（释放显存）
    del embeddings_for_forward, result_fake
    del fake_hidden_list, real_hidden_list, fake_hidden_detached
    del fake_pred_for_gen, real_pred, fake_pred_for_disc, pruning_masks
    del original_embeddings, vision_features_raw, vision_features_projected
    del question_embeddings, question_mask, emb_info
    del real_prob, fake_prob, real_correct, fake_correct, disc_accuracy

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

    # 保存原始loss（未加权）
    stats["raw_task_loss"] = layer_pruners_losses["task_loss"].item()
    stats["raw_adv_loss"] = layer_pruners_losses["adv_loss"].item()
    if "sparsity_loss" in layer_pruners_losses:
        stats["raw_sparsity_loss"] = layer_pruners_losses["sparsity_loss"].item()

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

    return {
        "discriminator": dict(disc_losses),
        "layer_pruners": dict(layer_pruners_losses),
        "metrics": stats
    }
