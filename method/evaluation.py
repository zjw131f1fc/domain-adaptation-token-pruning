"""Vision Token Pruning - 两阶段剪枝评估函数

实现Token Merge + Layer-wise Pruning的评估。
"""

import torch
from typing import Dict, Any, List

from .utils import (
    register_multi_layer_hooks,
    register_hard_pruning_at_model_level,
    remove_hooks,
    replace_vision_tokens_in_embeddings
)


def eval_step(batch: List[Any], device: torch.device, info: Dict[str, Any]) -> Dict[str, float]:
    """两阶段剪枝的评估step函数（批量预处理优化版本）

    评估模式：
    1. "origin" - Baseline（无剪枝）
    2. "merge_only" - 只Token Merge，不做Layer Pruning
    3. "soft" - Soft pruning（软mask，continuous 0-1）
    4. "hard" - Hard pruning（硬mask，binary 0/1）

    优化说明：
    - 预处理阶段使用backbone.preprocess_batch()批量处理
    - 生成阶段由于自回归特性仍需逐样本调用（无法真正并行）

    参数:
        batch: 数据batch
        device: 设备
        info: 包含config, models等的字典

    返回:
        评估指标字典
    """
    config = info["config"]
    backbone = info["models"]["backbone"]
    token_merger = info["models"].get("token_merger", None)  # 可能为None
    layer_pruners = info["models"]["layer_pruners"]
    dataset_bundle = config.get("_dataset_bundle")
    judge_fn = dataset_bundle["judge"] if dataset_bundle else None

    # 获取配置
    enable_token_merger = config["method_settings"].get("enable_token_merger", True)
    eval_use_gumbel = config["evaluation_settings"].get("eval_use_gumbel", False)
    enable_true_batch = config["backbone_settings"]["mllm_settings"].get("enable_true_batch", False)

    # 设置为eval模式
    if token_merger is not None:
        token_merger.eval()
    layer_pruners.eval()  # 评估使用确定性mask（禁用gumbel/dropout）

    eval_modes = config["evaluation_settings"]["eval_mode"]

    results = {
        "accuracy_baseline": 0.0,
        "accuracy_merge_only": 0.0,
        "accuracy_soft": 0.0,
        "accuracy_hard": 0.0,
        "keep_ratio_merge": 0.0,
        "avg_merged_tokens": 0.0,
        "avg_original_tokens": 0.0,
        # Hard pruning 统计
        "hard_final_tokens": 0.0,          # 最终剩余的token数
        "hard_avg_tokens": 0.0,            # 所有LLM层的平均token数（新增）
        "hard_keep_ratio_total": 0.0,      # 最终保留率
        "hard_avg_keep_ratio": 0.0         # 平均保留率（新增）
    }

    valid_samples = 0

    def get_ref_answer(sample):
        """获取用于评估的参考答案"""
        if "answers" in sample and isinstance(sample["answers"], list) and len(sample["answers"]) > 0:
            return sample["answers"]
        return sample["answer"]

    # ============ 批量预处理阶段（优化点） ============
    # 如果启用batch且需要pruning评估，批量预处理所有样本
    if enable_true_batch and ("merge_only" in eval_modes or "soft" in eval_modes or "hard" in eval_modes):
        images = [sample["image"] for sample in batch]
        questions = [sample["question"] for sample in batch]

        with torch.no_grad():
            # 批量预处理
            emb_info_batch = backbone.preprocess_batch(images, questions, None)
            original_embeddings_batch = emb_info_batch['embeddings']  # (B, seq_len, dim)
            original_vision_pos_batch = emb_info_batch['vision_token_positions']  # (B, 2)
            vision_features_raw_batch = emb_info_batch['raw_vision_features']  # (B, n_vision, 1024)
            original_attention_mask_batch = emb_info_batch['attention_mask']  # (B, seq_len)

            if vision_features_raw_batch is None:
                raise ValueError("backbone未返回raw_vision_features，请检查backbone实现")

            # Token Merge阶段（批量）
            if enable_token_merger and token_merger is not None:
                # 批量提取question embeddings
                # 获取vision结束位置（batch内统一）
                v_end = original_vision_pos_batch[0, 1].item()
                question_embeddings_for_merger_batch = original_embeddings_batch[:, v_end+1:, :]  # (B, q_len, dim)

                # 批量Token Merge
                if config.method_settings.merger_type in ["question_aware", "fixed_pooling"]:
                    merge_result = token_merger(vision_features_raw_batch, question_embeddings_for_merger_batch, use_gumbel=eval_use_gumbel)
                else:
                    merge_result = token_merger(vision_features_raw_batch, use_gumbel=eval_use_gumbel)
                merged_vision_batch = merge_result['merged_features']  # (B, M, 1024)

                # 批量投影到LLM维度
                merged_vision_batch = backbone.model.multi_modal_projector(merged_vision_batch)  # (B, M, 4096)

                # 批量替换vision部分
                embeddings_merged_batch, new_vision_pos_batch, new_attention_mask_batch = replace_vision_tokens_in_embeddings(
                    original_embeddings_batch,
                    original_vision_pos_batch,
                    merged_vision_batch,
                    original_attention_mask_batch
                )
            else:
                # 禁用token merger，使用原始vision features
                vision_features_projected_batch = backbone.model.multi_modal_projector(vision_features_raw_batch)  # (B, 576, 4096)
                embeddings_merged_batch, new_vision_pos_batch, new_attention_mask_batch = replace_vision_tokens_in_embeddings(
                    original_embeddings_batch,
                    original_vision_pos_batch,
                    vision_features_projected_batch,
                    original_attention_mask_batch
                )

            # 批量提取question embeddings（用于layer pruners）
            new_v_end = new_vision_pos_batch[0, 1].item()
            question_embeddings_batch = embeddings_merged_batch[:, new_v_end+1:, :]  # (B, q_len, dim)
    else:
        # Fallback：非batch模式或仅baseline评估
        embeddings_merged_batch = None
        new_vision_pos_batch = None
        new_attention_mask_batch = None
        question_embeddings_batch = None
        original_vision_pos_batch = None

    # ============ 逐样本生成阶段（无法避免） ============
    for idx, sample in enumerate(batch):
        ref_answer = get_ref_answer(sample)

        # ============ Baseline评估（无剪枝） ============
        if "origin" in eval_modes and judge_fn:
            with torch.no_grad():
                pred_baseline = backbone.generate(
                    sample["image"], sample["question"], max_new_tokens=20
                )
                judge_result = judge_fn(pred_baseline, ref_answer, sample)
                results["accuracy_baseline"] += judge_result["accuracy"]

        # ============ Soft/Hard Pruning评估 ============
        if ("merge_only" in eval_modes or "soft" in eval_modes or "hard" in eval_modes):
            valid_samples += 1

            # --- 使用批量预处理的结果（如果启用batch） ---
            if enable_true_batch and embeddings_merged_batch is not None:
                # 从批量结果中提取当前样本
                embeddings_merged = embeddings_merged_batch[idx:idx+1]  # (1, seq_len, dim)
                new_vision_pos = (new_vision_pos_batch[idx, 0].item(), new_vision_pos_batch[idx, 1].item())
                new_attention_mask = new_attention_mask_batch[idx:idx+1]  # (1, seq_len)
                question_embeddings = question_embeddings_batch[idx:idx+1]  # (1, q_len, dim)
                original_vision_pos = (original_vision_pos_batch[idx, 0].item(), original_vision_pos_batch[idx, 1].item())
            else:
                # --- Fallback: 单样本预处理（非batch模式） ---
                with torch.no_grad():
                    emb_info = backbone.preprocess(sample["image"], sample["question"], None)
                    original_embeddings = emb_info['embeddings']
                    original_vision_pos_tuple = emb_info['vision_token_positions']  # tuple (start, end)
                    # 转换为tensor格式以适配批量函数
                    original_vision_pos = torch.tensor(
                        [[original_vision_pos_tuple[0], original_vision_pos_tuple[1]]],
                        device=original_embeddings.device,
                        dtype=torch.long
                    )  # (1, 2)

                    # 获取未投影的vision features (1024维，CLIP输出)
                    vision_features_raw = emb_info['raw_vision_features']  # (1, 576, 1024)

                    if vision_features_raw is None:
                        raise ValueError("backbone未返回raw_vision_features，请检查backbone实现")

                    if enable_token_merger and token_merger is not None:
                        # 提取question embeddings（用于question-aware merger）
                        v_end = original_vision_pos[0, 1].item()
                        question_embeddings_for_merger = original_embeddings[:, v_end+1:, :]

                        # Token Merge (在1024维空间操作)
                        if config.method_settings.merger_type in ["question_aware", "fixed_pooling"]:
                            merge_result = token_merger(vision_features_raw, question_embeddings_for_merger, use_gumbel=eval_use_gumbel)
                        else:
                            merge_result = token_merger(vision_features_raw, use_gumbel=eval_use_gumbel)
                        merged_vision = merge_result['merged_features']  # (1, M, 1024)

                        # 投影到LLM维度 (1024 → 4096)
                        merged_vision = backbone.model.multi_modal_projector(merged_vision)  # (1, M, 4096)

                        # 替换vision部分
                        embeddings_merged, new_vision_pos, new_attention_mask = replace_vision_tokens_in_embeddings(
                            original_embeddings,
                            original_vision_pos,
                            merged_vision,
                            emb_info['attention_mask']
                        )
                    else:
                        # 禁用token merger，直接使用原始vision features
                        vision_features_projected = backbone.model.multi_modal_projector(vision_features_raw)  # (1, 576, 4096)
                        embeddings_merged, new_vision_pos, new_attention_mask = replace_vision_tokens_in_embeddings(
                            original_embeddings,
                            original_vision_pos,
                            vision_features_projected,
                            emb_info['attention_mask']
                        )

                    # 提取question embeddings（用于layer pruners）
                    question_embeddings = embeddings_merged[:, new_vision_pos[0, 1].item()+1:, :]

            # 统计
            # 处理不同情况下vision_pos的格式
            if isinstance(original_vision_pos, torch.Tensor):
                num_original_tokens = original_vision_pos[0, 1].item() - original_vision_pos[0, 0].item() + 1
            else:
                num_original_tokens = original_vision_pos[1] - original_vision_pos[0] + 1

            if isinstance(new_vision_pos, torch.Tensor):
                num_merged_tokens = new_vision_pos[0, 1].item() - new_vision_pos[0, 0].item() + 1
            else:
                num_merged_tokens = new_vision_pos[1] - new_vision_pos[0] + 1
            results["avg_original_tokens"] += float(num_original_tokens)
            results["avg_merged_tokens"] += float(num_merged_tokens)
            results["keep_ratio_merge"] += float(num_merged_tokens) / float(num_original_tokens)

            # --- Phase 1.5: Merge Only（只merge，不pruning） ---
            if "merge_only" in eval_modes and judge_fn and enable_token_merger and token_merger is not None:
                with torch.no_grad():
                    pred_merge_only = backbone.generate(
                        embeddings=embeddings_merged,
                        attention_mask=new_attention_mask,
                        max_new_tokens=20
                    )
                judge_result = judge_fn(pred_merge_only, ref_answer, sample)
                results["accuracy_merge_only"] += judge_result["accuracy"]

            # --- Phase 2: Soft Pruning（带layer pruners） ---
            if "soft" in eval_modes and judge_fn:
                # 注册hooks
                use_attn_residual = config["method_settings"].get("use_attn_residual", False)
                handles = register_multi_layer_hooks(
                    backbone,
                    layer_pruners,
                    new_vision_pos,
                    question_embeddings,
                    use_attn_residual=use_attn_residual
                )

                try:
                    with torch.no_grad():
                        pred_soft = backbone.generate(
                            embeddings=embeddings_merged,
                            attention_mask=new_attention_mask,
                            max_new_tokens=20
                        )
                    judge_result = judge_fn(pred_soft, ref_answer, sample)
                    results["accuracy_soft"] += judge_result["accuracy"]
                finally:
                    remove_hooks(handles)

            # --- Phase 3: Hard Pruning（真正移除tokens） ---
            if "hard" in eval_modes and judge_fn:
                # 获取hard pruning阈值
                hard_threshold = config["method_settings"].get("hard_pruning_threshold", 0.5)

                # 注册hard pruning - 使用新的model-level方法
                restore_fn, hard_context = register_hard_pruning_at_model_level(
                    backbone,
                    layer_pruners,
                    new_vision_pos,
                    question_embeddings,
                    threshold=hard_threshold,
                    use_gumbel=eval_use_gumbel
                )

                try:
                    with torch.no_grad():
                        pred_hard = backbone.generate(
                            embeddings=embeddings_merged.clone(),  # clone避免修改原始数据
                            attention_mask=new_attention_mask.clone(),
                            max_new_tokens=20
                        )
                    judge_result = judge_fn(pred_hard, ref_answer, sample)
                    results["accuracy_hard"] += judge_result["accuracy"]

                    # 收集hard pruning统计信息 - 计算所有LLM层的平均token数
                    pruning_stats = hard_context.get_stats()

                    # 获取LLM总层数（从backbone配置中获取）
                    total_llm_layers = backbone.model.model.language_model.config.num_hidden_layers

                    # 构建每一层的token数
                    # 初始：前面的层使用merge后的token数
                    # 每次剪枝后：后续层使用剪枝后的token数
                    tokens_per_layer = []
                    current_tokens = num_merged_tokens  # 初始token数（merge后的）

                    for layer_idx in range(total_llm_layers):
                        tokens_per_layer.append(current_tokens)

                        # 检查这一层之后是否有剪枝
                        for stat in pruning_stats:
                            if stat['layer_idx'] == layer_idx:
                                current_tokens = stat['kept_count']
                                break

                    # 计算平均token数（所有层的平均）
                    avg_tokens_across_layers = sum(tokens_per_layer) / len(tokens_per_layer) if len(tokens_per_layer) > 0 else 0.0

                    # 统计最终token数（用于参考）
                    final_v_start, final_v_end = hard_context.get_positions()
                    final_tokens = final_v_end - final_v_start + 1 if final_v_end >= final_v_start else 0

                    # 更新结果
                    results["hard_final_tokens"] += float(final_tokens)  # 最终token数
                    results["hard_avg_tokens"] += avg_tokens_across_layers  # 所有层的平均token数
                    results["hard_keep_ratio_total"] += float(final_tokens) / float(num_merged_tokens) if num_merged_tokens > 0 else 0.0
                    results["hard_avg_keep_ratio"] += avg_tokens_across_layers / float(num_merged_tokens) if num_merged_tokens > 0 else 0.0

                finally:
                    # 恢复原始forward方法
                    restore_fn()

            # 清理单样本变量（如果是fallback模式）
            if not enable_true_batch or embeddings_merged_batch is None:
                # 只在非batch模式下需要清理这些变量
                if 'vision_features_raw' in locals():
                    del vision_features_raw
                if 'embeddings_merged' in locals():
                    del embeddings_merged
                if enable_token_merger and token_merger is not None and 'merged_vision' in locals():
                    del merged_vision

    # 清理批量变量
    if enable_true_batch and embeddings_merged_batch is not None:
        del embeddings_merged_batch, new_vision_pos_batch, new_attention_mask_batch
        del question_embeddings_batch, original_vision_pos_batch
        if 'vision_features_raw_batch' in locals():
            del vision_features_raw_batch
        if enable_token_merger and token_merger is not None and 'merged_vision_batch' in locals():
            del merged_vision_batch

    torch.cuda.empty_cache()

    # 归一化结果
    total_samples = len(batch)
    if total_samples > 0:
        if "origin" in eval_modes:
            results["accuracy_baseline"] /= total_samples

    if valid_samples > 0:
        if "merge_only" in eval_modes and enable_token_merger:
            results["accuracy_merge_only"] /= valid_samples
        if "soft" in eval_modes:
            results["accuracy_soft"] /= valid_samples
        if "hard" in eval_modes:
            results["accuracy_hard"] /= valid_samples
            results["hard_final_tokens"] /= valid_samples
            results["hard_avg_tokens"] /= valid_samples  # 归一化平均token数
            results["hard_keep_ratio_total"] /= valid_samples
            results["hard_avg_keep_ratio"] /= valid_samples  # 归一化平均保留率
        results["keep_ratio_merge"] /= valid_samples
        results["avg_original_tokens"] /= valid_samples
        results["avg_merged_tokens"] /= valid_samples

    # 计算score用于Optuna优化
    if valid_samples > 0 and "hard" in eval_modes and "origin" in eval_modes:
        acc_drop = results["accuracy_baseline"] - results["accuracy_hard"]
        # 当禁用token merger时，keep_ratio_merge = 1.0（无merge）
        # 当启用时，keep_ratio_merge < 1.0（有merge）
        # hard_avg_keep_ratio 是所有LLM层的平均保留率（更准确地反映计算成本）

        # 修改score计算：
        # 1. 更突出"不掉点"（对acc_drop进行惩罚性加权）
        # 2. hard_avg_keep_ratio必须在[0.10, 0.60]区间，区间内越小越好

        # === 准确率惩罚 ===
        # 当 acc_drop > 0 时（性能下降），使用指数惩罚
        if acc_drop > 0:
            # 指数惩罚：掉点越多，惩罚越重
            # 例如：acc_drop=0.01 -> penalty≈1.5, acc_drop=0.05 -> penalty≈8
            acc_penalty = acc_drop * (1 + 50 * acc_drop)
        else:
            # 性能提升（极少见），给予奖励
            acc_penalty = acc_drop

        # === Keep Ratio约束（使用平均保留率） ===
        keep_ratio = results["hard_avg_keep_ratio"]  # 修改：使用所有层的平均保留率
        # 要求keep_ratio在[0.10, 0.60]区间
        if keep_ratio < 0.10:
            # 小于10%：严重惩罚（可能信息丢失过多）
            keep_ratio_penalty = 10.0 * (0.10 - keep_ratio)  # 距离下界越远，惩罚越重
        elif keep_ratio > 0.60:
            # 大于60%：严重惩罚（剪枝不足）
            keep_ratio_penalty = 10.0 * (keep_ratio - 0.60)  # 距离上界越远，惩罚越重
        else:
            # 在区间内：鼓励更小的值（权重较小）
            keep_ratio_penalty = 0.1 * keep_ratio

        # 总分 = -准确率惩罚 - keep ratio惩罚
        raw_score = -acc_penalty - keep_ratio_penalty

        # 使用EMA平滑score
        persistent_state = info["persistent_state"]
        ema_alpha = 0.3

        if "ema_score" not in persistent_state:
            ema_score = raw_score
        else:
            ema_score = ema_alpha * raw_score + (1 - ema_alpha) * persistent_state["ema_score"]

        persistent_state["ema_score"] = ema_score
        results["raw_score"] = raw_score
        results["score"] = ema_score
    else:
        results["score"] = float('-inf')

    torch.cuda.empty_cache()

    return results
