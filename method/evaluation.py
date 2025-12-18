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
    """两阶段剪枝的评估step函数

    评估模式：
    1. "origin" - Baseline（无剪枝）
    2. "merge_only" - 只Token Merge，不做Layer Pruning
    3. "soft" - Soft pruning（软mask，continuous 0-1）
    4. "hard" - Hard pruning（硬mask，binary 0/1）

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
        "hard_final_tokens": 0.0,
        "hard_keep_ratio_total": 0.0
    }

    valid_samples = 0

    def get_ref_answer(sample):
        """获取用于评估的参考答案"""
        if "answers" in sample and isinstance(sample["answers"], list) and len(sample["answers"]) > 0:
            return sample["answers"]
        return sample["answer"]

    for sample in batch:
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

            # --- Phase 1: Token Merge ---
            with torch.no_grad():
                emb_info = backbone.preprocess(sample["image"], sample["question"], None)
                original_embeddings = emb_info['embeddings']
                original_vision_pos = emb_info['vision_token_positions']

                # 获取未投影的vision features (1024维，CLIP输出)
                vision_features_raw = emb_info['raw_vision_features']  # (1, 576, 1024)

                if vision_features_raw is None:
                    raise ValueError("backbone未返回raw_vision_features，请检查backbone实现")

                if enable_token_merger and token_merger is not None:
                    # 提取question embeddings（用于question-aware merger）
                    # 结构(answer=None): [0:v_start] = "USER: "
                    #                   [v_start:v_end+1] = vision tokens
                    #                   [v_end+1:] = "\n{question}\nASSISTANT:"
                    # 只提取真正的question部分
                    _, v_end = original_vision_pos
                    question_embeddings_for_merger = original_embeddings[:, v_end+1:, :]

                    # Token Merge (在1024维空间操作)
                    if config.method_settings.merger_type in ["question_aware", "fixed_pooling"]:
                        merge_result = token_merger(vision_features_raw, question_embeddings_for_merger, use_gumbel=False)
                    else:
                        merge_result = token_merger(vision_features_raw, use_gumbel=False)
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
                # merge后序列长度变化，只提取question部分（不包含"USER: "）
                question_embeddings = embeddings_merged[:, new_vision_pos[1]+1:, :]

                # 统计
                num_original_tokens = original_vision_pos[1] - original_vision_pos[0] + 1
                num_merged_tokens = new_vision_pos[1] - new_vision_pos[0] + 1
                results["avg_original_tokens"] += float(num_original_tokens)
                results["avg_merged_tokens"] += float(num_merged_tokens)
                results["keep_ratio_merge"] += float(num_merged_tokens) / float(num_original_tokens)

            # --- Phase 1.5: Merge Only（只merge，不pruning） ---
            # 只在启用token merger时才有merge_only评估模式
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
                handles = register_multi_layer_hooks(
                    backbone,
                    layer_pruners,
                    new_vision_pos,
                    question_embeddings
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
                    threshold=hard_threshold
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

                    # 收集hard pruning统计信息
                    final_v_start, final_v_end = hard_context.get_positions()
                    final_tokens = final_v_end - final_v_start + 1 if final_v_end >= final_v_start else 0
                    results["hard_final_tokens"] += float(final_tokens)
                    results["hard_keep_ratio_total"] += float(final_tokens) / float(num_merged_tokens) if num_merged_tokens > 0 else 0.0

                finally:
                    # 恢复原始forward方法
                    restore_fn()

            # 清理
            del vision_features_raw, embeddings_merged
            # merged_vision 只在启用token_merger时存在
            if enable_token_merger and token_merger is not None:
                del merged_vision
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
            results["hard_keep_ratio_total"] /= valid_samples
        results["keep_ratio_merge"] /= valid_samples
        results["avg_original_tokens"] /= valid_samples
        results["avg_merged_tokens"] /= valid_samples

    # 计算score用于Optuna优化
    if valid_samples > 0 and "hard" in eval_modes and "origin" in eval_modes:
        acc_drop = results["accuracy_baseline"] - results["accuracy_hard"]
        # 当禁用token merger时，keep_ratio_merge = 1.0（无merge）
        # 当启用时，keep_ratio_merge < 1.0（有merge）
        # hard_keep_ratio_total 是最终的总体保留率（merge + layer pruning）
        # 我们希望最小化accuracy下降，同时鼓励更少的token（更小的keep ratio）
        raw_score = -acc_drop - 0.5 * results["hard_keep_ratio_total"]

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
