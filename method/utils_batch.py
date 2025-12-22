"""Vision Token Pruning - Batch化辅助函数

提供batch处理所需的工具函数
"""

import torch
from typing import Tuple, List, Optional


def replace_vision_tokens_in_embeddings_batch(
    full_embeddings: torch.Tensor,
    original_vision_pos: torch.Tensor,
    merged_vision_features: torch.Tensor,
    original_attention_mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将合并后的vision features替换回完整embeddings（Batch版本）

    参数:
        full_embeddings: (batch_size, seq_len, d) - 原始完整序列
        original_vision_pos: (batch_size, 2) - 原始vision位置，每行为(start, end)
        merged_vision_features: (batch_size, M, d) - 合并后的vision features
        original_attention_mask: (batch_size, seq_len) - 原始attention mask

    返回:
        new_embeddings: (batch_size, new_seq_len, d) - 更新后的序列
        new_vision_pos: (batch_size, 2) - 更新后的vision位置
        new_attention_mask: (batch_size, new_seq_len) - 更新后的mask

    注意：batch模式要求所有样本的vision位置相同
    """
    batch_size = full_embeddings.shape[0]

    # 检查所有样本的vision位置是否相同（batch化要求）
    v_start_first = original_vision_pos[0, 0].item()
    v_end_first = original_vision_pos[0, 1].item()

    for b in range(1, batch_size):
        if original_vision_pos[b, 0].item() != v_start_first or original_vision_pos[b, 1].item() != v_end_first:
            raise ValueError(
                f"Batch mode requires all samples have the same vision token positions. "
                f"Sample 0: ({v_start_first}, {v_end_first}), "
                f"Sample {b}: ({original_vision_pos[b, 0].item()}, {original_vision_pos[b, 1].item()})"
            )

    v_start, v_end = v_start_first, v_end_first

    # 批量拼接: text_before + merged_vision + text_after
    new_embeddings = torch.cat([
        full_embeddings[:, :v_start, :],           # 图像前的文本
        merged_vision_features,                     # 合并后的vision tokens
        full_embeddings[:, v_end+1:, :]            # 图像后的文本
    ], dim=1)

    # 更新vision位置（所有样本相同）
    new_v_start = v_start
    new_v_end = v_start + merged_vision_features.shape[1] - 1
    new_vision_pos = torch.tensor(
        [[new_v_start, new_v_end]] * batch_size,
        dtype=torch.long,
        device=full_embeddings.device
    )

    # 更新attention_mask
    before_mask = original_attention_mask[:, :v_start]
    after_mask = original_attention_mask[:, v_end+1:]
    vision_mask = torch.ones(
        (batch_size, merged_vision_features.shape[1]),
        device=merged_vision_features.device,
        dtype=original_attention_mask.dtype
    )
    new_attention_mask = torch.cat([before_mask, vision_mask, after_mask], dim=1)

    return new_embeddings, new_vision_pos, new_attention_mask


def register_multi_layer_hooks_batch(
    backbone,
    layer_pruners,
    vision_positions: torch.Tensor,
    question_embeddings: torch.Tensor,
    mask_collector: Optional[List] = None,
    use_attn_residual: bool = False
):
    """注册多层pruning hooks（Batch版本）

    参数:
        backbone: Backbone模型
        layer_pruners: LayerSpecificPruner实例
        vision_positions: (batch_size, 2) - vision token位置，每行为(start, end)
        question_embeddings: (batch_size, q_len, d) - question embeddings
        mask_collector: 用于收集每层的soft_mask（可选）
        use_attn_residual: 是否启用attention residual

    返回:
        handles: hook句柄列表

    注意：batch模式要求所有样本的vision位置相同
    """
    batch_size = vision_positions.shape[0]

    # 检查所有样本的vision位置是否相同
    v_start_first = vision_positions[0, 0].item()
    v_end_first = vision_positions[0, 1].item()

    for b in range(1, batch_size):
        if vision_positions[b, 0].item() != v_start_first or vision_positions[b, 1].item() != v_end_first:
            raise ValueError(
                f"Batch hook registration requires all samples have the same vision positions. "
                f"Sample 0: ({v_start_first}, {v_end_first}), "
                f"Sample {b}: ({vision_positions[b, 0].item()}, {vision_positions[b, 1].item()})"
            )

    # 使用统一的vision位置
    vision_pos_tuple = (v_start_first, v_end_first)

    handles = []
    pruning_layers = layer_pruners.get_all_layers()

    for layer_idx in pruning_layers:
        pruner = layer_pruners.get_pruner(layer_idx)

        def make_hook(pruner_obj, layer_id):
            def hook_fn(module, input_tuple, output):
                # output是(batch_size, seq_len, hidden_dim)的tuple
                if isinstance(output, tuple):
                    hidden_states = output[0]
                else:
                    hidden_states = output

                v_start, v_end = vision_pos_tuple
                n_vision = v_end - v_start + 1

                # 提取vision tokens（batch维度）
                vision_hidden = hidden_states[:, v_start:v_end+1, :]  # (batch_size, n_vision, d)

                # Pruner forward（batch处理）
                pruner_result = pruner_obj(
                    vision_hidden,
                    question_embeddings,
                    use_gumbel=True
                )

                soft_mask = pruner_result['soft_mask']  # (batch_size, n_vision)
                importance_scores = pruner_result.get('importance_scores', None)

                # 收集mask用于sparsity loss
                if mask_collector is not None:
                    mask_collector.append(soft_mask.detach())

                # 应用soft_mask到vision tokens
                masked_vision = vision_hidden * soft_mask.unsqueeze(-1)  # (batch_size, n_vision, d)

                # Attention residual（如果启用）
                if use_attn_residual and importance_scores is not None:
                    # 提取text hidden states
                    text_before = hidden_states[:, :v_start, :]
                    text_after = hidden_states[:, v_end+1:, :]
                    text_hidden = torch.cat([text_before, text_after], dim=1)  # (batch_size, text_len, d)

                    # Cross-attention: text → vision
                    # 简化版：使用importance scores作为权重
                    attn_weights = torch.softmax(importance_scores, dim=-1)  # (batch_size, n_vision)
                    text_context = torch.mean(text_hidden, dim=1, keepdim=True)  # (batch_size, 1, d)
                    text_context = text_context.expand(-1, n_vision, -1)  # (batch_size, n_vision, d)

                    # 加权融合
                    if hasattr(pruner_obj, 'attn_residual_weight'):
                        alpha = torch.sigmoid(pruner_obj.attn_residual_weight)
                    else:
                        alpha = 0.5

                    masked_vision = alpha * masked_vision + (1 - alpha) * text_context * soft_mask.unsqueeze(-1)

                # 替换vision部分
                new_hidden_states = hidden_states.clone()
                new_hidden_states[:, v_start:v_end+1, :] = masked_vision

                if isinstance(output, tuple):
                    return (new_hidden_states,) + output[1:]
                else:
                    return new_hidden_states

            return hook_fn

        # 注册hook
        layer_module = backbone.model.language_model.model.layers[layer_idx]
        handle = layer_module.register_forward_hook(make_hook(pruner, layer_idx))
        handles.append(handle)

    return handles
