"""Vision Token Pruning - Batch化辅助函数

提供batch处理所需的工具函数
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional


def compute_task_loss_batch(
    logits: torch.Tensor,
    answer_positions: torch.Tensor,
    answers: List[str],
    processor
) -> torch.Tensor:
    """批量计算任务损失（预测answer的交叉熵损失）

    参数:
        logits: (batch_size, seq_len, vocab_size) - 模型输出的logits
        answer_positions: (batch_size, 2) - answer在序列中的位置（支持负索引）
        answers: List[str] - 答案文本列表
        processor: tokenizer所在的processor

    返回:
        task_loss: torch.Tensor - 批量平均后的交叉熵损失

    注意：当前实现要求batch内所有answer位置相同（通过padding保证）
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # 检查是否所有answer位置相同
    # 如果位置相同，可以进行高效的批量计算
    positions_unique = torch.unique(answer_positions, dim=0)
    if positions_unique.shape[0] == 1:
        # 所有样本的answer位置相同，可以批量处理
        answer_start, answer_end = answer_positions[0].tolist()

        # 转换负索引
        if answer_start < 0:
            answer_start = seq_len + answer_start
        if answer_end < 0:
            answer_end = seq_len + answer_end

        # 批量tokenize所有answers
        answer_token_ids_list = []
        for answer in answers:
            token_ids = processor.tokenizer.encode(answer, add_special_tokens=False)
            if len(token_ids) == 0:
                raise ValueError(f"answer '{answer}' 被分词后长度为0")
            answer_token_ids_list.append(token_ids)

        # 检查所有answer长度是否相同（通常由于padding会相同）
        answer_lengths = [len(ids) for ids in answer_token_ids_list]
        if len(set(answer_lengths)) == 1:
            # 长度相同，可以批量处理
            answer_token_ids = torch.tensor(
                answer_token_ids_list,
                device=device,
                dtype=torch.long
            )  # (batch_size, answer_len)

            # 提取用于预测的logits
            logits_for_answer = logits[:, answer_start-1:answer_end, :]  # (batch_size, answer_len, vocab_size)

            # 批量计算交叉熵
            loss = F.cross_entropy(
                logits_for_answer.reshape(-1, vocab_size),  # (batch_size * answer_len, vocab_size)
                answer_token_ids.reshape(-1),  # (batch_size * answer_len,)
                reduction='mean'
            )
            return loss
        else:
            # 长度不同，退回到逐样本计算
            pass

    # Fallback：逐样本计算（当位置或长度不一致时）
    total_loss = torch.tensor(0.0, device=device)
    for b in range(batch_size):
        answer_start, answer_end = answer_positions[b].tolist()

        # 转换负索引
        if answer_start < 0:
            answer_start = seq_len + answer_start
        if answer_end < 0:
            answer_end = seq_len + answer_end

        answer_token_ids_list = processor.tokenizer.encode(answers[b], add_special_tokens=False)
        if len(answer_token_ids_list) == 0:
            raise ValueError(f"answer '{answers[b]}' 被分词后长度为0")

        answer_token_ids = torch.tensor(answer_token_ids_list, device=device, dtype=torch.long)

        logits_for_answer = logits[b:b+1, answer_start-1:answer_end, :]

        loss_b = F.cross_entropy(
            logits_for_answer.reshape(-1, vocab_size),
            answer_token_ids,
            reduction='mean'
        )
        total_loss = total_loss + loss_b

    return total_loss / batch_size


def extract_text_hidden_batch(
    hidden_states: torch.Tensor,
    vision_positions: torch.Tensor
) -> torch.Tensor:
    """向量化提取text hidden states（去除vision tokens）

    参数:
        hidden_states: (batch_size, seq_len, hidden_dim)
        vision_positions: (batch_size, 2) - 每行为 (v_start, v_end)

    返回:
        text_hidden: (batch_size, text_len, hidden_dim) - 拼接后的text部分

    注意：要求batch内所有样本的vision位置相同
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape

    # 验证所有样本的vision位置相同
    v_start = vision_positions[0, 0].item()
    v_end = vision_positions[0, 1].item()

    # 向量化提取：text_before + text_after
    text_before = hidden_states[:, :v_start, :]  # (batch_size, v_start, dim)
    text_after = hidden_states[:, v_end+1:, :]   # (batch_size, seq_len-v_end-1, dim)

    # 拼接
    text_hidden = torch.cat([text_before, text_after], dim=1)  # (batch_size, text_len, dim)

    return text_hidden


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
                # 当use_attn_residual=False时，跳过text_to_vision_attn计算以节约时间
                if use_attn_residual:
                    # 提取text→vision attention
                    text_to_vision_attn = None
                    q_len = question_embeddings.shape[1]
                    if v_end + 1 + q_len <= hidden_states.shape[1]:
                        text_hidden = hidden_states[:, v_end+1:v_end+1+q_len, :]  # (batch, q_len, d)

                        # 计算text→vision的attention分数
                        d_model = text_hidden.shape[-1]
                        attn_scores = torch.matmul(text_hidden, vision_hidden.transpose(1, 2))  # (batch, q_len, n_vision)
                        attn_scores = attn_scores / (d_model ** 0.5)
                        attn_weights = torch.softmax(attn_scores, dim=-1)  # (batch, q_len, n_vision)

                        # 平均所有text token对vision的attention
                        text_to_vision_attn = attn_weights.mean(dim=1)  # (batch, n_vision)

                    soft_mask = pruner_obj(
                        vision_hidden,
                        question_embeddings,
                        use_gumbel=True,
                        text_to_vision_attn=text_to_vision_attn
                    )  # (batch_size, n_vision)
                else:
                    # 不启用attn_residual时，直接调用pruner，无需计算attention
                    soft_mask = pruner_obj(
                        vision_hidden,
                        question_embeddings,
                        use_gumbel=True,
                        text_to_vision_attn=None
                    )  # (batch_size, n_vision)

                # 收集mask用于sparsity loss
                if mask_collector is not None:
                    mask_collector.append(soft_mask.detach())

                # 应用soft_mask到vision tokens
                soft_mask = soft_mask.to(vision_hidden.dtype)
                masked_vision = vision_hidden * soft_mask.unsqueeze(-1)  # (batch_size, n_vision, d)

                # 替换vision部分
                new_hidden_states = hidden_states.clone()
                new_hidden_states[:, v_start:v_end+1, :] = masked_vision

                if isinstance(output, tuple):
                    return (new_hidden_states,) + output[1:]
                else:
                    return new_hidden_states

            return hook_fn

        # 注册hook
        layer_module = backbone.model.model.language_model.layers[layer_idx]
        handle = layer_module.register_forward_hook(make_hook(pruner, layer_idx))
        handles.append(handle)

    return handles
