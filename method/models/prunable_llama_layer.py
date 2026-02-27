"""Attention Consistency Pruning - Prunable LLaMA Decoder Layer

包装 LlamaDecoderLayer，在剪枝层添加剪枝逻辑。

核心功能：
1. 非剪枝层：直接调用原始 layer
2. 剪枝层：
   - 手动执行 attention 计算以获取 weights 和 V
   - 计算 h_real（完整聚合）和 h_fake（剪枝后聚合）
   - 应用 mask 修改 attention weights
   - 返回剪枝信息供后续 loss 计算
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)


class PrunableLlamaDecoderLayer(nn.Module):
    """可剪枝的 LLaMA Decoder Layer

    包装原始的 LlamaDecoderLayer，在剪枝层执行特殊的前向传播。

    参数:
        original_layer: 原始的 LlamaDecoderLayer
        layer_idx: 层索引
        pruner: LayerPruner 实例（None 表示非剪枝层）
        discriminator: LayerDiscriminator 实例（None 表示非剪枝层）
        adapter: PruningAdapter 实例（None 表示非剪枝层）
        separated_adapters: (vision_adapter, text_adapter, answer_adapter) 元组
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        pruner: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        adapter: Optional[nn.Module] = None,
        separated_adapters: Optional[Tuple[nn.Module, nn.Module, nn.Module]] = None
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.pruner = pruner
        self.discriminator = discriminator
        self.adapter = adapter
        self.separated_adapters = separated_adapters  # (vision, text, answer)
        self.is_pruning_layer = pruner is not None

        # 从原始层获取配置
        self.hidden_size = original_layer.hidden_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        # === 剪枝相关参数 ===
        vision_start: Optional[int] = None,
        vision_end: Optional[int] = None,
        question_starts: Optional[list] = None,  # 每个样本的 question 开始位置
        question_ends: Optional[list] = None,    # 每个样本的 question 结束位置
        answer_starts: Optional[list] = None,    # 每个样本的 answer 开始位置
        answer_ends: Optional[list] = None,      # 每个样本的 answer 结束位置
        return_pruning_info: bool = False,
        cumulative_vision_mask: Optional[torch.Tensor] = None,  # 累积 vision mask
        detach_h_fake_for_adv: bool = False,  # 是否 detach h_fake（阻止 adv_loss 梯度流向 pruner）
        **kwargs
    ):
        """前向传播

        参数:
            hidden_states: (batch, seq, hidden_size)
            attention_mask: 4D causal mask
            position_ids: 位置 IDs
            vision_start/end: vision tokens 的位置范围
            question_start/end: question tokens 的位置范围
            answer_start/end: answer tokens 的位置范围
            return_pruning_info: 是否返回剪枝信息

        返回:
            hidden_states: (batch, seq, hidden_size)
            pruning_info: dict（如果 return_pruning_info=True）
        """
        # 如果没有提供剪枝参数（如 generate 调用），直接调用原始 layer
        if vision_start is None or vision_end is None:
            output = self.original_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )
            if return_pruning_info:
                return output, None
            return output

        if not self.is_pruning_layer:
            # 非剪枝层：如果有 cumulative_mask，应用 post-softmax masking
            if cumulative_vision_mask is not None and return_pruning_info:
                return self._forward_with_mask_only(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    vision_start=vision_start,
                    vision_end=vision_end,
                    cumulative_mask=cumulative_vision_mask,
                    **kwargs
                )
            else:
                # 没有 cumulative_mask，直接调用原始 layer
                output = self.original_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs
                )
                if return_pruning_info:
                    return output, None
                return output

        # === 剪枝层的特殊处理（训练模式）===
        return self._forward_with_pruning(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            vision_start=vision_start,
            vision_end=vision_end,
            question_starts=question_starts,
            question_ends=question_ends,
            answer_starts=answer_starts,
            answer_ends=answer_ends,
            return_pruning_info=return_pruning_info,
            cumulative_mask=cumulative_vision_mask,  # 兼容旧参数名
            detach_h_fake_for_adv=detach_h_fake_for_adv,
            **kwargs
        )

    def _forward_with_mask_only(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        vision_start: int,
        vision_end: int,
        cumulative_mask: torch.Tensor,
        **kwargs
    ):
        """非剪枝层应用 cumulative_mask 的前向传播（post-softmax masking）

        Args:
            cumulative_mask: 累积 mask (batch, n_vision)，1=保留，0=已剪掉

        Returns:
            hidden_states: 输出 hidden states
            None: 非剪枝层不返回 pruning_info
        """
        layer = self.original_layer
        attn = layer.self_attn

        batch_size, seq_len, hidden_size = hidden_states.shape
        n_vision = vision_end - vision_start
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 获取配置
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        # === Step 1: LayerNorm + Q/K/V 投影 ===
        residual = hidden_states
        hidden_states_normed = layer.input_layernorm(hidden_states)

        query_states = attn.q_proj(hidden_states_normed)
        key_states = attn.k_proj(hidden_states_normed)
        value_states = attn.v_proj(hidden_states_normed)

        # Reshape
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache if needed
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, num_kv_groups)
        value_states = repeat_kv(value_states, num_kv_groups)

        # === Step 2: 计算 Attention Weights ===
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * attn.scaling

        # 仅使用 causal mask（不使用 -inf 的 vision pruning mask）
        kv_len = key_states.shape[-2]
        min_val = torch.finfo(dtype).min
        causal_mask = torch.triu(
            torch.full((seq_len, kv_len), min_val, device=device, dtype=dtype),
            diagonal=1
        )
        attn_scores = attn_scores + causal_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype)

        # === Step 3: 应用 post-softmax masking ===
        # 创建完整的 mask（非 vision 部分为 1，vision 部分为 cumulative_mask）
        mask_expanded = cumulative_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_vision)
        ones_before = torch.ones(batch_size, num_heads, seq_len, vision_start,
                                 device=device, dtype=dtype)
        ones_after = torch.ones(batch_size, num_heads, seq_len, kv_len - vision_end,
                                device=device, dtype=dtype)
        mask_vision = mask_expanded.expand(-1, num_heads, seq_len, -1)
        full_mask = torch.cat([ones_before, mask_vision, ones_after], dim=-1)

        # 应用 mask 并重新归一化
        attn_weights_masked = attn_weights * full_mask
        attn_sum = attn_weights_masked.sum(dim=-1, keepdim=True)
        attn_weights_masked = (attn_weights_masked / (attn_sum + 1e-8)).to(dtype)

        # === Step 4: 计算 attention output ===
        attn_output = torch.matmul(attn_weights_masked, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn.o_proj(attn_output)

        # 残差连接
        hidden_states = residual + attn_output

        # === Step 5: MLP ===
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, None

    def _forward_with_pruning(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        vision_start: int,
        vision_end: int,
        question_starts: list,
        question_ends: list,
        answer_starts: list,
        answer_ends: list,
        return_pruning_info: bool,
        cumulative_mask: Optional[torch.Tensor] = None,  # 之前层的累积 mask
        detach_h_fake_for_adv: bool = False,  # 是否 detach h_fake（阻止 adv_loss 梯度流向 pruner）
        **kwargs
    ):
        """剪枝层的前向传播

        Args:
            cumulative_mask: 之前层的累积 mask (batch, n_vision)，1=保留，0=已剪掉

        Returns:
            hidden_states: 输出 hidden states
            pruning_info: {
                'current_mask': 当前层的决策 (batch, n_vision)
                'cumulative_mask': 新的累积 mask = cumulative_mask * current_mask
                ...
            }
        """
        layer = self.original_layer
        attn = layer.self_attn

        batch_size, seq_len, hidden_size = hidden_states.shape
        n_vision = vision_end - vision_start
        device = hidden_states.device
        dtype = hidden_states.dtype

        # 获取配置
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        # === Step 1: LayerNorm + Q/K/V 投影 ===
        residual = hidden_states
        hidden_states_normed = layer.input_layernorm(hidden_states)

        query_states = attn.q_proj(hidden_states_normed)
        key_states = attn.k_proj(hidden_states_normed)
        value_states = attn.v_proj(hidden_states_normed)

        # 保存 query_states 用于 adapter
        query_states_flat = query_states

        # Reshape
        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache if needed
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, num_kv_groups)
        value_states = repeat_kv(value_states, num_kv_groups)

        # === Step 2: 计算 Attention Weights ===
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * attn.scaling

        # 使用传入的 attention_mask（仅 causal mask，vision pruning 通过 post-softmax masking 实现）
        kv_len = key_states.shape[-2]
        if attention_mask is not None:
            # attention_mask: (batch, 1, seq_len, kv_len), 0=参与, -inf=不参与
            attn_scores = attn_scores + attention_mask
        else:
            # Fallback: 仅 causal mask
            min_val = torch.finfo(dtype).min
            causal_mask = torch.triu(
                torch.full((seq_len, kv_len), min_val, device=device, dtype=dtype),
                diagonal=1
            )
            attn_scores = attn_scores + causal_mask

        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(dtype)

        # === Step 3: 提取 question→vision attention 并生成 mask ===
        q2v_attn_list = []
        for i in range(batch_size):
            q_start, q_end = question_starts[i], question_ends[i]
            # print(f"  [q2v] sample {i}: q=[{q_start},{q_end})")
            q2v_i = attn_weights[i, :, q_start:q_end, vision_start:vision_end]
            q2v_avg_i = q2v_i.mean(dim=(0, 1))
            q2v_attn_list.append(q2v_avg_i)

        q2v_attn_avg = torch.stack(q2v_attn_list, dim=0)  # (batch, n_vision)

        # 提取 vision tokens 的 hidden states
        vision_hidden = hidden_states_normed[:, vision_start:vision_end, :]

        # 提取 question tokens 的 hidden states（用于条件化 pruner，仅在启用时）
        question_hidden = None
        question_lengths = None
        if self.pruner.use_question_condition:
            question_hidden_list = []
            question_lengths_list = []
            for i in range(batch_size):
                q_start, q_end = question_starts[i], question_ends[i]
                question_hidden_list.append(hidden_states_normed[i, q_start:q_end, :])
                question_lengths_list.append(q_end - q_start)
            # Pad to same length for batching
            max_q_len = max(qh.shape[0] for qh in question_hidden_list)
            question_hidden = torch.zeros(batch_size, max_q_len, hidden_size, device=device, dtype=dtype)
            for i, qh in enumerate(question_hidden_list):
                question_hidden[i, :qh.shape[0], :] = qh
            question_lengths = torch.tensor(question_lengths_list, device=device, dtype=torch.long)

        # Pruner 生成当前层的 mask
        current_mask, pruner_info = self.pruner.forward_full(
            vision_hidden, q2v_attn_avg,
            cumulative_vision_mask=cumulative_mask,
            question_hidden=question_hidden,
            question_lengths=question_lengths,
            return_debug=True
        )
        # current_mask: (batch, n_vision) - 当前层的决策

        # === 计算新的累积 mask ===
        # new_cumulative_mask = cumulative_mask * current_mask
        if cumulative_mask is not None:
            new_cumulative_mask = cumulative_mask * current_mask
        else:
            new_cumulative_mask = current_mask

        # === Step 4: 计算 h_real 和 h_fake ===
        # h_real: 完整 attention 聚合
        attn_output_real = torch.matmul(attn_weights, value_states)

        # h_fake: 当前层 mask 剪枝后的 attention 聚合
        # 使用 post-softmax mask + renormalize（与推理一致）
        # 如果 detach_h_fake_for_adv=True，则 detach mask，阻止 adv_loss 梯度流向 pruner
        if detach_h_fake_for_adv:
            current_mask_for_fake = current_mask.detach()
            cumulative_mask_for_adapter = new_cumulative_mask.detach()
        else:
            current_mask_for_fake = current_mask
            cumulative_mask_for_adapter = new_cumulative_mask
        mask_expanded = current_mask_for_fake.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_vision)

        # 创建完整的 mask（非 vision 部分为 1，vision 部分为 current_mask）
        ones_before = torch.ones(batch_size, num_heads, seq_len, vision_start,
                                 device=device, dtype=dtype)
        ones_after = torch.ones(batch_size, num_heads, seq_len, kv_len - vision_end,
                                device=device, dtype=dtype)
        mask_vision = mask_expanded.expand(-1, num_heads, seq_len, -1)
        full_mask = torch.cat([ones_before, mask_vision, ones_after], dim=-1)

        # 应用 mask 并重新归一化
        attn_weights_fake = attn_weights * full_mask
        attn_sum = attn_weights_fake.sum(dim=-1, keepdim=True)
        attn_weights_fake = (attn_weights_fake / (attn_sum + 1e-8)).to(dtype)

        attn_output_fake = torch.matmul(attn_weights_fake, value_states)

        # === Step 4.5: Adapter 修正 ===
        # 保存 Adapter 处理前的 attn_output_fake（用于可视化分析）
        attn_output_fake_before_adapter = attn_output_fake.clone()

        if self.separated_adapters is not None:
            # 分离式 Adapter：对 vision/text 分别处理（text 包含 question 和 answer）
            vision_adapter, text_adapter = self.separated_adapters
            attn_output_fake_flat = attn_output_fake.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            adapted_output = attn_output_fake_flat.clone()

            # Vision tokens: [vision_start, vision_end)
            vision_slice = attn_output_fake_flat[:, vision_start:vision_end, :]
            vision_query = query_states_flat[:, vision_start:vision_end, :]
            adapted_vision = vision_adapter(
                vision_slice,
                mask=cumulative_mask_for_adapter,
                query=vision_query,
                vision_hidden=vision_hidden  # 传入 vision hidden states
            )
            adapted_output[:, vision_start:vision_end, :] = adapted_vision

            # Text tokens (question): 每个样本可能不同
            for i in range(batch_size):
                q_start, q_end = question_starts[i], question_ends[i]
                text_slice = attn_output_fake_flat[i:i+1, q_start:q_end, :]
                text_query = query_states_flat[i:i+1, q_start:q_end, :]
                adapted_text = text_adapter(
                    text_slice,
                    mask=cumulative_mask_for_adapter[i:i+1],
                    query=text_query,
                    vision_hidden=vision_hidden[i:i+1]  # 传入 vision hidden states
                )
                adapted_output[i, q_start:q_end, :] = adapted_text.squeeze(0)

            # Generator tokens (answer): 使用 text_adapter
            for i in range(batch_size):
                gen_start = answer_starts[i] - 1
                gen_end = answer_ends[i] - 1
                gen_slice = attn_output_fake_flat[i:i+1, gen_start:gen_end, :]
                gen_query = query_states_flat[i:i+1, gen_start:gen_end, :]
                adapted_gen = text_adapter(
                    gen_slice,
                    mask=cumulative_mask_for_adapter[i:i+1],
                    query=gen_query,
                    vision_hidden=vision_hidden[i:i+1]  # 传入 vision hidden states
                )
                adapted_output[i, gen_start:gen_end, :] = adapted_gen.squeeze(0)

            attn_output_fake = adapted_output.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        elif self.adapter is not None:
            # 统一 Adapter（向后兼容）
            attn_output_fake_flat = attn_output_fake.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            attn_output_fake_adapted = self.adapter(
                attn_output_fake_flat,
                mask=cumulative_mask_for_adapter,
                query=query_states_flat,
                vision_hidden=vision_hidden  # 传入 vision hidden states
            )
            attn_output_fake = attn_output_fake_adapted.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # === Step 5: 提取 h_real 和 h_fake（answer 位置）- 暂存，后面 FFN 后再更新 ===
        # 注意：这里先保存 attention output，后面会用 FFN 后的 hidden states 替换
        h_real_attn_list = []
        h_fake_attn_list = []
        for i in range(batch_size):
            ans_start, ans_end = answer_starts[i], answer_ends[i]
            gen_start = ans_start - 1
            gen_end = ans_end - 1
            h_real_attn_list.append(attn_output_real[i, :, gen_start:gen_end, :])
            h_fake_attn_list.append(attn_output_fake[i, :, gen_start:gen_end, :])
        h_real_attn = h_real_attn_list
        h_fake_attn = h_fake_attn_list

        # === Step 6: 分别计算 real 和 fake 的完整前向（包括 o_proj + 残差 + FFN）===
        # Real 路径：使用完整的 attention output
        attn_output_real_flat = attn_output_real.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output_real_proj = attn.o_proj(attn_output_real_flat)
        hidden_states_real = residual + attn_output_real_proj
        residual_real = hidden_states_real
        hidden_states_real = layer.post_attention_layernorm(hidden_states_real)
        hidden_states_real = layer.mlp(hidden_states_real)
        hidden_states_real = residual_real + hidden_states_real

        # Fake 路径：使用剪枝后的 attention output（已经过 Adapter）
        attn_output_fake_flat = attn_output_fake.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output_fake_proj = attn.o_proj(attn_output_fake_flat)
        hidden_states_fake = residual + attn_output_fake_proj
        residual_fake = hidden_states_fake
        hidden_states_fake = layer.post_attention_layernorm(hidden_states_fake)
        hidden_states_fake = layer.mlp(hidden_states_fake)
        hidden_states_fake = residual_fake + hidden_states_fake

        # Fake 路径（Adapter 前）：用于可视化分析
        attn_output_fake_before_flat = attn_output_fake_before_adapter.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output_fake_before_proj = attn.o_proj(attn_output_fake_before_flat)
        hidden_states_fake_before = residual + attn_output_fake_before_proj
        residual_fake_before = hidden_states_fake_before
        hidden_states_fake_before = layer.post_attention_layernorm(hidden_states_fake_before)
        hidden_states_fake_before = layer.mlp(hidden_states_fake_before)
        hidden_states_fake_before = residual_fake_before + hidden_states_fake_before

        # === Step 7: 提取 FFN 后的 h_real, h_fake (Adapter 前), h_corrected (Adapter 后) ===
        h_real_list = []
        h_fake_list = []  # Adapter 前
        h_corrected_list = []  # Adapter 后
        for i in range(batch_size):
            ans_start, ans_end = answer_starts[i], answer_ends[i]
            gen_start = ans_start - 1
            gen_end = ans_end - 1
            # FFN 后的 hidden states: (batch, seq, hidden_size)
            # 转换为 (heads, n_ans, head_dim) 格式以兼容判别器
            h_real_i = hidden_states_real[i, gen_start:gen_end, :]  # (n_ans, hidden_size)
            h_fake_before_i = hidden_states_fake_before[i, gen_start:gen_end, :]  # Adapter 前
            h_corrected_i = hidden_states_fake[i, gen_start:gen_end, :]  # Adapter 后
            # 重塑为 (heads, n_ans, head_dim) 以兼容现有判别器
            h_real_i = h_real_i.view(-1, num_heads, head_dim).permute(1, 0, 2)  # (heads, n_ans, head_dim)
            h_fake_before_i = h_fake_before_i.view(-1, num_heads, head_dim).permute(1, 0, 2)
            h_corrected_i = h_corrected_i.view(-1, num_heads, head_dim).permute(1, 0, 2)
            h_real_list.append(h_real_i)
            h_fake_list.append(h_fake_before_i)
            h_corrected_list.append(h_corrected_i)

        h_real = h_real_list
        h_fake = h_fake_list  # 注意：这里改为 Adapter 前的版本
        h_corrected = h_corrected_list  # Adapter 后的版本

        # 使用 fake 路径的输出作为最终输出
        hidden_states = hidden_states_fake

        if return_pruning_info:
            pruning_info = {
                'h_real': h_real,
                'h_fake': h_fake,  # Adapter 前
                'h_corrected': h_corrected,  # Adapter 后
                'h_real_attn': h_real_attn,  # attention 输出（未过 FFN）
                'h_fake_attn': h_fake_attn,  # attention 输出（未过 FFN）
                'cumulative_mask': new_cumulative_mask,  # 新的累积 mask
                'current_mask': current_mask,            # 当前层的决策
                'q2v_attn': q2v_attn_avg,
                'keep_logits': pruner_info.get('keep_logits'),
                'attn_score': pruner_info.get('attn_score'),
                'token_score': pruner_info.get('token_score'),
                'baseline': pruner_info.get('baseline'),
                'delta': pruner_info.get('delta'),
                'gumbel_debug': pruner_info.get('gumbel_debug'),
                'layer_idx': self.layer_idx,
                'n_vision': n_vision,
            }
            return hidden_states, pruning_info

        return hidden_states


class PrunableLlamaLayerWrapper:
    """工具类：将普通的 LlamaDecoderLayer 转换为 PrunableLlamaDecoderLayer

    用法:
        wrapper = PrunableLlamaLayerWrapper(pruning_layers, pruners, discriminators)
        wrapper.wrap_layers(llm.layers)
    """

    def __init__(
        self,
        pruning_layers: list,
        pruners: nn.ModuleDict,
        discriminators: nn.ModuleDict
    ):
        self.pruning_layers = pruning_layers
        self.pruners = pruners
        self.discriminators = discriminators

    def wrap_layers(self, layers: nn.ModuleList) -> nn.ModuleList:
        """将指定层替换为可剪枝层

        参数:
            layers: LlamaModel.layers

        返回:
            修改后的 layers（原地修改）
        """
        for layer_idx in self.pruning_layers:
            original_layer = layers[layer_idx]
            pruner = self.pruners[str(layer_idx)]
            discriminator = self.discriminators[str(layer_idx)]

            layers[layer_idx] = PrunableLlamaDecoderLayer(
                original_layer=original_layer,
                layer_idx=layer_idx,
                pruner=pruner,
                discriminator=discriminator
            )

        return layers

    def unwrap_layers(self, layers: nn.ModuleList) -> nn.ModuleList:
        """还原剪枝层为原始层

        参数:
            layers: 包含 PrunableLlamaDecoderLayer 的 layers

        返回:
            还原后的 layers
        """
        for layer_idx in self.pruning_layers:
            prunable_layer = layers[layer_idx]
            if isinstance(prunable_layer, PrunableLlamaDecoderLayer):
                layers[layer_idx] = prunable_layer.original_layer

        return layers
