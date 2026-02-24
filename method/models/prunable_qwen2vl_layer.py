"""Attention Consistency Pruning - Prunable Qwen2-VL Decoder Layer

包装 Qwen2VLDecoderLayer，在剪枝层添加剪枝逻辑。

核心功能：
1. 非剪枝层：直接调用原始 layer
2. 剪枝层：
   - 手动执行 attention 计算以获取 weights 和 V
   - 计算 h_real（完整聚合）和 h_fake（剪枝后聚合）
   - 应用 mask 修改 attention weights
   - 返回剪枝信息供后续 loss 计算

与 LLaMA 的主要差异：
1. 使用 apply_multimodal_rotary_pos_emb 代替 apply_rotary_pos_emb
2. Q/K/V 投影有 bias
3. 支持 sliding window attention（不影响剪枝逻辑）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


class PrunableQwen2VLDecoderLayer(nn.Module):
    """可剪枝的 Qwen2-VL Decoder Layer

    包装原始的 Qwen2VLDecoderLayer，在剪枝层执行特殊的前向传播。

    参数:
        original_layer: 原始的 Qwen2VLDecoderLayer
        layer_idx: 层索引
        pruner: LayerPruner 实例（None 表示非剪枝层）
        discriminator: LayerDiscriminator 实例（None 表示非剪枝层）
        adapter: PruningAdapter 实例（None 表示非剪枝层）
        separated_adapters: (vision_adapter, text_adapter) 元组
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        pruner: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        adapter: Optional[nn.Module] = None,
        separated_adapters: Optional[Tuple[nn.Module, nn.Module]] = None
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.pruner = pruner
        self.discriminator = discriminator
        self.adapter = adapter
        self.separated_adapters = separated_adapters  # (vision, text)
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
        question_starts: Optional[list] = None,
        question_ends: Optional[list] = None,
        answer_starts: Optional[list] = None,
        answer_ends: Optional[list] = None,
        return_pruning_info: bool = False,
        cumulative_vision_mask: Optional[torch.Tensor] = None,
        detach_h_fake_for_adv: bool = False,
        **kwargs
    ):
        """前向传播

        参数:
            hidden_states: (batch, seq, hidden_size)
            attention_mask: 4D causal mask
            position_ids: 位置 IDs
            position_embeddings: (cos, sin) 元组，用于 M-RoPE
            vision_start/end: vision tokens 的位置范围
            question_starts/ends: question tokens 的位置范围（每个样本）
            answer_starts/ends: answer tokens 的位置范围（每个样本）
            return_pruning_info: 是否返回剪枝信息
            cumulative_vision_mask: 累积 vision mask

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
                return output[0] if isinstance(output, tuple) else output, None
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
                    return output[0] if isinstance(output, tuple) else output, None
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
            cumulative_mask=cumulative_vision_mask,
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
        """非剪枝层应用 cumulative_mask 的前向传播（post-softmax masking）"""
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

        # Apply M-RoPE
        cos, sin = position_embeddings
        mrope_section = attn.rope_scaling["mrope_section"]
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, mrope_section
        )

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

        # 仅使用 causal mask
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
        mask_expanded = cumulative_mask.unsqueeze(1).unsqueeze(2)
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
        cumulative_mask: Optional[torch.Tensor] = None,
        detach_h_fake_for_adv: bool = False,
        **kwargs
    ):
        """剪枝层的前向传播

        Args:
            cumulative_mask: 之前层的累积 mask (batch, n_vision)，1=保留，0=已剪掉

        Returns:
            hidden_states: 输出 hidden states
            pruning_info: 剪枝信息字典
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

        # Apply M-RoPE
        cos, sin = position_embeddings
        mrope_section = attn.rope_scaling["mrope_section"]
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, mrope_section
        )

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

        # 使用传入的 attention_mask（仅 causal mask）
        kv_len = key_states.shape[-2]
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        else:
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

        # === 计算新的累积 mask ===
        if cumulative_mask is not None:
            new_cumulative_mask = cumulative_mask * current_mask
        else:
            new_cumulative_mask = current_mask

        # === Step 4: 计算 h_real 和 h_fake ===
        # h_real: 完整 attention 聚合
        attn_output_real = torch.matmul(attn_weights, value_states)

        # h_fake: 当前层 mask 剪枝后的 attention 聚合
        if detach_h_fake_for_adv:
            current_mask_for_fake = current_mask.detach()
            cumulative_mask_for_adapter = new_cumulative_mask.detach()
        else:
            current_mask_for_fake = current_mask
            cumulative_mask_for_adapter = new_cumulative_mask
        mask_expanded = current_mask_for_fake.unsqueeze(1).unsqueeze(2)

        # 创建完整的 mask
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
        if self.separated_adapters is not None:
            vision_adapter, text_adapter = self.separated_adapters
            attn_output_fake_flat = attn_output_fake.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            adapted_output = attn_output_fake_flat.clone()

            # Vision tokens
            vision_slice = attn_output_fake_flat[:, vision_start:vision_end, :]
            vision_query = query_states_flat[:, vision_start:vision_end, :]
            adapted_vision = vision_adapter(vision_slice, mask=cumulative_mask_for_adapter, query=vision_query)
            adapted_output[:, vision_start:vision_end, :] = adapted_vision

            # Text tokens (question)
            for i in range(batch_size):
                q_start, q_end = question_starts[i], question_ends[i]
                text_slice = attn_output_fake_flat[i:i+1, q_start:q_end, :]
                text_query = query_states_flat[i:i+1, q_start:q_end, :]
                adapted_text = text_adapter(text_slice, mask=cumulative_mask_for_adapter[i:i+1], query=text_query)
                adapted_output[i, q_start:q_end, :] = adapted_text.squeeze(0)

            # Generator tokens (answer)
            for i in range(batch_size):
                gen_start = answer_starts[i] - 1
                gen_end = answer_ends[i] - 1
                gen_slice = attn_output_fake_flat[i:i+1, gen_start:gen_end, :]
                gen_query = query_states_flat[i:i+1, gen_start:gen_end, :]
                adapted_gen = text_adapter(gen_slice, mask=cumulative_mask_for_adapter[i:i+1], query=gen_query)
                adapted_output[i, gen_start:gen_end, :] = adapted_gen.squeeze(0)

            attn_output_fake = adapted_output.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)
        elif self.adapter is not None:
            attn_output_fake_flat = attn_output_fake.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            attn_output_fake_adapted = self.adapter(
                attn_output_fake_flat,
                mask=cumulative_mask_for_adapter,
                query=query_states_flat
            )
            attn_output_fake = attn_output_fake_adapted.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # === Step 5: 提取 h_real 和 h_fake（answer 位置）===
        h_real_list = []
        h_fake_list = []
        for i in range(batch_size):
            ans_start, ans_end = answer_starts[i], answer_ends[i]
            gen_start = ans_start - 1
            gen_end = ans_end - 1
            h_real_list.append(attn_output_real[i, :, gen_start:gen_end, :])
            h_fake_list.append(attn_output_fake[i, :, gen_start:gen_end, :])

        h_real = h_real_list
        h_fake = h_fake_list

        # === Step 6: 使用 fake attention 作为输出 ===
        attn_output = attn_output_fake
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn.o_proj(attn_output)

        # 残差连接
        hidden_states = residual + attn_output

        # === Step 7: MLP ===
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if return_pruning_info:
            pruning_info = {
                'h_real': h_real,
                'h_fake': h_fake,
                'cumulative_mask': new_cumulative_mask,
                'current_mask': current_mask,
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
