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
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        pruner: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None,
        adapter: Optional[nn.Module] = None
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.pruner = pruner
        self.discriminator = discriminator
        self.adapter = adapter
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
        if not self.is_pruning_layer:
            # 非剪枝层：直接调用原始 layer
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
            cumulative_vision_mask=cumulative_vision_mask,
            **kwargs
        )

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
        cumulative_vision_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """剪枝层的前向传播

        核心步骤：
        1. LayerNorm + Q/K/V 投影
        2. 计算 attention weights
        3. 提取 question→vision attention 并通过 pruner 生成 mask
        4. 计算 h_real（完整聚合）和 h_fake（剪枝后聚合）
        5. 用剪枝后的 attention 计算 output
        6. 残差连接 + MLP
        """
        layer = self.original_layer
        attn = layer.self_attn

        batch_size, seq_len, _ = hidden_states.shape

        # 获取配置
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        # === Step 1: LayerNorm + Q/K/V 投影 ===
        residual = hidden_states
        hidden_states_normed = layer.input_layernorm(hidden_states)

        # Q/K/V 投影
        query_states = attn.q_proj(hidden_states_normed)
        key_states = attn.k_proj(hidden_states_normed)
        value_states = attn.v_proj(hidden_states_normed)

        # 保存 query_states 用于 adapter（reshape 之前）
        query_states_flat = query_states  # (batch, seq, hidden_size)

        # Reshape: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
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
        # attn_scores: (batch, heads, seq, seq) - softmax 前的 scores
        attn_scores = torch.matmul(query_states, key_states.transpose(-2, -1)) * attn.scaling

        # 应用 causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_scores = attn_scores + causal_mask

        # 保存 softmax 前的 scores（用于后续 attn_weights_fake 计算）
        attn_scores_for_fake = attn_scores.clone()

        # Softmax（用于 h_real 和 q2v_attn 提取）
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # === Step 3: 提取 question→vision attention 并生成 mask ===
        n_vision = vision_end - vision_start

        # 逐样本提取 question→vision attention（每个样本的 question 位置不同）
        q2v_attn_list = []
        for i in range(batch_size):
            q_start, q_end = question_starts[i], question_ends[i]
            # question→vision attention: (heads, n_question_i, n_vision)
            q2v_i = attn_weights[i, :, q_start:q_end, vision_start:vision_end]
            # 平均所有 heads 和 question tokens: (n_vision,)
            q2v_avg_i = q2v_i.mean(dim=(0, 1))
            q2v_attn_list.append(q2v_avg_i)

        q2v_attn_avg = torch.stack(q2v_attn_list, dim=0)  # (batch, n_vision)

        # 提取 vision tokens 的 hidden states（用于 CrossAttentionPruner）
        # 使用 LayerNorm 后的 hidden states
        vision_hidden = hidden_states_normed[:, vision_start:vision_end, :]  # (batch, n_vision, hidden_size)

        # === 根据累积 mask 过滤 pruner 输入（模拟推理时的物理删除）===
        if cumulative_vision_mask is not None:
            # cumulative_vision_mask: (batch, n_vision), 1=keep, 0=pruned
            # 只把"逻辑上保留"的 tokens 传给 pruner
            # 为了保持可导性，用 mask 乘以 hidden/attn，被剪掉的位置变成 0
            # 然后 pruner 对这些位置的输出无关紧要（因为后面会用累积 mask 再过滤一次）
            vision_hidden_masked = vision_hidden * cumulative_vision_mask.unsqueeze(-1)
            q2v_attn_masked = q2v_attn_avg * cumulative_vision_mask
            # DEBUG: 打印 pruner 输入的统计信息
            print(f"\n[Pruner Input DEBUG - Layer {self.layer_idx}]")
            print(f"  cumulative_vision_mask.sum: {cumulative_vision_mask.sum().item()}")
            print(f"  vision_hidden.mean: {vision_hidden.mean().item():.6f}")
            print(f"  vision_hidden_masked.mean: {vision_hidden_masked.mean().item():.6f}")
            print(f"  vision_hidden_masked (non-zero).mean: {vision_hidden_masked[cumulative_vision_mask.bool()].mean().item():.6f}")
            print(f"  q2v_attn_avg.mean: {q2v_attn_avg.mean().item():.6f}")
            print(f"  q2v_attn_masked.mean: {q2v_attn_masked.mean().item():.6f}")
        else:
            vision_hidden_masked = vision_hidden
            q2v_attn_masked = q2v_attn_avg
            print(f"\n[Pruner Input DEBUG - Layer {self.layer_idx}]")
            print(f"  cumulative_vision_mask: None (first pruning layer)")
            print(f"  vision_hidden.mean: {vision_hidden.mean().item():.6f}")

        # Pruner 生成 mask（新接口：传入 vision_hidden 和 q2v_attn）
        hard_mask, pruner_info = self.pruner.forward_full(vision_hidden_masked, q2v_attn_masked)
        # hard_mask: (batch, n_vision), 0/1

        # DEBUG: 打印 pruner 输出
        print(f"  hard_mask (before cumulative).sum: {hard_mask.sum().item()}")

        # === 与累积 mask 结合：只有之前保留的 token 才能被当前层保留 ===
        if cumulative_vision_mask is not None:
            # 当前层只能在"之前保留"的 tokens 中选择
            # hard_mask = hard_mask * cumulative_vision_mask 会导致：
            # - 之前被剪掉的 token：无论当前层如何决定，最终都是 0
            # - 之前保留的 token：由当前层决定
            hard_mask = hard_mask * cumulative_vision_mask
            print(f"  hard_mask (after cumulative).sum: {hard_mask.sum().item()}")

        # === Step 4: 计算 h_real 和 h_fake ===
        # h_real: 用完整 attention 聚合
        # attn_output_real: (batch, heads, seq, head_dim)
        attn_output_real = torch.matmul(attn_weights, value_states)

        # h_fake: 用剪枝后的 attention 聚合
        # 关键：在 softmax 前应用 mask（-inf penalty），而不是 softmax 后 mask + renormalize
        # 这样与推理时物理删除的 softmax 归一化方式一致
        # hard_mask: (batch, n_vision), 1=keep, 0=prune
        batch_size_local, num_heads_local, seq_len_local, kv_len = attn_weights.shape

        # 创建 penalty: 1->0, 0->-10000（在 softmax 前加到 scores 上）
        # hard_mask: (batch, n_vision) -> (batch, 1, 1, n_vision)
        mask_expanded = hard_mask.unsqueeze(1).unsqueeze(2)
        penalty_vision = (1.0 - mask_expanded) * (-10000.0)
        penalty_vision = penalty_vision.expand(-1, num_heads_local, seq_len_local, -1)

        # 创建完整的 penalty（非 vision 部分为 0，vision 部分为 penalty）
        zeros_before = torch.zeros(batch_size_local, num_heads_local, seq_len_local, vision_start,
                                   device=attn_scores_for_fake.device, dtype=attn_scores_for_fake.dtype)
        zeros_after = torch.zeros(batch_size_local, num_heads_local, seq_len_local, kv_len - vision_end,
                                  device=attn_scores_for_fake.device, dtype=attn_scores_for_fake.dtype)
        full_penalty = torch.cat([zeros_before, penalty_vision, zeros_after], dim=-1)

        # 在 softmax 前应用 penalty
        attn_scores_fake = attn_scores_for_fake + full_penalty

        # Softmax（被剪掉的位置自然变成 0，且不影响归一化 - 与物理删除等效）
        attn_weights_fake = F.softmax(attn_scores_fake, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output_fake = torch.matmul(attn_weights_fake, value_states)

        # === Step 4.5: Adapter 修正 ===
        if self.adapter is not None:
            # (batch, heads, seq, head_dim) -> (batch, seq, hidden_size)
            attn_output_fake_flat = attn_output_fake.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            attn_output_fake_adapted = self.adapter(
                attn_output_fake_flat,
                mask=hard_mask,
                query=query_states_flat
            )
            # reshape back to (batch, heads, seq, head_dim)
            attn_output_fake = attn_output_fake_adapted.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # === Step 5: 提取 h_real 和 h_fake ===
        # h_real: 完整 attention 聚合
        # h_fake: adapter 修正后的剪枝 attention 聚合
        # 注意：autoregressive LLM 中，位置 i 的 hidden state 预测位置 i+1 的 token
        # 所以生成 answer[ans_start:ans_end] 需要 hidden[ans_start-1:ans_end-1]
        h_real_list = []
        h_fake_list = []
        for i in range(batch_size):
            ans_start, ans_end = answer_starts[i], answer_ends[i]
            # 生成 answer tokens 的位置：ans_start-1 到 ans_end-2（inclusive）
            gen_start = ans_start - 1
            gen_end = ans_end - 1  # exclusive，即到 ans_end-2
            h_real_list.append(attn_output_real[i, :, gen_start:gen_end, :])  # (heads, n_ans_i, head_dim)
            h_fake_list.append(attn_output_fake[i, :, gen_start:gen_end, :])  # (heads, n_ans_i, head_dim)

        # 返回 list，因为每个样本的 answer 长度不同
        h_real = h_real_list  # List[(heads, n_ans_i, head_dim)]
        h_fake = h_fake_list  # List[(heads, n_ans_i, head_dim)]

        # === Step 5: 用剪枝后的 attention 计算最终 output ===
        # 使用 fake attention 的输出作为最终输出
        attn_output = attn_output_fake

        # Reshape: (batch, heads, seq, head_dim) -> (batch, seq, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # Output projection
        attn_output = attn.o_proj(attn_output)

        # 残差连接
        hidden_states = residual + attn_output

        # === Step 6: MLP ===
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        if return_pruning_info:
            pruning_info = {
                'h_real': h_real,           # List[(heads, n_ans_i, head_dim)] - 每个样本的 answer tokens
                'h_fake': h_fake,           # List[(heads, n_ans_i, head_dim)] - 每个样本的 answer tokens
                'hard_mask': hard_mask,     # (batch, n_vision)
                'q2v_attn': q2v_attn_avg,   # (batch, n_vision)
                'keep_logits': pruner_info.get('keep_logits'),
                'attn_score': pruner_info.get('attn_score'),
                'token_score': pruner_info.get('token_score'),
                'baseline': pruner_info.get('baseline'),  # DEBUG: 添加 baseline
                'delta': pruner_info.get('delta'),        # DEBUG: 添加 delta
                'layer_idx': self.layer_idx,
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
