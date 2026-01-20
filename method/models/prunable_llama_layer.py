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

        # === 推理剪枝模式 ===
        # 缓存的 hard_mask，用于 generate 时复用
        self._cached_mask: Optional[torch.Tensor] = None
        self._inference_pruning_enabled: bool = False
        self._vision_start: Optional[int] = None
        self._vision_end: Optional[int] = None

    def enable_inference_pruning(self, vision_start: int, vision_end: int):
        """启用推理剪枝模式"""
        self._inference_pruning_enabled = True
        self._vision_start = vision_start
        self._vision_end = vision_end
        self._cached_mask = None  # 清除旧缓存

    def disable_inference_pruning(self):
        """禁用推理剪枝模式"""
        self._inference_pruning_enabled = False
        self._cached_mask = None
        self._vision_start = None
        self._vision_end = None

    def set_cached_mask(self, mask: torch.Tensor):
        """设置缓存的 mask（外部预计算后设置）"""
        self._cached_mask = mask

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

        # === 推理剪枝模式（generate 调用时使用缓存的 mask）===
        if self._inference_pruning_enabled and self._cached_mask is not None:
            return self._forward_with_cached_mask(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs
            )

        # 如果没有提供剪枝参数（如 generate 调用且未启用推理剪枝），直接调用原始 layer
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
        # attn_weights: (batch, heads, seq, seq)
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * attn.scaling

        # 应用 causal mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

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

        # Pruner 生成 mask（新接口：传入 vision_hidden 和 q2v_attn）
        hard_mask, pruner_info = self.pruner.forward_full(vision_hidden, q2v_attn_avg)
        # hard_mask: (batch, n_vision), 0/1

        # === Step 4: 计算 h_real 和 h_fake ===
        # h_real: 用完整 attention 聚合
        # attn_output_real: (batch, heads, seq, head_dim)
        attn_output_real = torch.matmul(attn_weights, value_states)

        # h_fake: 用剪枝后的 attention 聚合
        # 修改 attention weights，把被剪掉的 vision tokens 权重置零
        # 使用非 inplace 操作以保持梯度
        # hard_mask: (batch, n_vision) -> (batch, 1, 1, n_vision)
        mask_expanded = hard_mask.unsqueeze(1).unsqueeze(2)

        # 创建完整的 mask（非 vision 部分为 1，vision 部分为 hard_mask）
        # 使用 torch.cat 避免 inplace 操作
        batch_size_local, num_heads_local, seq_len_local, kv_len = attn_weights.shape
        ones_before = torch.ones(batch_size_local, num_heads_local, seq_len_local, vision_start,
                                  device=attn_weights.device, dtype=attn_weights.dtype)
        ones_after = torch.ones(batch_size_local, num_heads_local, seq_len_local, kv_len - vision_end,
                                 device=attn_weights.device, dtype=attn_weights.dtype)
        # mask_expanded 需要扩展到 (batch, heads, seq, n_vision)
        mask_vision = mask_expanded.expand(-1, num_heads_local, seq_len_local, -1)
        full_mask = torch.cat([ones_before, mask_vision, ones_after], dim=-1)

        # 非 inplace 方式应用 mask
        attn_weights_fake = attn_weights * full_mask

        # 重新归一化（防止除零）
        # NOTE: 当前是全局归一化，剪掉的 vision tokens 权重会分配给所有剩余 tokens
        # （包括 system, question, answer 等非 vision tokens）
        # 另一种选择是只在 vision tokens 内部重新分配，待后续实验对比
        attn_sum = attn_weights_fake.sum(dim=-1, keepdim=True)
        attn_weights_fake = attn_weights_fake / (attn_sum + 1e-8)

        attn_output_fake = torch.matmul(attn_weights_fake, value_states)

        # === Step 4.5: Adapter 修正 ===
        # 将 attn_output_fake 通过 adapter 修正
        if self.adapter is not None:
            # attn_output_fake: (batch, heads, seq, head_dim) -> (batch, seq, heads, head_dim)
            attn_output_fake_perm = attn_output_fake.permute(0, 2, 1, 3)
            # reshape to (batch, seq, hidden_size)
            attn_output_fake_flat = attn_output_fake_perm.reshape(batch_size, seq_len, -1)
            # adapter 修正
            attn_output_fake_adapted = self.adapter(attn_output_fake_flat)
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

    def _forward_with_cached_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Any],
        use_cache: bool,
        cache_position: Optional[torch.LongTensor],
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        **kwargs
    ):
        """使用缓存的 mask 进行推理（用于 generate）

        与 _forward_with_pruning 类似，但使用预先计算的 mask，
        不计算 h_real/h_fake，也不返回 pruning_info。
        """
        layer = self.original_layer
        attn = layer.self_attn

        batch_size, seq_len, _ = hidden_states.shape

        # 获取配置
        num_heads = attn.config.num_attention_heads
        num_kv_heads = attn.config.num_key_value_heads
        head_dim = attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        vision_start = self._vision_start
        vision_end = self._vision_end

        # === Step 1: LayerNorm + Q/K/V 投影 ===
        residual = hidden_states
        hidden_states_normed = layer.input_layernorm(hidden_states)

        query_states = attn.q_proj(hidden_states_normed)
        key_states = attn.k_proj(hidden_states_normed)
        value_states = attn.v_proj(hidden_states_normed)

        query_states = query_states.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Handle KV cache
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Repeat KV for GQA
        key_states = repeat_kv(key_states, num_kv_groups)
        value_states = repeat_kv(value_states, num_kv_groups)

        # === Step 2: 计算 Attention Weights ===
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * attn.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # === Step 3: 应用缓存的 mask ===
        # _cached_mask: (batch, n_vision)
        hard_mask = self._cached_mask

        # 确保 batch size 匹配（可能因为 generate 时 batch=1）
        if hard_mask.shape[0] != batch_size:
            hard_mask = hard_mask[:batch_size]

        # 扩展 mask: (batch, n_vision) -> (batch, 1, 1, n_vision)
        mask_expanded = hard_mask.unsqueeze(1).unsqueeze(2)

        # 只对 vision tokens 部分应用 mask
        # 注意：在 KV cache 模式下，key_states 可能包含历史 tokens
        kv_seq_len = key_states.shape[-2]
        if kv_seq_len >= vision_end:
            # 完整序列，直接应用 mask
            attn_weights[:, :, :, vision_start:vision_end] = \
                attn_weights[:, :, :, vision_start:vision_end] * mask_expanded

        # 重新归一化
        attn_sum = attn_weights.sum(dim=-1, keepdim=True)
        attn_weights = attn_weights / (attn_sum + 1e-8)

        # === Step 4: 计算 output ===
        attn_output = torch.matmul(attn_weights, value_states)

        # === Step 4.5: Adapter 修正（推理时也使用）===
        if self.adapter is not None:
            attn_output_perm = attn_output.permute(0, 2, 1, 3)
            attn_output_flat = attn_output_perm.reshape(batch_size, seq_len, -1)
            attn_output_adapted = self.adapter(attn_output_flat)
            attn_output = attn_output_adapted.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        attn_output = attn.o_proj(attn_output)

        hidden_states = residual + attn_output

        # === Step 5: MLP ===
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

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
