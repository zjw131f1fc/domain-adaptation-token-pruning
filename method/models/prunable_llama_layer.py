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
    """

    def __init__(
        self,
        original_layer: nn.Module,
        layer_idx: int,
        pruner: Optional[nn.Module] = None,
        discriminator: Optional[nn.Module] = None
    ):
        super().__init__()
        self.original_layer = original_layer
        self.layer_idx = layer_idx
        self.pruner = pruner
        self.discriminator = discriminator
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
        question_start: Optional[int] = None,
        question_end: Optional[int] = None,
        answer_start: Optional[int] = None,
        answer_end: Optional[int] = None,
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

        # === 剪枝层的特殊处理 ===
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
            question_start=question_start,
            question_end=question_end,
            answer_start=answer_start,
            answer_end=answer_end,
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
        question_start: int,
        question_end: int,
        answer_start: int,
        answer_end: int,
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
        n_question = question_end - question_start
        n_answer = answer_end - answer_start

        # question→vision attention: (batch, heads, n_question, n_vision)
        q2v_attn = attn_weights[:, :, question_start:question_end, vision_start:vision_end]

        # 平均所有 heads 和 question tokens: (batch, n_vision)
        q2v_attn_avg = q2v_attn.mean(dim=(1, 2))

        # Pruner 生成 mask
        hard_mask, pruner_info = self.pruner.forward_full(q2v_attn_avg)
        # hard_mask: (batch, n_vision), 0/1

        # === Step 4: 计算 h_real 和 h_fake ===
        # h_real: 用完整 attention 聚合
        # attn_output_real: (batch, heads, seq, head_dim)
        attn_output_real = torch.matmul(attn_weights, value_states)
        # 只取 answer tokens 的聚合结果
        h_real = attn_output_real[:, :, answer_start:answer_end, :]  # (batch, heads, n_answer, head_dim)

        # h_fake: 用剪枝后的 attention 聚合
        # 修改 attention weights，把被剪掉的 vision tokens 权重置零
        attn_weights_fake = attn_weights.clone()
        # hard_mask: (batch, n_vision) -> (batch, 1, 1, n_vision)
        mask_expanded = hard_mask.unsqueeze(1).unsqueeze(2)
        # 只修改对 vision tokens 的 attention
        attn_weights_fake[:, :, :, vision_start:vision_end] = \
            attn_weights_fake[:, :, :, vision_start:vision_end] * mask_expanded

        # 重新归一化（防止除零）
        attn_sum = attn_weights_fake.sum(dim=-1, keepdim=True)
        attn_weights_fake = attn_weights_fake / (attn_sum + 1e-8)

        attn_output_fake = torch.matmul(attn_weights_fake, value_states)
        h_fake = attn_output_fake[:, :, answer_start:answer_end, :]  # (batch, heads, n_answer, head_dim)

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
                'h_real': h_real,           # (batch, heads, n_answer, head_dim)
                'h_fake': h_fake,           # (batch, heads, n_answer, head_dim)
                'hard_mask': hard_mask,     # (batch, n_vision)
                'q2v_attn': q2v_attn_avg,   # (batch, n_vision)
                'importance': pruner_info['importance'],
                'residual': pruner_info['residual'],
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
