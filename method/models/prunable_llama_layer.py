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
        detach_mask_for_adv: bool = False,  # 是否对 adv_loss 的 h_fake 使用 detached mask
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
            detach_mask_for_adv=detach_mask_for_adv,
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
        detach_mask_for_adv: bool = False,
        **kwargs
    ):
        """剪枝层的前向传播（完全对齐推理）

        注意：物理删除由外层 (prunable_llava.py) 管理。
        这里收到的 hidden_states 已经是之前层删除后的序列。
        vision_start/end 是当前序列中 vision tokens 的位置。
        cumulative_vision_mask 用于追踪哪些原始 tokens 仍然保留（用于 scatter 回原始维度）。

        核心步骤：
        1. 在当前（可能已删除部分 tokens 的）序列上计算 attention
        2. 提取 q2v_attn 并通过 pruner 生成当前层的 mask
        3. 计算 h_real 和 h_fake
        4. 返回 mask 供外层做物理删除
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

        # 创建 causal mask
        kv_len = key_states.shape[-2]
        causal_mask = torch.triu(
            torch.full((seq_len, kv_len), float('-inf'), device=device, dtype=dtype),
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

        # Pruner 生成 mask（输入维度 = 当前 n_vision，与推理完全一致）
        hard_mask, pruner_info = self.pruner.forward_full(vision_hidden, q2v_attn_avg, return_debug=True)
        # hard_mask: (batch, n_vision) - 当前 vision tokens 的 mask

        # === 将当前层的 mask scatter 回原始 n_vision_orig 维度（如果需要）===
        # cumulative_vision_mask 告诉我们当前 n_vision tokens 对应原始的哪些位置
        if cumulative_vision_mask is not None:
            # cumulative_vision_mask: (batch, n_vision_orig), 1=kept, 0=pruned
            n_vision_orig = cumulative_vision_mask.shape[1]
            # 强制二值化，避免 bfloat16 精度问题
            cumulative_vision_mask_clean = (cumulative_vision_mask > 0.5).to(dtype)

            # 对每个样本单独 scatter，支持不同样本有不同的 mask
            hard_mask_full_list = []
            for b in range(batch_size):
                kept_indices_b = cumulative_vision_mask_clean[b].nonzero(as_tuple=True)[0]
                n_kept_b = len(kept_indices_b)
                # hard_mask[b] 的长度是 n_vision（union mask 的保留数）
                # 但该样本只保留了 n_kept_b 个，所以只取前 n_kept_b 个值
                hm_b = torch.zeros(n_vision_orig, device=device, dtype=dtype)
                hm_b = hm_b.scatter(0, kept_indices_b, hard_mask[b, :n_kept_b])
                hard_mask_full_list.append(hm_b)
            hard_mask_full = torch.stack(hard_mask_full_list, dim=0)

            # 同时保存每个样本的 kept_indices 供后续使用
            kept_indices_per_sample = [
                cumulative_vision_mask_clean[b].nonzero(as_tuple=True)[0]
                for b in range(batch_size)
            ]
        else:
            hard_mask_full = hard_mask
            n_vision_orig = n_vision
            kept_indices_per_sample = None

        # === Step 4: 计算 h_real 和 h_fake ===
        # h_real: 完整 attention 聚合
        attn_output_real = torch.matmul(attn_weights, value_states)

        # h_fake: 当前层 mask 剪枝后的 attention 聚合
        # 使用 post-softmax mask + renormalize（与推理一致，更接近物理删除效果）
        mask_expanded = hard_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_vision)

        # 创建完整的 mask（非 vision 部分为 1，vision 部分为 hard_mask）
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
        if self.adapter is not None:
            attn_output_fake_flat = attn_output_fake.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
            attn_output_fake_adapted = self.adapter(
                attn_output_fake_flat,
                mask=hard_mask_full,  # 使用原始维度的 mask
                query=query_states_flat
            )
            attn_output_fake = attn_output_fake_adapted.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

        # === Step 5: 提取 h_real 和 h_fake（answer 位置）===
        h_real_list = []
        h_fake_list = []
        h_fake_for_adv_list = []  # 用于 adv_loss 的 h_fake（可选 detach mask）
        for i in range(batch_size):
            ans_start, ans_end = answer_starts[i], answer_ends[i]
            gen_start = ans_start - 1
            gen_end = ans_end - 1
            h_real_list.append(attn_output_real[i, :, gen_start:gen_end, :])
            h_fake_list.append(attn_output_fake[i, :, gen_start:gen_end, :])

        h_real = h_real_list
        h_fake = h_fake_list

        # 如果需要 detach mask for adv，额外计算一个 h_fake_for_adv
        if detach_mask_for_adv:
            # 使用 detached mask 重新计算 attn_output_fake_for_adv
            mask_expanded_detached = hard_mask.detach().unsqueeze(1).unsqueeze(2)
            mask_vision_detached = mask_expanded_detached.expand(-1, num_heads, seq_len, -1)
            full_mask_detached = torch.cat([ones_before, mask_vision_detached, ones_after], dim=-1)

            attn_weights_fake_detached = attn_weights * full_mask_detached
            attn_sum_detached = attn_weights_fake_detached.sum(dim=-1, keepdim=True)
            attn_weights_fake_detached = (attn_weights_fake_detached / (attn_sum_detached + 1e-8)).to(dtype)
            attn_output_fake_for_adv = torch.matmul(attn_weights_fake_detached, value_states)

            # Adapter 修正（adapter 的梯度仍然可以传递）
            if self.adapter is not None:
                attn_output_fake_for_adv_flat = attn_output_fake_for_adv.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
                attn_output_fake_for_adv_adapted = self.adapter(
                    attn_output_fake_for_adv_flat,
                    mask=hard_mask_full.detach(),  # mask 也 detach
                    query=query_states_flat
                )
                attn_output_fake_for_adv = attn_output_fake_for_adv_adapted.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3)

            for i in range(batch_size):
                ans_start, ans_end = answer_starts[i], answer_ends[i]
                gen_start = ans_start - 1
                gen_end = ans_end - 1
                h_fake_for_adv_list.append(attn_output_fake_for_adv[i, :, gen_start:gen_end, :])
            h_fake_for_adv = h_fake_for_adv_list
        else:
            h_fake_for_adv = h_fake  # 不 detach 时，直接使用 h_fake

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
            # 返回原始维度的 q2v_attn（对每个样本单独处理）
            if cumulative_vision_mask is not None and kept_indices_per_sample is not None:
                q2v_attn_full = torch.zeros(batch_size, n_vision_orig, device=device, dtype=dtype)
                for b in range(batch_size):
                    kept_indices_b = kept_indices_per_sample[b]
                    n_kept_b = len(kept_indices_b)
                    # q2v_attn_avg[b] 长度是 n_vision（union），只取前 n_kept_b 个
                    q2v_attn_full[b, kept_indices_b] = q2v_attn_avg[b, :n_kept_b]
            else:
                q2v_attn_full = q2v_attn_avg

            pruning_info = {
                'h_real': h_real,
                'h_fake': h_fake,
                'h_fake_for_adv': h_fake_for_adv,  # 用于 adv_loss（可选 detach mask）
                'hard_mask': hard_mask_full,  # (batch, n_vision_orig) - 相对于原始位置
                'hard_mask_current': hard_mask,  # (batch, n_vision) - 当前层的 mask
                'q2v_attn': q2v_attn_full,  # (batch, n_vision_orig)
                'keep_logits': pruner_info.get('keep_logits'),
                'attn_score': pruner_info.get('attn_score'),
                'token_score': pruner_info.get('token_score'),
                'baseline': pruner_info.get('baseline'),
                'delta': pruner_info.get('delta'),
                'gumbel_debug': pruner_info.get('gumbel_debug'),  # Gumbel noise debug info
                'layer_idx': self.layer_idx,
                'n_vision': n_vision,  # 当前层的 vision token 数量
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
