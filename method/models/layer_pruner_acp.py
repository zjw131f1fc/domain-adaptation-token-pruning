"""Attention Consistency Pruning - Cross-Attention Layer Pruner

基于 Cross-Attention 的剪枝器设计。

设计理念：
- 使用可学习的 "pruning queries" 来评估每个 vision token 的重要性
- Cross-attention 让模型学习哪些 token 对回答问题最重要
- 全局上下文：每个 token 的决策考虑其他 token 的信息
- 可学习偏置：初始化为负值以鼓励剪枝
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class CrossAttentionPruner(nn.Module):
    """基于 Cross-Attention 的 Vision Token 剪枝器

    使用可学习的 pruning queries 通过 cross-attention 评估 vision tokens 重要性。

    参数:
        d_model: 输入 hidden states 的维度
        d_internal: 内部特征维度
        n_heads: Cross-attention 头数
        temperature: Gumbel-Sigmoid 温度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        d_model: int,
        d_internal: int = 128,
        n_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
        use_gumbel_noise: bool = True,  # 是否使用 Gumbel noise
        pruning_threshold: float = 0.5,  # sigmoid 后的阈值，用于训练第三阶段和推理
        use_question_condition: bool = False,  # 是否使用 question embedding 条件化
    ):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.n_heads = n_heads
        self.temperature = temperature
        self.use_gumbel_noise = use_gumbel_noise
        self.pruning_threshold = pruning_threshold
        self.use_question_condition = use_question_condition

        # 可学习的 pruning queries (多个 query 学习不同的重要性模式)
        self.n_queries = 4
        self.pruning_queries = nn.Parameter(torch.randn(1, self.n_queries, d_internal) * 0.02)

        # Question embedding projection (用于条件化 pruning queries)
        if use_question_condition:
            self.question_proj = nn.Linear(d_model, d_internal)
        else:
            self.question_proj = None

        # Vision token projection
        self.vision_proj = nn.Linear(d_model, d_internal)

        # Cross-attention: pruning queries attend to vision tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_internal,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Per-token scoring head (补充 attention-based 分数)
        self.token_scorer = nn.Sequential(
            nn.Linear(d_internal, d_internal),
            nn.LayerNorm(d_internal),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_internal, 1)
        )

        # Query aggregation: 将多个 query 的注意力聚合为单一分数
        self.query_aggregator = nn.Linear(self.n_queries, 1)

        # 可学习偏置，初始化为正数使初始保留率较高
        # keep_bias=2.0 时，初始 logits 偏向正值，保留更多 token
        self.keep_bias = nn.Parameter(torch.tensor(2.0))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # Vision projection: 小权重初始化
        nn.init.xavier_uniform_(self.vision_proj.weight, gain=0.1)
        nn.init.zeros_(self.vision_proj.bias)

        # Question projection: 小权重初始化
        if self.question_proj is not None:
            nn.init.xavier_uniform_(self.question_proj.weight, gain=0.1)
            nn.init.zeros_(self.question_proj.bias)

        # Token scorer 最后一层零初始化，让初始输出接近 0
        nn.init.zeros_(self.token_scorer[-1].weight)
        nn.init.zeros_(self.token_scorer[-1].bias)

        # Query aggregator 初始化为均匀聚合
        nn.init.constant_(self.query_aggregator.weight, 1.0 / self.n_queries)
        nn.init.zeros_(self.query_aggregator.bias)

    def forward(
        self,
        vision_hidden: torch.Tensor,
        q2v_attn: Optional[torch.Tensor] = None,
        cumulative_vision_mask: Optional[torch.Tensor] = None,
        n_pruned_tokens: int = 0,
        question_hidden: Optional[torch.Tensor] = None,
        question_lengths: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """计算 keep logits

        残差设计：以 q2v_attn 作为 baseline，pruner 学习 delta
        keep_logits = baseline + delta + bias

        参数:
            vision_hidden: (batch, n_vision, d_model) - vision token hidden states
            q2v_attn: (batch, n_vision) - LLM 的 question→vision attention 权重（作为 baseline）
            cumulative_vision_mask: (batch, n_vision) - 累积 mask，1=保留，0=已被剪掉
            n_pruned_tokens: 已被剪掉的 tokens 数量（推理时物理删除后使用，用于修正 baseline_mean）
            question_hidden: (batch, max_q_len, d_model) - question tokens 的 hidden states（可能有 padding）
            question_lengths: (batch,) - 每个样本的 question 实际长度（用于 masked mean）
            return_components: 是否返回中间结果

        返回:
            keep_logits: (batch, n_vision) - 保留 logits（越大越倾向保留）
        """
        batch_size, n_vision, _ = vision_hidden.shape

        # === Baseline: 基于 LLM attention 的初始分数 ===
        if q2v_attn is not None:
            # 将 attention 转换为 logit 空间（log 变换 + 中心化）
            baseline_raw = torch.log(q2v_attn.clamp(min=1e-6))

            # 计算 baseline_mean：只考虑当前保留的位置
            if cumulative_vision_mask is not None:
                # 只对 mask 为 1 的位置计算 mean
                masked_baseline = baseline_raw * cumulative_vision_mask
                baseline_sum = masked_baseline.sum(dim=-1, keepdim=True)
                mask_count = cumulative_vision_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                baseline_mean = baseline_sum / mask_count
            elif n_pruned_tokens > 0:
                # 推理时物理删除后，需要修正 baseline_mean（旧逻辑，保留兼容）
                import math
                pruned_baseline = math.log(1e-6)  # ≈ -13.8
                total_sum = baseline_raw.sum(dim=-1, keepdim=True) + n_pruned_tokens * pruned_baseline
                total_count = n_vision + n_pruned_tokens
                baseline_mean = total_sum / total_count
            else:
                baseline_mean = baseline_raw.mean(dim=-1, keepdim=True)

            baseline = baseline_raw - baseline_mean
        else:
            baseline = torch.zeros(batch_size, n_vision, device=vision_hidden.device)

        # === Delta: Pruner 学习的修正量 ===
        # 1. Project vision tokens
        v = self.vision_proj(vision_hidden)  # (batch, n_vision, d_internal)

        # 2. Expand pruning queries 并融入 question embedding
        queries = self.pruning_queries.expand(batch_size, -1, -1)  # (batch, n_queries, d_internal)
        if self.use_question_condition and question_hidden is not None:
            # Masked mean：只对有效位置求均值，避免 padding 稀释
            if question_lengths is not None:
                # question_lengths: (batch,) - 每个样本的实际长度
                # 创建 mask: (batch, max_q_len)
                max_q_len = question_hidden.shape[1]
                mask = torch.arange(max_q_len, device=question_hidden.device).unsqueeze(0) < question_lengths.unsqueeze(1)
                mask = mask.unsqueeze(-1).float()  # (batch, max_q_len, 1)
                # Masked sum / count
                question_sum = (question_hidden * mask).sum(dim=1)  # (batch, d_model)
                question_emb = question_sum / question_lengths.unsqueeze(-1).clamp(min=1)  # (batch, d_model)
            else:
                # Fallback: 普通均值（假设无 padding）
                question_emb = question_hidden.mean(dim=1)  # (batch, d_model)
            question_proj = self.question_proj(question_emb)  # (batch, d_internal)
            # 加到 pruning queries 上作为 condition
            queries = queries + question_proj.unsqueeze(1)  # (batch, n_queries, d_internal)

        # 3. Cross-attention: queries attend to vision tokens
        # attn_weights: (batch, n_queries, n_vision)
        # 构建 key_padding_mask 屏蔽已被剪掉的 tokens
        if cumulative_vision_mask is not None:
            # key_padding_mask: (batch, n_vision), True = 忽略该位置
            key_padding_mask = (cumulative_vision_mask < 0.5)
        else:
            key_padding_mask = None

        _, attn_weights = self.cross_attn(
            query=queries,
            key=v,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True  # 对 heads 取平均
        )

        # 4. Aggregate attention weights from multiple queries
        # (batch, n_queries, n_vision) -> (batch, n_vision, n_queries) -> (batch, n_vision, 1)
        attn_weights_t = attn_weights.transpose(1, 2)  # (batch, n_vision, n_queries)
        attn_score = self.query_aggregator(attn_weights_t).squeeze(-1)  # (batch, n_vision)

        # 5. Per-token score
        token_score = self.token_scorer(v).squeeze(-1)  # (batch, n_vision)

        # 6. Delta = attn_score + token_score
        delta = attn_score + token_score

        # === 残差连接: keep_logits = baseline + delta + bias ===
        keep_logits = baseline + delta + self.keep_bias

        if return_components:
            return keep_logits, {
                'baseline': baseline,
                'delta': delta,
                'attn_score': attn_score,
                'token_score': token_score,
                'attn_weights': attn_weights,
                'keep_logits': keep_logits
            }

        return keep_logits

    def gumbel_sigmoid_mask(
        self,
        keep_logits: torch.Tensor,
        temperature: Optional[float] = None,
        return_debug: bool = False
    ) -> torch.Tensor:
        """将 keep logits 转换为 0/1 hard mask

        两种模式：
        1. use_gumbel_noise=True (Gumbel-Sigmoid):
           - 训练时: sigmoid((x + logistic_noise) / tau) > 0.5
           - 有随机性，但训练和推理可能不一致
        2. use_gumbel_noise=False (纯 STE):
           - 训练时: sigmoid(x / tau) > 0.5，即 x > 0
           - 无随机性，训练和推理完全一致

        参数:
            keep_logits: (batch, n_vision) - 保留 logits
            temperature: 可选的温度覆盖
            return_debug: 是否返回 debug 信息

        返回:
            hard_mask: (batch, n_vision) - 0/1 mask，dtype 与输入一致
            debug_info: dict (仅当 return_debug=True)
        """
        temp = temperature if temperature is not None else self.temperature
        input_dtype = keep_logits.dtype

        # 在 float32 下计算以避免 bfloat16 精度问题
        keep_logits_f32 = keep_logits.float()

        debug_info = {}

        if self.training:
            if self.use_gumbel_noise:
                # Gumbel-Sigmoid 模式：加 Logistic noise
                u = torch.rand_like(keep_logits_f32).clamp(1e-8, 1 - 1e-8)
                logistic_noise = torch.log(u) - torch.log(1 - u)
                noisy_logits = keep_logits_f32 + logistic_noise
            else:
                # 纯 STE 模式：不加 noise
                logistic_noise = torch.zeros_like(keep_logits_f32)
                noisy_logits = keep_logits_f32

            # sigmoid + hard 决策 + STE
            y_soft = torch.sigmoid(noisy_logits / temp)
            y_hard = (y_soft > self.pruning_threshold).float()
            hard_mask = y_hard - y_soft.detach() + y_soft

            if return_debug:
                debug_info = {
                    'use_gumbel_noise': self.use_gumbel_noise,
                    'logistic_noise_mean': logistic_noise.mean().item(),
                    'logistic_noise_std': logistic_noise.std().item(),
                    'noisy_logits_mean': noisy_logits.mean().item(),
                    'noisy_logits_std': noisy_logits.std().item(),
                    'y_soft_mean': y_soft.mean().item(),
                    'temperature': temp,
                    'pruning_threshold': self.pruning_threshold,
                }
        else:
            # 推理模式：sigmoid(x / temp) > threshold
            # 注意：必须除以温度，与训练时保持一致
            y_soft = torch.sigmoid(keep_logits_f32 / temp)
            hard_mask = (y_soft > self.pruning_threshold).float()

        if return_debug:
            return hard_mask.to(input_dtype), debug_info
        return hard_mask.to(input_dtype)

    def forward_full(
        self,
        vision_hidden: torch.Tensor,
        q2v_attn: Optional[torch.Tensor] = None,
        cumulative_vision_mask: Optional[torch.Tensor] = None,
        n_pruned_tokens: int = 0,
        question_hidden: Optional[torch.Tensor] = None,
        question_lengths: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None,
        return_debug: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """完整的前向传播：从 hidden states 到 hard mask

        参数:
            vision_hidden: (batch, n_vision, d_model) - vision token hidden states
            q2v_attn: (batch, n_vision) - 可选的 LLM attention 权重
            cumulative_vision_mask: (batch, n_vision) - 累积 mask，1=保留，0=已被剪掉
            n_pruned_tokens: 已被剪掉的 tokens 数量（推理时物理删除后使用）
            question_hidden: (batch, max_q_len, d_model) - question tokens 的 hidden states
            question_lengths: (batch,) - 每个样本的 question 实际长度
            temperature: 可选的温度覆盖
            return_debug: 是否返回 debug 信息

        返回:
            hard_mask: (batch, n_vision) - 0/1 mask
            info: dict - 中间结果
        """
        keep_logits, components = self.forward(
            vision_hidden, q2v_attn,
            cumulative_vision_mask=cumulative_vision_mask,
            n_pruned_tokens=n_pruned_tokens,
            question_hidden=question_hidden,
            question_lengths=question_lengths,
            return_components=True
        )

        if return_debug:
            hard_mask, debug_info = self.gumbel_sigmoid_mask(keep_logits, temperature, return_debug=True)
            return hard_mask, {
                **components,
                'hard_mask': hard_mask,
                'gumbel_debug': debug_info,
            }
        else:
            hard_mask = self.gumbel_sigmoid_mask(keep_logits, temperature)
            return hard_mask, {
                **components,
                'hard_mask': hard_mask,
            }

    def set_temperature(self, temperature: float):
        """设置 Gumbel-Sigmoid 温度"""
        self.temperature = temperature

    def set_use_gumbel_noise(self, use_gumbel_noise: bool):
        """设置是否使用 Gumbel noise"""
        self.use_gumbel_noise = use_gumbel_noise

    def set_pruning_threshold(self, threshold: float):
        """设置 sigmoid 后的剪枝阈值"""
        self.pruning_threshold = threshold


class LayerPrunerManager(nn.Module):
    """多层剪枝器管理器

    管理多个层的 CrossAttentionPruner，提供统一接口。

    参数:
        layer_indices: 要剪枝的层索引列表
        d_model: 输入 hidden states 的维度
        d_internal: 内部特征维度
        n_heads: Cross-attention 头数
        temperature: 初始温度
        dropout: Dropout 比例
        use_gumbel_noise: 是否使用 Gumbel noise（False 则使用纯 STE）
        use_question_condition: 是否使用 question embedding 条件化
    """

    def __init__(
        self,
        layer_indices: list,
        d_model: int,
        d_internal: int = 128,
        n_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
        use_gumbel_noise: bool = True,
        pruning_threshold: float = 0.5,
        use_question_condition: bool = False,
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.d_model = d_model
        self.use_question_condition = use_question_condition

        # 为每层创建独立的 pruner
        self.pruners = nn.ModuleDict({
            str(idx): CrossAttentionPruner(
                d_model=d_model,
                d_internal=d_internal,
                n_heads=n_heads,
                temperature=temperature,
                dropout=dropout,
                use_gumbel_noise=use_gumbel_noise,
                pruning_threshold=pruning_threshold,
                use_question_condition=use_question_condition,
            )
            for idx in layer_indices
        })

    def get_pruner(self, layer_idx: int) -> CrossAttentionPruner:
        """获取指定层的剪枝器"""
        key = str(layer_idx)
        if key not in self.pruners:
            raise ValueError(f"No pruner for layer {layer_idx}. Available: {self.layer_indices}")
        return self.pruners[key]

    def set_temperature(self, temperature: float):
        """设置所有剪枝器的温度"""
        for pruner in self.pruners.values():
            pruner.set_temperature(temperature)

    def set_use_gumbel_noise(self, use_gumbel_noise: bool):
        """设置所有剪枝器是否使用 Gumbel noise"""
        for pruner in self.pruners.values():
            pruner.set_use_gumbel_noise(use_gumbel_noise)

    def set_pruning_threshold(self, threshold: float):
        """设置所有剪枝器的 sigmoid 阈值"""
        for pruner in self.pruners.values():
            pruner.set_pruning_threshold(threshold)

    def get_all_layers(self) -> list:
        """返回所有剪枝层的索引"""
        return self.layer_indices


# 保持向后兼容的别名
LayerPruner = CrossAttentionPruner
