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
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


class CrossAttentionPruner(nn.Module):
    """基于 Cross-Attention 的 Vision Token 剪枝器

    使用可学习的 pruning queries 通过 cross-attention 评估 vision tokens 重要性。

    参数:
        d_model: 输入 hidden states 的维度
        d_internal: 内部特征维度
        n_heads: Cross-attention 头数
        temperature: Gumbel-Softmax 温度
        dropout: Dropout 比例
        threshold: 推理时的剪枝阈值（概率空间，0-1）
                   保留概率 > threshold 时保留 token
                   默认 0.5 等价于训练时的行为
    """

    def __init__(
        self,
        d_model: int,
        d_internal: int = 128,
        n_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
        threshold: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.d_internal = d_internal
        self.n_heads = n_heads
        self.temperature = temperature
        self.threshold = threshold

        # 推理模式配置
        self.inference_mode = 'threshold'  # 'threshold' 或 'topk'
        self.topk_k = None  # topk 模式下保留的 token 数量

        # 可学习的 pruning queries (多个 query 学习不同的重要性模式)
        self.n_queries = 4
        self.pruning_queries = nn.Parameter(torch.randn(1, self.n_queries, d_internal) * 0.02)

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

        # 可学习偏置，初始化为 0 使初始保留率接近 50%
        # sigmoid(0) = 0.5，softmax([0, 0])[1] = 0.5
        self.keep_bias = nn.Parameter(torch.tensor(0.0))

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # Vision projection: 小权重初始化
        nn.init.xavier_uniform_(self.vision_proj.weight, gain=0.1)
        nn.init.zeros_(self.vision_proj.bias)

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
        key_padding_mask: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """计算 keep logits

        残差设计：以 q2v_attn 作为 baseline，pruner 学习 delta
        keep_logits = baseline + delta + bias

        参数:
            vision_hidden: (batch, n_vision, d_model) - vision token hidden states
            q2v_attn: (batch, n_vision) - LLM 的 question→vision attention 权重（作为 baseline）
            key_padding_mask: (batch, n_vision) - True 表示该位置被 mask（不参与计算）
            return_components: 是否返回中间结果

        返回:
            keep_logits: (batch, n_vision) - 保留 logits（越大越倾向保留）
        """
        batch_size, n_vision, _ = vision_hidden.shape

        # === Baseline: 基于 LLM attention 的初始分数 ===
        if q2v_attn is not None:
            # 将 attention 转换为 logit 空间（log 变换 + 中心化）
            baseline_raw = torch.log(q2v_attn.clamp(min=1e-6))
            # 中心化：只对 kept 位置计算 mean（避免 masked 位置的 -inf 影响）
            if key_padding_mask is not None:
                kept_mask = (~key_padding_mask).to(baseline_raw.dtype)  # 转换为与 baseline 相同的 dtype
                n_kept = kept_mask.sum(dim=-1, keepdim=True).clamp(min=1)
                baseline_mean = (baseline_raw * kept_mask).sum(dim=-1, keepdim=True) / n_kept
            else:
                baseline_mean = baseline_raw.mean(dim=-1, keepdim=True)
            baseline = baseline_raw - baseline_mean

            # DEBUG: 打印 baseline 计算细节
            print(f"\n[Baseline DEBUG]")
            print(f"  q2v_attn - shape: {q2v_attn.shape}, sum: {q2v_attn.sum().item():.6f}")
            if key_padding_mask is not None:
                kept_indices = (~key_padding_mask[0]).nonzero(as_tuple=True)[0]
                q2v_kept = q2v_attn[0, kept_indices]
                baseline_raw_kept = baseline_raw[0, kept_indices]
                print(f"  q2v_attn[kept] - count: {len(kept_indices)}, mean: {q2v_kept.mean().item():.6f}, min: {q2v_kept.min().item():.6f}, max: {q2v_kept.max().item():.6f}")
                print(f"  baseline_raw[kept] - mean: {baseline_raw_kept.mean().item():.4f}, min: {baseline_raw_kept.min().item():.4f}, max: {baseline_raw_kept.max().item():.4f}")
            else:
                print(f"  q2v_attn - mean: {q2v_attn.mean().item():.6f}, min: {q2v_attn.min().item():.6f}, max: {q2v_attn.max().item():.6f}")
                print(f"  baseline_raw - mean: {baseline_raw.mean().item():.4f}, min: {baseline_raw.min().item():.4f}, max: {baseline_raw.max().item():.4f}")
            print(f"  baseline_mean: {baseline_mean.item():.4f}")
        else:
            baseline = torch.zeros(batch_size, n_vision, device=vision_hidden.device)

        # === Delta: Pruner 学习的修正量 ===
        # 1. Project vision tokens
        v = self.vision_proj(vision_hidden)  # (batch, n_vision, d_internal)

        # 2. Expand pruning queries
        queries = self.pruning_queries.expand(batch_size, -1, -1)  # (batch, n_queries, d_internal)

        # 3. Cross-attention: queries attend to vision tokens
        # attn_weights: (batch, n_queries, n_vision)
        _, attn_weights = self.cross_attn(
            query=queries,
            key=v,
            value=v,
            key_padding_mask=key_padding_mask,  # 传入 mask
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
        # - attn_score: 基于 cross-attention 的全局重要性修正
        # - token_score: 基于 token 自身特征的重要性修正
        delta = attn_score + token_score

        # === 残差连接: keep_logits = baseline + delta + bias ===
        keep_logits = baseline + delta + self.keep_bias

        # DEBUG: 打印 pruner 内部各组件的统计信息
        print(f"\n[Pruner Internal DEBUG]")
        print(f"  n_vision: {n_vision}")
        print(f"  baseline - mean: {baseline.mean().item():.4f}, min: {baseline.min().item():.4f}, max: {baseline.max().item():.4f}")
        print(f"  attn_score - mean: {attn_score.mean().item():.4f}, min: {attn_score.min().item():.4f}, max: {attn_score.max().item():.4f}")
        print(f"  token_score - mean: {token_score.mean().item():.4f}, min: {token_score.min().item():.4f}, max: {token_score.max().item():.4f}")
        print(f"  delta - mean: {delta.mean().item():.4f}, min: {delta.min().item():.4f}, max: {delta.max().item():.4f}")
        print(f"  keep_bias: {self.keep_bias.item():.4f}")
        print(f"  keep_logits - mean: {keep_logits.mean().item():.4f}, min: {keep_logits.min().item():.4f}, max: {keep_logits.max().item():.4f}")
        if key_padding_mask is not None:
            n_masked = key_padding_mask.sum().item()
            n_kept = n_vision - n_masked
            print(f"  key_padding_mask: {n_masked} masked, {n_kept} kept")
            # 打印被 mask 位置和保留位置的分别统计
            kept_mask = ~key_padding_mask  # True = kept
            if n_kept > 0:
                print(f"  baseline[kept] - mean: {baseline[kept_mask].mean().item():.4f}")
                print(f"  token_score[kept] - mean: {token_score[kept_mask].mean().item():.4f}")
                print(f"  keep_logits[kept] - mean: {keep_logits[kept_mask].mean().item():.4f}")
            if n_masked > 0:
                print(f"  baseline[masked] - mean: {baseline[key_padding_mask].mean().item():.4f}")
                print(f"  token_score[masked] - mean: {token_score[key_padding_mask].mean().item():.4f}")
                print(f"  keep_logits[masked] - mean: {keep_logits[key_padding_mask].mean().item():.4f}")

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

    def gumbel_softmax_mask(
        self,
        keep_logits: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """将 keep logits 转换为 0/1 hard mask

        使用 Gumbel-Softmax with hard=True 实现可微的离散决策。

        参数:
            keep_logits: (batch, n_vision) - 保留 logits
            temperature: 可选的温度覆盖

        返回:
            hard_mask: (batch, n_vision) - 0/1 mask，dtype 与输入一致
        """
        temp = temperature if temperature is not None else self.temperature
        input_dtype = keep_logits.dtype

        # 构建二分类 logits: [drop_logit, keep_logit]
        drop_logits = torch.zeros_like(keep_logits)
        stacked = torch.stack([drop_logits, keep_logits], dim=-1)  # (batch, n_vision, 2)

        if self.training:
            # 训练模式：Gumbel-Softmax with hard=True
            y = F.gumbel_softmax(stacked, tau=temp, hard=True, dim=-1)
            hard_mask = y[..., 1]  # 取 keep 的决策
        else:
            # 推理模式：根据 inference_mode 选择
            if self.inference_mode == 'topk' and self.topk_k is not None:
                # Top-k 模式：保留 sigmoid 最高的 k 个 token
                hard_mask = self._topk_mask(keep_logits, self.topk_k)
            else:
                # 阈值模式：sigmoid(keep_logits) > threshold
                keep_prob = torch.sigmoid(keep_logits)
                hard_mask = (keep_prob > self.threshold).to(input_dtype)

        return hard_mask.to(input_dtype)

    def _topk_mask(self, keep_logits: torch.Tensor, k: int) -> torch.Tensor:
        """生成 top-k mask

        参数:
            keep_logits: (batch, n_vision) - 保留 logits
            k: 保留的 token 数量

        返回:
            hard_mask: (batch, n_vision) - 0/1 mask，dtype 与输入一致
        """
        batch_size, n_vision = keep_logits.shape
        k = min(k, n_vision)  # 确保 k 不超过 token 数量

        keep_prob = torch.sigmoid(keep_logits)

        # 找到 top-k 的阈值
        topk_values, _ = torch.topk(keep_prob, k, dim=-1)
        threshold = topk_values[:, -1:]  # 第 k 大的值，形状 (batch, 1)

        # 生成 mask（>= threshold 的保留），保持输入 dtype
        hard_mask = (keep_prob >= threshold).to(keep_logits.dtype)

        return hard_mask

    def set_inference_mode(self, mode: str):
        """设置推理模式

        参数:
            mode: 'threshold' 或 'topk'
        """
        assert mode in ('threshold', 'topk'), f"Unknown inference mode: {mode}"
        self.inference_mode = mode

    def set_topk_k(self, k: int):
        """设置 topk 模式下保留的 token 数量"""
        self.topk_k = k

    def forward_full(
        self,
        vision_hidden: torch.Tensor,
        q2v_attn: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        temperature: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """完整的前向传播：从 hidden states 到 hard mask

        参数:
            vision_hidden: (batch, n_vision, d_model) - vision token hidden states
            q2v_attn: (batch, n_vision) - 可选的 LLM attention 权重
            key_padding_mask: (batch, n_vision) - True 表示该位置被 mask（不参与计算）
            temperature: 可选的温度覆盖

        返回:
            hard_mask: (batch, n_vision) - 0/1 mask
            info: dict - 中间结果
        """
        keep_logits, components = self.forward(vision_hidden, q2v_attn, key_padding_mask, return_components=True)
        hard_mask = self.gumbel_softmax_mask(keep_logits, temperature)

        return hard_mask, {
            **components,
            'hard_mask': hard_mask,
        }

    def set_temperature(self, temperature: float):
        """设置 Gumbel-Softmax 温度"""
        self.temperature = temperature

    def set_threshold(self, threshold: float):
        """设置推理时的剪枝阈值"""
        self.threshold = threshold


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
        thresholds: 每层的推理阈值字典 {layer_idx: threshold}
    """

    def __init__(
        self,
        layer_indices: list,
        d_model: int,
        d_internal: int = 128,
        n_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
        thresholds: Optional[Dict[int, float]] = None
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.d_model = d_model

        # 默认阈值为 0
        if thresholds is None:
            thresholds = {}

        # 为每层创建独立的 pruner
        self.pruners = nn.ModuleDict({
            str(idx): CrossAttentionPruner(
                d_model=d_model,
                d_internal=d_internal,
                n_heads=n_heads,
                temperature=temperature,
                dropout=dropout,
                threshold=thresholds.get(idx, 0.5)
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

    def set_threshold(self, layer_idx: int, threshold: float):
        """设置指定层的推理阈值"""
        self.get_pruner(layer_idx).set_threshold(threshold)

    def set_thresholds(self, thresholds: Dict[int, float]):
        """设置多层的推理阈值"""
        for layer_idx, threshold in thresholds.items():
            if layer_idx in self.layer_indices:
                self.set_threshold(layer_idx, threshold)

    def get_all_layers(self) -> list:
        """返回所有剪枝层的索引"""
        return self.layer_indices

    def set_inference_mode(self, mode: str):
        """设置所有剪枝器的推理模式

        参数:
            mode: 'threshold' 或 'topk'
        """
        for pruner in self.pruners.values():
            pruner.set_inference_mode(mode)

    def set_topk_k(self, layer_idx: int, k: int):
        """设置指定层的 topk k 值"""
        self.get_pruner(layer_idx).set_topk_k(k)

    def set_topk_ks(self, topk_ks: Dict[int, int]):
        """设置多层的 topk k 值

        参数:
            topk_ks: {layer_idx: k} 字典
        """
        for layer_idx, k in topk_ks.items():
            if layer_idx in self.layer_indices:
                self.set_topk_k(layer_idx, k)


# 保持向后兼容的别名
LayerPruner = CrossAttentionPruner
