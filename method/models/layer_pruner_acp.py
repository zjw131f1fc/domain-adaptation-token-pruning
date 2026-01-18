"""Attention Consistency Pruning - Layer Pruner

轻量级剪枝网络，输出对 LLM attention 的残差调整。

设计理念：
- 输入是 LLM 自己的 question→vision attention（已经是很好的 baseline）
- 输出是残差调整（可正可负）
- 初始化为零，初始时完全依赖 LLM attention
- Gumbel-Softmax 实现可微的 0/1 决策
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerPruner(nn.Module):
    """单层 Vision Token 剪枝器

    输入 LLM 的 question→vision attention，输出对其的残差调整，
    然后通过 Gumbel-Softmax 生成 0/1 hard mask。

    参数:
        d_internal: MLP 内部维度
        temperature: Gumbel-Softmax 温度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        d_internal: int = 128,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_internal = d_internal
        self.temperature = temperature

        # MLP: 输入 attention 值，输出残差
        # 输入维度为 1（单个 attention 值）
        self.mlp = nn.Sequential(
            nn.Linear(1, d_internal),
            nn.LayerNorm(d_internal),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_internal, 1)
        )

        # 初始化最后一层为零，初始时输出残差为 0
        # 这样初始行为完全由 LLM 的 attention 决定
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        q2v_attn: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """计算残差调整

        参数:
            q2v_attn: (batch, n_vision) - LLM 的 question→vision attention（归一化后）
            return_components: 是否返回中间结果

        返回:
            residual: (batch, n_vision) - 残差调整
            或 (residual, components_dict) 如果 return_components=True
        """
        # (batch, n_vision) -> (batch, n_vision, 1)
        x = q2v_attn.unsqueeze(-1)

        # MLP
        residual = self.mlp(x).squeeze(-1)  # (batch, n_vision)

        if return_components:
            return residual, {'input_attn': q2v_attn, 'residual': residual}
        return residual

    def compute_importance(
        self,
        q2v_attn: torch.Tensor,
        return_components: bool = False
    ):
        """计算最终的重要性分数

        参数:
            q2v_attn: (batch, n_vision) - LLM 的 question→vision attention
            return_components: 是否返回中间结果

        返回:
            importance: (batch, n_vision) - 最终重要性分数
        """
        residual = self.forward(q2v_attn)
        importance = q2v_attn + residual

        if return_components:
            return importance, {
                'q2v_attn': q2v_attn,
                'residual': residual,
                'importance': importance
            }
        return importance

    def gumbel_softmax_mask(
        self,
        importance: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """将 importance score 转换为 0/1 hard mask

        使用 Gumbel-Softmax with hard=True 实现可微的离散决策。

        参数:
            importance: (batch, n_vision) - 重要性分数
            temperature: 可选的温度覆盖

        返回:
            hard_mask: (batch, n_vision) - 0/1 mask
        """
        temp = temperature if temperature is not None else self.temperature

        # 转换为二分类 logits: [drop_logit, keep_logit]
        # drop_logit 固定为 0，keep_logit 为 importance
        stacked = torch.stack([
            torch.zeros_like(importance),  # drop logit = 0
            importance                      # keep logit
        ], dim=-1)  # (batch, n_vision, 2)

        if self.training:
            # 训练模式：Gumbel-Softmax with hard=True
            # hard=True: 前向传播用 one-hot，反向传播用软梯度
            y = F.gumbel_softmax(stacked, tau=temp, hard=True, dim=-1)
            hard_mask = y[..., 1]  # 取 keep 的概率/决策
        else:
            # 推理模式：直接 argmax（importance > 0 则保留）
            hard_mask = (importance > 0).float()

        return hard_mask

    def forward_full(
        self,
        q2v_attn: torch.Tensor,
        temperature: Optional[float] = None
    ):
        """完整的前向传播：从 attention 到 hard mask

        参数:
            q2v_attn: (batch, n_vision) - LLM 的 question→vision attention
            temperature: 可选的温度覆盖

        返回:
            hard_mask: (batch, n_vision) - 0/1 mask
            info: dict - 中间结果
        """
        residual = self.forward(q2v_attn)
        importance = q2v_attn + residual
        hard_mask = self.gumbel_softmax_mask(importance, temperature)

        return hard_mask, {
            'q2v_attn': q2v_attn,
            'residual': residual,
            'importance': importance,
            'hard_mask': hard_mask
        }

    def set_temperature(self, temperature: float):
        """设置 Gumbel-Softmax 温度"""
        self.temperature = temperature


class LayerPrunerManager(nn.Module):
    """多层剪枝器管理器

    管理多个层的 LayerPruner，提供统一接口。

    参数:
        layer_indices: 要剪枝的层索引列表
        d_internal: MLP 内部维度
        temperature: 初始温度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        layer_indices: list = [4, 14, 24],
        d_internal: int = 128,
        temperature: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layer_indices = layer_indices

        # 为每层创建独立的 pruner
        self.pruners = nn.ModuleDict({
            str(idx): LayerPruner(
                d_internal=d_internal,
                temperature=temperature,
                dropout=dropout
            )
            for idx in layer_indices
        })

    def get_pruner(self, layer_idx: int) -> LayerPruner:
        """获取指定层的剪枝器"""
        key = str(layer_idx)
        if key not in self.pruners:
            raise ValueError(f"No pruner for layer {layer_idx}. Available: {self.layer_indices}")
        return self.pruners[key]

    def set_temperature(self, temperature: float):
        """设置所有剪枝器的温度"""
        for pruner in self.pruners.values():
            pruner.set_temperature(temperature)

    def get_all_layers(self) -> list:
        """返回所有剪枝层的索引"""
        return self.layer_indices
