"""Adapter 模块 - 补偿剪枝后的信息损失"""

import torch
import torch.nn as nn
from typing import Optional

# 全局调试计数器
_DEBUG_COUNTER = {"train": 0, "eval": 0, "eval_decode": 0}

def reset_adapter_debug_counter():
    """重置调试计数器"""
    global _DEBUG_COUNTER
    _DEBUG_COUNTER = {"train": 0, "eval": 0, "eval_decode": 0}


class LightweightAdapter(nn.Module):
    """轻量级 Adapter：Mask-Aware + Query-Aware FiLM 调制

    用 pruning mask + 当前 token 的 attention query 进行补偿。
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 512,
        n_vision: int = 576,
        dropout: float = 0.15,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # Mask encoder: 编码哪些 token 被剪掉
        self.mask_encoder = nn.Sequential(
            nn.Linear(n_vision, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, bottleneck_dim)
        )

        # Query encoder: 投影 attention query 到 bottleneck
        self.query_proj = nn.Linear(hidden_size, bottleneck_dim)

        # FiLM: 根据 (mask + query) 生成调制参数
        # 输入是 mask_emb + query_emb，都是 bottleneck 维度
        self.gamma_net = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.beta_net = nn.Linear(bottleneck_dim, bottleneck_dim)

        # 主干
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        self.act = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        # Query proj 小值初始化
        nn.init.xavier_uniform_(self.query_proj.weight, gain=0.1)
        nn.init.zeros_(self.query_proj.bias)

        # FiLM 初始化：gamma=1, beta=0
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

        # Up projection 初始化为小值
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
        debug: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        参数:
            x: (batch, seq, hidden_size) - attention output
            mask: (batch, n_vision) - pruning mask (1=keep, 0=prune)
            query: (batch, seq, hidden_size) - attention query states
            debug: 是否打印调试信息
        """
        # 调试：打印训练/推理时的输入差异
        mode = "train" if self.training else "eval"
        # 区分 prefill 和 decode 阶段
        seq_len = x.shape[1]
        if not self.training and seq_len == 1:
            mode = "eval_decode"
        _DEBUG_COUNTER[mode] += 1
        if _DEBUG_COUNTER[mode] <= 3:  # 只打印前3次
            print(f"\n[Adapter DEBUG - {mode} #{_DEBUG_COUNTER[mode]}]")
            print(f"  x.shape: {x.shape}")
            print(f"  x.mean: {x.mean().item():.6f}, x.std: {x.std().item():.6f}")
            print(f"  mask: {mask.shape if mask is not None else None}")
            if mask is not None:
                print(f"  mask.sum: {mask.sum().item()}, mask.mean: {mask.mean().item():.4f}")
            print(f"  query: {query.shape if query is not None else None}")
            if query is not None:
                print(f"  query.mean: {query.mean().item():.6f}, query.std: {query.std().item():.6f}")
                print(f"  query[:, 0, :5]: {query[0, 0, :5].tolist()}")  # 打印部分值

        if debug:
            print(f"[Adapter] x.shape: {x.shape}")
            print(f"[Adapter] mask: {mask.shape if mask is not None else None}, sum: {mask.sum().item() if mask is not None else None}")
            print(f"[Adapter] query: {query.shape if query is not None else None}")
            print(f"[Adapter] training: {self.training}")

        h = self.dropout(self.act(self.down(x)))  # (batch, seq, bottleneck)

        # 构建 condition
        condition = torch.zeros_like(h)  # (batch, seq, bottleneck)

        if mask is not None:
            mask_emb = self.mask_encoder(mask.to(dtype=x.dtype))  # (batch, bottleneck)
            condition = condition + mask_emb.unsqueeze(1)  # broadcast to (batch, seq, bottleneck)

        if query is not None:
            query_emb = self.query_proj(query)  # (batch, seq, bottleneck)
            condition = condition + query_emb
            # 调试：打印 query_emb 的统计信息
            if _DEBUG_COUNTER[mode] <= 3:
                print(f"  query_emb.mean: {query_emb.mean().item():.6f}, query_emb.std: {query_emb.std().item():.6f}")
                print(f"  condition.mean: {condition.mean().item():.6f}, condition.std: {condition.std().item():.6f}")

        # FiLM modulation
        gamma = 1 + self.gamma_net(condition)  # (batch, seq, bottleneck)
        beta = self.beta_net(condition)
        h = gamma * h + beta
        h = self.dropout(h)  # Dropout after FiLM

        return x + self.up(h)


class AdapterManager(nn.Module):
    """多层 Adapter 管理器"""

    def __init__(
        self,
        layer_indices: list,
        hidden_size: int,
        bottleneck_dim: int = 512,
        adapter_type: str = 'lightweight',
        n_vision: int = 576,
        dropout: float = 0.15,
        **kwargs
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.adapter_type = adapter_type

        adapter_cls = {
            'lightweight': LightweightAdapter,
        }.get(adapter_type, LightweightAdapter)

        self.adapters = nn.ModuleDict({
            str(idx): adapter_cls(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                n_vision=n_vision,
                dropout=dropout
            )
            for idx in layer_indices
        })

    def get_adapter(self, layer_idx: int):
        return self.adapters[str(layer_idx)]
