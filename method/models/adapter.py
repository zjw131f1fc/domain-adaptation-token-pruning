"""Adapter 模块 - 补偿剪枝后的信息损失"""

import torch
import torch.nn as nn
from typing import Optional


class PruningAdapter(nn.Module):
    """剪枝补偿 Adapter（简单版本）

    两层 MLP，补偿剪枝造成的信息损失。
    """

    def __init__(self, hidden_size: int, bottleneck_dim: int = 512, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim

        self.down = nn.Linear(hidden_size, self.bottleneck_dim)
        self.mid = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.up = nn.Linear(self.bottleneck_dim, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.act(self.down(x))
        h = self.act(self.mid(h))
        return x + self.up(h)


class LightweightAdapter(nn.Module):
    """轻量级 Adapter：Mask-Aware + Query-Aware FiLM 调制

    用 pruning mask + 当前 token 的 attention query 进行补偿。
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 512,
        n_vision: int = 576,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim

        # Mask encoder: 编码哪些 token 被剪掉
        self.mask_encoder = nn.Sequential(
            nn.Linear(n_vision, bottleneck_dim),
            nn.GELU(),
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
        **kwargs
    ) -> torch.Tensor:
        """
        参数:
            x: (batch, seq, hidden_size) - attention output
            mask: (batch, n_vision) - pruning mask (1=keep, 0=prune)
            query: (batch, seq, hidden_size) - attention query states
        """
        h = self.act(self.down(x))  # (batch, seq, bottleneck)

        # 构建 condition
        condition = torch.zeros_like(h)  # (batch, seq, bottleneck)

        if mask is not None:
            mask_emb = self.mask_encoder(mask.to(dtype=x.dtype))  # (batch, bottleneck)
            condition = condition + mask_emb.unsqueeze(1)  # broadcast to (batch, seq, bottleneck)

        if query is not None:
            # 训练时：用同分布噪声代替 query，让 adapter 不依赖具体的 query 值
            if self.training:
                query_noise = torch.randn_like(query) * query.std() + query.mean()
                query_emb = self.query_proj(query_noise)
            else:
                query_emb = self.query_proj(query)
            condition = condition + query_emb

        # FiLM modulation
        gamma = 1 + self.gamma_net(condition)  # (batch, seq, bottleneck)
        beta = self.beta_net(condition)
        h = gamma * h + beta

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
        **kwargs
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.adapter_type = adapter_type

        adapter_cls = {
            'simple': PruningAdapter,
            'lightweight': LightweightAdapter,
        }.get(adapter_type, LightweightAdapter)

        self.adapters = nn.ModuleDict({
            str(idx): adapter_cls(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                n_vision=n_vision,
            )
            for idx in layer_indices
        })

    def get_adapter(self, layer_idx: int):
        return self.adapters[str(layer_idx)]
