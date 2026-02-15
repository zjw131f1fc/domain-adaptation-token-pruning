"""Adapter 模块 - 补偿剪枝后的信息损失"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MaskAttentionEncoder(nn.Module):
    """Attention 池化的 Mask Encoder

    用位置编码 + Attention 池化来编码 mask，保留空间信息。
    比直接 Linear(576, bottleneck) 更轻量且更有效。
    """

    def __init__(self, n_vision: int = 576, d_pos: int = 64, bottleneck_dim: int = 512):
        super().__init__()
        self.n_vision = n_vision
        self.d_pos = d_pos

        # 位置编码（可学习）
        self.pos_embedding = nn.Parameter(torch.randn(n_vision, d_pos) * 0.02)

        # Attention query（可学习）
        self.attn_query = nn.Parameter(torch.randn(1, d_pos) * 0.02)

        # 输出投影
        self.out_proj = nn.Linear(d_pos, bottleneck_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mask: (batch, n_vision) - pruning mask, 1=keep, 0=prune

        Returns:
            mask_emb: (batch, bottleneck_dim)
        """
        batch_size = mask.shape[0]

        # mask * pos_emb: (batch, n_vision, d_pos)
        # 被剪掉的位置 (mask=0) 对应的 embedding 为 0
        mask_with_pos = mask.unsqueeze(-1) * self.pos_embedding  # (batch, 576, 64)

        # Attention: query @ keys
        # (1, d_pos) @ (batch, d_pos, n_vision) -> (batch, 1, n_vision)
        scale = 1.0 / math.sqrt(self.d_pos)
        attn_scores = torch.matmul(self.attn_query, mask_with_pos.transpose(-1, -2)) * scale

        # Softmax（被剪掉的位置 embedding 为 0，attention 会自然降低）
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, 1, n_vision)

        # Weighted sum: (batch, 1, n_vision) @ (batch, n_vision, d_pos) -> (batch, 1, d_pos)
        pooled = torch.matmul(attn_weights, mask_with_pos).squeeze(1)  # (batch, d_pos)

        # 投影到 bottleneck
        return self.out_proj(pooled)  # (batch, bottleneck_dim)


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
        mask_encoder_type: str = 'attention',  # 'attention' or 'linear'
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim

        # Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout)

        # Mask encoder: 编码哪些 token 被剪掉
        if mask_encoder_type == 'attention':
            self.mask_encoder = MaskAttentionEncoder(
                n_vision=n_vision,
                d_pos=64,
                bottleneck_dim=bottleneck_dim
            )
        else:
            # 原始的 Linear encoder（向后兼容）
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
        **kwargs
    ) -> torch.Tensor:
        """
        参数:
            x: (batch, seq, hidden_size) - attention output
            mask: (batch, n_vision) - pruning mask (1=keep, 0=prune)
            query: (batch, seq, hidden_size) - attention query states
        """
        h = self.dropout(self.act(self.down(x)))  # (batch, seq, bottleneck)

        # 构建 condition
        condition = torch.zeros_like(h)  # (batch, seq, bottleneck)

        if mask is not None:
            mask_input = mask.to(dtype=x.dtype)
            mask_emb = self.mask_encoder(mask_input)  # (batch, bottleneck)
            condition = condition + mask_emb.unsqueeze(1)  # broadcast to (batch, seq, bottleneck)

        if query is not None:
            query_emb = self.query_proj(query)  # (batch, seq, bottleneck)
            condition = condition + query_emb

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
        mask_encoder_type: str = 'attention',
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
                dropout=dropout,
                mask_encoder_type=mask_encoder_type
            )
            for idx in layer_indices
        })

    def get_adapter(self, layer_idx: int):
        return self.adapters[str(layer_idx)]


class SeparatedAdapterManager(nn.Module):
    """分离式 Adapter 管理器 - 为 vision/text tokens 使用独立的 Adapter

    text_adapter 同时处理 question tokens 和 answer tokens（generator）
    """

    def __init__(
        self,
        layer_indices: list,
        hidden_size: int,
        vision_bottleneck_dim: int = 256,
        text_bottleneck_dim: int = 256,
        answer_bottleneck_dim: int = 512,  # 保留参数但不使用，向后兼容
        n_vision: int = 576,
        dropout: float = 0.15,
        mask_encoder_type: str = 'attention',
        **kwargs
    ):
        super().__init__()
        self.layer_indices = layer_indices

        # 每层有两个独立的 Adapter
        self.vision_adapters = nn.ModuleDict()
        self.text_adapters = nn.ModuleDict()

        for idx in layer_indices:
            self.vision_adapters[str(idx)] = LightweightAdapter(
                hidden_size=hidden_size,
                bottleneck_dim=vision_bottleneck_dim,
                n_vision=n_vision,
                dropout=dropout,
                mask_encoder_type=mask_encoder_type
            )
            self.text_adapters[str(idx)] = LightweightAdapter(
                hidden_size=hidden_size,
                bottleneck_dim=text_bottleneck_dim,
                n_vision=n_vision,
                dropout=dropout,
                mask_encoder_type=mask_encoder_type
            )

    def get_adapters(self, layer_idx: int):
        """返回指定层的两个 Adapter (vision, text)"""
        idx_str = str(layer_idx)
        return (
            self.vision_adapters[idx_str],
            self.text_adapters[idx_str],
        )
