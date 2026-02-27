"""Adapter 模块 - 补偿剪枝后的信息损失"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PrunedTokenAggregator(nn.Module):
    """聚合被剪枝 token 的信息

    将被剪掉的 vision tokens 的 hidden states 加权聚合，
    为 Adapter 提供"丢失了什么信息"的上下文。
    """

    def __init__(self, hidden_size: int, bottleneck_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim

        # 投影到 bottleneck
        self.proj = nn.Linear(hidden_size, bottleneck_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight, gain=0.1)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        vision_hidden: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            vision_hidden: (batch, n_vision, hidden_size) - 所有 vision tokens
            mask: (batch, n_vision) - pruning mask (1=keep, 0=prune)

        Returns:
            pruned_summary: (batch, bottleneck_dim) - 被剪枝 tokens 的聚合表示
        """
        # 反转 mask：1=被剪, 0=保留
        pruned_mask = 1 - mask  # (batch, n_vision)

        # 提取被剪枝 tokens 的 hidden states
        pruned_hidden = vision_hidden * pruned_mask.unsqueeze(-1)  # (batch, n_vision, hidden)

        # 加权平均（避免除零）
        pruned_sum = pruned_hidden.sum(dim=1)  # (batch, hidden)
        pruned_count = pruned_mask.sum(dim=1, keepdim=True).clamp(min=1)  # (batch, 1)
        pruned_avg = pruned_sum / pruned_count  # (batch, hidden)

        # 投影到 bottleneck
        return self.proj(pruned_avg)  # (batch, bottleneck)


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
        actual_n_vision = mask.shape[1]

        # 如果实际 vision tokens 数量与预设不同，插值 pos_embedding
        if actual_n_vision != self.n_vision:
            # (n_vision, d_pos) -> (1, d_pos, n_vision) -> interpolate -> (1, d_pos, actual) -> (actual, d_pos)
            pos_emb = self.pos_embedding.T.unsqueeze(0)  # (1, d_pos, n_vision)
            pos_emb = F.interpolate(pos_emb, size=actual_n_vision, mode='linear', align_corners=False)
            pos_emb = pos_emb.squeeze(0).T  # (actual_n_vision, d_pos)
        else:
            pos_emb = self.pos_embedding

        # mask * pos_emb: (batch, n_vision, d_pos)
        # 被剪掉的位置 (mask=0) 对应的 embedding 为 0
        mask_with_pos = mask.unsqueeze(-1) * pos_emb  # (batch, actual_n_vision, d_pos)

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
    """轻量级 Adapter：Mask-Aware + Query-Aware + Pruned-Info-Aware FiLM 调制

    用 pruning mask + 当前 token 的 attention query + 被剪枝信息 进行补偿。
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 512,
        n_vision: int = 576,
        dropout: float = 0.15,
        mask_encoder_type: str = 'attention',  # 'attention' or 'linear'
        use_pruned_info: bool = True,  # 是否使用被剪枝信息
        alpha_init: float = 0.1,
        track_delta_loss: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.use_pruned_info = use_pruned_info
        self.track_delta_loss = track_delta_loss
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self._last_delta_loss = None

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

        # Pruned info aggregator: 聚合被剪枝 token 的信息
        if use_pruned_info:
            self.pruned_aggregator = PrunedTokenAggregator(hidden_size, bottleneck_dim)

        # FiLM: 根据 (mask + query + pruned_info) 生成调制参数
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

        # Pruned aggregator 已经在自己的 __init__ 中初始化了，不需要在这里初始化

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
        vision_hidden: Optional[torch.Tensor] = None,  # 新增：vision tokens 的 hidden states
        **kwargs
    ) -> torch.Tensor:
        """
        参数:
            x: (batch, seq, hidden_size) - attention output
            mask: (batch, n_vision) - pruning mask (1=keep, 0=prune)
            query: (batch, seq, hidden_size) - attention query states
            vision_hidden: (batch, n_vision, hidden_size) - vision tokens 的 hidden states
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

        # 新增：被剪枝信息
        if vision_hidden is not None and mask is not None and self.use_pruned_info:
            pruned_emb = self.pruned_aggregator(vision_hidden, mask)  # (batch, bottleneck)
            condition = condition + pruned_emb.unsqueeze(1)  # broadcast to (batch, seq, bottleneck)

        # FiLM modulation
        gamma = 1 + self.gamma_net(condition)  # (batch, seq, bottleneck)
        beta = self.beta_net(condition)
        h = gamma * h + beta

        h = self.dropout(h)  # Dropout after FiLM

        delta = self.up(h)
        delta_scaled = self.alpha * delta
        if self.training and self.track_delta_loss:
            self._last_delta_loss = (delta_scaled.float() ** 2).mean()
        return x + delta_scaled

    def pop_delta_loss(self) -> Optional[torch.Tensor]:
        loss = self._last_delta_loss
        self._last_delta_loss = None
        return loss


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
        use_pruned_info: bool = True,  # 新增
        adapter_alpha_init: float = 0.1,
        track_delta_loss: bool = False,
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
                mask_encoder_type=mask_encoder_type,
                use_pruned_info=use_pruned_info,
                alpha_init=adapter_alpha_init,
                track_delta_loss=track_delta_loss,
            )
            for idx in layer_indices
        })

    def get_adapter(self, layer_idx: int):
        return self.adapters[str(layer_idx)]

    def collect_delta_loss(self) -> Optional[torch.Tensor]:
        total = None
        count = 0
        for adapter in self.adapters.values():
            loss = adapter.pop_delta_loss()
            if loss is None:
                continue
            total = loss if total is None else total + loss
            count += 1
        if total is None:
            return None
        return total / max(count, 1)


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
        use_pruned_info: bool = True,  # 新增
        adapter_alpha_init: float = 0.1,
        track_delta_loss: bool = False,
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
                mask_encoder_type=mask_encoder_type,
                use_pruned_info=use_pruned_info,
                alpha_init=adapter_alpha_init,
                track_delta_loss=track_delta_loss,
            )
            self.text_adapters[str(idx)] = LightweightAdapter(
                hidden_size=hidden_size,
                bottleneck_dim=text_bottleneck_dim,
                n_vision=n_vision,
                dropout=dropout,
                mask_encoder_type=mask_encoder_type,
                use_pruned_info=use_pruned_info,
                alpha_init=adapter_alpha_init,
                track_delta_loss=track_delta_loss,
            )

    def get_adapters(self, layer_idx: int):
        """返回指定层的两个 Adapter (vision, text)"""
        idx_str = str(layer_idx)
        return (
            self.vision_adapters[idx_str],
            self.text_adapters[idx_str],
        )

    def collect_delta_loss(self) -> Optional[torch.Tensor]:
        total = None
        count = 0
        for adapter in list(self.vision_adapters.values()) + list(self.text_adapters.values()):
            loss = adapter.pop_delta_loss()
            if loss is None:
                continue
            total = loss if total is None else total + loss
            count += 1
        if total is None:
            return None
        return total / max(count, 1)
