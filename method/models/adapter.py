"""Adapter 模块 - 补偿剪枝后的信息损失"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


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


class PrunedTokenMultiQueryPooler(nn.Module):
    """用多查询注意力池化聚合（通常聚焦于被剪掉的 tokens）。

    输出 K 个上下文向量，提供比单一均值向量更强的表达能力。
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int,
        num_context_tokens: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.num_context_tokens = int(num_context_tokens)

        self.key_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.value_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.queries = nn.Parameter(torch.randn(self.num_context_tokens, bottleneck_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.key_proj.weight, gain=0.1)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.xavier_uniform_(self.value_proj.weight, gain=0.1)
        nn.init.zeros_(self.value_proj.bias)

    def forward(
        self,
        vision_hidden: torch.Tensor,
        mask: torch.Tensor,
        *,
        relevance: Optional[torch.Tensor] = None,
        focus_pruned: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            vision_hidden: (batch, n_vision, hidden)
            mask: (batch, n_vision), 1=keep, 0=prune
            relevance: (batch, n_vision), optional; larger means more relevant
            focus_pruned: if True, pool over pruned tokens only; else pool over all tokens

        Returns:
            context_tokens: (batch, K, bottleneck_dim)
        """
        b, n, _ = vision_hidden.shape
        dtype = vision_hidden.dtype

        key = self.key_proj(vision_hidden)    # (b, n, d)
        value = self.value_proj(vision_hidden)  # (b, n, d)

        # (K, d) -> (1, K, d) -> (b, K, d)
        q = self.queries.unsqueeze(0).expand(b, -1, -1).to(dtype=dtype)

        # scores: (b, K, n)
        scale = 1.0 / math.sqrt(self.bottleneck_dim)
        scores = torch.einsum("bkd,bnd->bkn", q, key) * scale

        # Masking: focus on pruned tokens by default
        if focus_pruned:
            pruned_mask = (1.0 - mask.to(dtype=dtype)).clamp(min=0.0, max=1.0)  # (b, n)
            scores = scores.masked_fill(pruned_mask.unsqueeze(1) < 0.5, torch.finfo(scores.dtype).min)

        # Optional relevance bias: add log(relevance) to scores (stable for tiny relevance)
        if relevance is not None:
            rel = relevance.to(dtype=scores.dtype).clamp(min=1e-6)
            scores = scores + rel.log().unsqueeze(1)

        attn = F.softmax(scores, dim=-1)  # (b, K, n)
        attn = self.dropout(attn)
        ctx = torch.einsum("bkn,bnd->bkd", attn, value)  # (b, K, d)
        return ctx


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


class RepairContextEncoder(nn.Module):
    """将 (vision_hidden, mask) 编码成可缓存的修复上下文。

    设计目标：
    - 不缓存被剪掉 token 序列（显存/带宽代价大）
    - 只缓存低维向量，供后续“延迟修复”层使用

    输出：
    - mask_emb: (batch, bottleneck_dim)
    - pruned_emb: (batch, bottleneck_dim) 或 None（取决于 use_pruned_info）
    - context_tokens: (batch, K, bottleneck_dim) 或 None（取决于 num_context_tokens）
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 256,
        n_vision: int = 576,
        mask_encoder_type: str = "attention",
        use_pruned_info: bool = True,
        num_context_tokens: int = 0,
        context_dropout: float = 0.0,
        use_q2v_relevance: bool = False,
    ):
        super().__init__()
        self.use_pruned_info = use_pruned_info
        self.num_context_tokens = int(num_context_tokens)
        self.use_q2v_relevance = bool(use_q2v_relevance)

        if mask_encoder_type == "attention":
            self.mask_encoder = MaskAttentionEncoder(
                n_vision=n_vision,
                d_pos=64,
                bottleneck_dim=bottleneck_dim,
            )
        else:
            # 兼容旧逻辑：Linear 编码 mask
            self.mask_encoder = nn.Sequential(
                nn.Linear(n_vision, bottleneck_dim),
                nn.GELU(),
                nn.Linear(bottleneck_dim, bottleneck_dim),
            )

        self.pruned_aggregator = PrunedTokenAggregator(hidden_size, bottleneck_dim)
        self.context_pooler = None
        if self.num_context_tokens > 0:
            self.context_pooler = PrunedTokenMultiQueryPooler(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                num_context_tokens=self.num_context_tokens,
                dropout=context_dropout,
            )

    def forward(
        self,
        vision_hidden: torch.Tensor,
        mask: torch.Tensor,
        *,
        q2v_attn: Optional[torch.Tensor] = None,
        use_q2v_relevance: bool = False,
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        mask_input = mask.to(dtype=vision_hidden.dtype)
        mask_emb = self.mask_encoder(mask_input)
        pruned_emb = None
        if self.use_pruned_info:
            pruned_emb = self.pruned_aggregator(vision_hidden, mask)
        if self.context_pooler is None:
            return mask_emb, pruned_emb

        relevance = None
        if use_q2v_relevance and self.use_q2v_relevance and (q2v_attn is not None):
            relevance = q2v_attn
        context_tokens = self.context_pooler(vision_hidden, mask, relevance=relevance, focus_pruned=True)
        return mask_emb, pruned_emb, context_tokens


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

        # 可选：使用外部缓存的 embedding（用于 delayed repair）
        cached_mask_emb = kwargs.get('mask_emb', None)
        cached_pruned_emb = kwargs.get('pruned_emb', None)

        # --- mask embedding ---
        # delayed repair 场景下可能只有 cached_mask_emb，没有完整的 mask
        mask_emb = None
        if cached_mask_emb is not None:
            mask_emb = cached_mask_emb
        elif mask is not None:
            mask_input = mask.to(dtype=x.dtype)
            mask_emb = self.mask_encoder(mask_input)  # (batch, bottleneck)
        if mask_emb is not None:
            condition = condition + mask_emb.unsqueeze(1)  # broadcast to (batch, seq, bottleneck)

        if query is not None:
            query_emb = self.query_proj(query)  # (batch, seq, bottleneck)
            condition = condition + query_emb

        # 新增：被剪枝信息
        pruned_emb = None
        if cached_pruned_emb is not None:
            pruned_emb = cached_pruned_emb
        elif self.use_pruned_info and (mask is not None) and (vision_hidden is not None):
            pruned_emb = self.pruned_aggregator(vision_hidden, mask)  # (batch, bottleneck)
        if pruned_emb is not None:
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


class CrossAttentionRepairAdapter(nn.Module):
    """更强的 delayed-repair adapter：对 K 个 context tokens 做 cross-attention，再生成残差。

    设计目标：
    - 仍然只作用于 gen_answer tokens（由上层调用保证）
    - 每个 answer token 可以对不同的 context token 关注，实现 token-wise 条件化修复
    """

    def __init__(
        self,
        hidden_size: int,
        bottleneck_dim: int = 512,
        num_context_tokens: int = 8,
        dropout: float = 0.15,
        alpha_init: float = 0.1,
        track_delta_loss: bool = False,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim
        self.num_context_tokens = int(num_context_tokens)
        self.track_delta_loss = bool(track_delta_loss)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self._last_delta_loss = None

        self.dropout = nn.Dropout(dropout)

        # Project to bottleneck for attention
        self.q_proj = nn.Linear(hidden_size, bottleneck_dim)
        self.k_proj = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.v_proj = nn.Linear(bottleneck_dim, bottleneck_dim)

        # Main residual path
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        self.act = nn.GELU()

        # Fusion FiLM (condition = cross_attn_out + query_emb + optional cached mask/pruned embs)
        self.film_gamma = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.film_beta = nn.Linear(bottleneck_dim, bottleneck_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.zeros_(self.v_proj.bias)

        nn.init.zeros_(self.film_gamma.weight)
        nn.init.zeros_(self.film_gamma.bias)
        nn.init.zeros_(self.film_beta.weight)
        nn.init.zeros_(self.film_beta.bias)

        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, hidden_size) - (通常是 padded 的 gen_answer hidden)
        Kwargs:
            context_tokens: (batch, K, bottleneck_dim)
            mask_emb: (batch, bottleneck_dim) optional
            pruned_emb: (batch, bottleneck_dim) optional
        """
        context_tokens = kwargs.get("context_tokens", None)
        if context_tokens is None:
            # For this adapter type, context tokens are required.
            if self.training:
                raise ValueError("CrossAttentionRepairAdapter requires `context_tokens` during training.")
            return x

        # Project queries
        q = self.q_proj(x)  # (b, L, d)
        k = self.k_proj(context_tokens)  # (b, K, d)
        v = self.v_proj(context_tokens)  # (b, K, d)

        # Attention: (b, L, K)
        scale = 1.0 / math.sqrt(self.bottleneck_dim)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # (b, L, d)

        # Build condition (token-wise)
        cond = ctx + q

        mask_emb = kwargs.get("mask_emb", None)
        if mask_emb is not None:
            cond = cond + mask_emb.to(dtype=cond.dtype).unsqueeze(1)
        pruned_emb = kwargs.get("pruned_emb", None)
        if pruned_emb is not None:
            cond = cond + pruned_emb.to(dtype=cond.dtype).unsqueeze(1)

        h = self.dropout(self.act(self.down(x)))  # (b, L, d)
        gamma = 1 + self.film_gamma(cond)
        beta = self.film_beta(cond)
        h = gamma * h + beta
        h = self.dropout(h)

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
            'crossattn_repair': CrossAttentionRepairAdapter,
        }.get(adapter_type, LightweightAdapter)

        self.adapters = nn.ModuleDict({
            str(idx): adapter_cls(
                hidden_size=hidden_size,
                bottleneck_dim=bottleneck_dim,
                num_context_tokens=int(kwargs.get("num_context_tokens", 8)),
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
