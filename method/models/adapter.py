"""Adapter 模块 - 补偿剪枝后的信息损失

剪枝后的 attention 聚合结果经过 adapter 修正后：
1. 用于判别（训练时）
2. 实际进入 FFN（训练和推理都用）
"""

import torch
import torch.nn as nn
from typing import Optional


class PruningAdapter(nn.Module):
    """剪枝补偿 Adapter（原始版本）

    两层 MLP，补偿剪枝造成的信息损失。

    参数:
        hidden_size: 输入/输出维度（与 LLM hidden_size 相同）
        bottleneck_dim: 瓶颈层维度（默认 512）
    """

    def __init__(self, hidden_size: int, bottleneck_dim: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck_dim = bottleneck_dim or 512

        # 两层结构 + 残差
        self.down = nn.Linear(hidden_size, self.bottleneck_dim)
        self.mid = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.up = nn.Linear(self.bottleneck_dim, hidden_size)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        参数:
            x: (batch, seq, hidden_size) - attention output
            **kwargs: 忽略其他参数（兼容 QueryAwareAdapter 接口）
        """
        h = self.act(self.down(x))
        h = self.act(self.mid(h))
        return x + self.up(h)


class QueryAwareAdapter(nn.Module):
    """Query-Aware + Mask-Aware Adapter

    根据剪枝 mask 和 question 语义动态调整修正策略。

    设计理念：
    1. Mask encoding: 编码哪些 token 被剪掉
    2. Query pooling: 提取问题的语义表示
    3. FiLM conditioning: 根据 (mask, query) 动态生成变换参数

    参数:
        hidden_size: LLM hidden size
        n_vision: vision token 数量（用于 mask encoding）
        bottleneck_dim: adapter 瓶颈维度
        n_heads: query pooling 的 attention 头数
        dropout: dropout 比例
    """

    def __init__(
        self,
        hidden_size: int,
        n_vision: int = 576,
        bottleneck_dim: int = 512,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_vision = n_vision
        self.bottleneck_dim = bottleneck_dim

        # 1. Mask encoder: 编码剪枝 pattern
        # mask: (batch, n_vision) binary -> (batch, hidden)
        self.mask_encoder = nn.Sequential(
            nn.Linear(n_vision, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # 2. Query pooling: 用可学习 token 通过 cross-attention 聚合 question 信息
        self.query_token = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.02)
        self.query_pool = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_size)

        # 3. Condition fusion: 融合 mask 和 query 信息
        self.condition_fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size)
        )

        # 4. FiLM generator: 生成 scale (gamma) 和 shift (beta)
        self.gamma_net = nn.Linear(hidden_size, bottleneck_dim)
        self.beta_net = nn.Linear(hidden_size, bottleneck_dim)

        # 5. Main adapter: down -> modulate -> up
        self.down = nn.Linear(hidden_size, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        # FiLM 初始化：gamma=1, beta=0（初始时不改变）
        nn.init.ones_(self.gamma_net.weight.data.mean(dim=1, keepdim=True))
        nn.init.zeros_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

        # Up projection 初始化为小值（残差连接初始贡献小）
        nn.init.xavier_uniform_(self.up.weight, gain=0.1)
        nn.init.zeros_(self.up.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        question_hidden: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        参数:
            x: (batch, seq, hidden_size) - attention output
            mask: (batch, n_vision) - pruning mask (1=keep, 0=prune)
            question_hidden: (batch, n_question, hidden_size) - question token hidden states

        返回:
            (batch, seq, hidden_size) - 修正后的 attention output
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        # 如果没有提供 mask 或 question_hidden，退化为普通 adapter
        if mask is None or question_hidden is None:
            h = self.act(self.down(x))
            h = self.dropout(h)
            return x + self.up(h)

        # 1. Encode mask pattern
        # mask: (batch, n_vision) -> (batch, hidden)
        # 确保 mask 的 dtype 与模型权重一致
        mask_float = mask.to(dtype=dtype)
        mask_emb = self.mask_encoder(mask_float)

        # 2. Pool question into single vector
        # query_token: (1, 1, hidden) -> (batch, 1, hidden)
        query_token = self.query_token.expand(batch_size, -1, -1).to(dtype)
        # Cross-attention: query_token attends to question_hidden
        q_pooled, _ = self.query_pool(
            query_token,
            question_hidden,
            question_hidden
        )
        q_pooled = self.query_norm(q_pooled.squeeze(1))  # (batch, hidden)

        # 3. Fuse mask and query condition
        condition = torch.cat([mask_emb, q_pooled], dim=-1)  # (batch, hidden*2)
        condition = self.condition_fusion(condition)  # (batch, hidden)

        # 4. Generate FiLM parameters
        gamma = self.gamma_net(condition)  # (batch, bottleneck)
        beta = self.beta_net(condition)    # (batch, bottleneck)

        # Expand for broadcasting: (batch, bottleneck) -> (batch, 1, bottleneck)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # 5. Conditioned transformation with FiLM
        h = self.down(x)           # (batch, seq, bottleneck)
        h = self.act(h)
        h = gamma * h + beta       # FiLM modulation
        h = self.dropout(h)
        out = x + self.up(h)       # Residual connection

        return out


class AdapterManager(nn.Module):
    """多层 Adapter 管理器

    参数:
        layer_indices: 需要 adapter 的层索引
        hidden_size: LLM hidden size
        bottleneck_dim: adapter 瓶颈维度
        adapter_type: 'simple' 或 'query_aware'
        n_vision: vision token 数量（query_aware 模式需要）
        n_heads: query pooling 的 attention 头数
        dropout: dropout 比例
    """

    def __init__(
        self,
        layer_indices: list,
        hidden_size: int,
        bottleneck_dim: int = None,
        adapter_type: str = 'simple',
        n_vision: int = 576,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layer_indices = layer_indices
        self.adapter_type = adapter_type

        if adapter_type == 'query_aware':
            self.adapters = nn.ModuleDict({
                str(idx): QueryAwareAdapter(
                    hidden_size=hidden_size,
                    n_vision=n_vision,
                    bottleneck_dim=bottleneck_dim or 512,
                    n_heads=n_heads,
                    dropout=dropout
                )
                for idx in layer_indices
            })
        else:
            # 默认使用简单 adapter
            self.adapters = nn.ModuleDict({
                str(idx): PruningAdapter(hidden_size, bottleneck_dim)
                for idx in layer_indices
            })

    def get_adapter(self, layer_idx: int):
        return self.adapters[str(layer_idx)]
