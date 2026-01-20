"""Adapter 模块 - 补偿剪枝后的信息损失

剪枝后的 attention 聚合结果经过 adapter 修正后：
1. 用于判别（训练时）
2. 实际进入 FFN（训练和推理都用）
"""

import torch
import torch.nn as nn


class PruningAdapter(nn.Module):
    """剪枝补偿 Adapter

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.down(x))
        h = self.act(self.mid(h))
        return x + self.up(h)


class AdapterManager(nn.Module):
    """多层 Adapter 管理器"""

    def __init__(self, layer_indices: list, hidden_size: int, bottleneck_dim: int = None):
        super().__init__()
        self.layer_indices = layer_indices

        self.adapters = nn.ModuleDict({
            str(idx): PruningAdapter(hidden_size, bottleneck_dim)
            for idx in layer_indices
        })

    def get_adapter(self, layer_idx: int) -> PruningAdapter:
        return self.adapters[str(layer_idx)]
