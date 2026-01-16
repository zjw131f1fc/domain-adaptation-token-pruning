"""统一的模型 Wrapper，用于 FSDP 分布式训练

将 backbone、layer_pruners、discriminator 包装成一个模型，
让 HuggingFace Trainer 的 FSDP 可以统一管理。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class PruningModelWrapper(nn.Module):
    """包装所有训练组件的统一模型

    FSDP 会自动分片这个模型的所有子模块。
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_pruners: nn.Module,
        discriminator: nn.Module,
        freeze_backbone: bool = True
    ):
        super().__init__()
        self.backbone = backbone
        self.layer_pruners = layer_pruners
        self.discriminator = discriminator

        # 冻结 backbone
        if freeze_backbone and hasattr(backbone, "model"):
            for param in backbone.model.parameters():
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        """Forward 方法（实际训练逻辑在 train_step 中）"""
        raise NotImplementedError("Use train_step for actual training")
