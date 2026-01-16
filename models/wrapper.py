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
        backbone,  # LLaVAMLLMBackbone (非 nn.Module)
        layer_pruners: nn.Module,
        discriminator: nn.Module,
        freeze_backbone: bool = True
    ):
        super().__init__()
        # 保留 backbone wrapper 引用（用于调用 preprocess_batch, generate 等方法）
        self._backbone_wrapper = backbone
        # 注册 backbone.model 为子模块，让 FSDP 能找到 LlamaDecoderLayer
        self.backbone_model = backbone.model
        self.layer_pruners = layer_pruners
        self.discriminator = discriminator

        # 冻结 backbone
        if freeze_backbone:
            for param in self.backbone_model.parameters():
                param.requires_grad = False

    @property
    def backbone(self):
        """返回 backbone wrapper，保留所有方法（preprocess_batch, generate 等）"""
        return self._backbone_wrapper

    def forward(self, *args, **kwargs):
        """Forward 方法（实际训练逻辑在 train_step 中）"""
        raise NotImplementedError("Use train_step for actual training")
