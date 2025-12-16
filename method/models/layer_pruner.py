"""Layer-Specific Pruner - LLM内部多层剪枝模块

实现在LLM的多个层（例如Layer 10/20/31）分别进行独立的vision token剪枝。

核心思想：
- 早期层（Layer 10）: 去除明显不相关的vision tokens（如背景）
- 中期层（Layer 20）: 进一步精炼，去除冗余细节
- 后期层（Layer 31）: 只保留对最终预测最关键的tokens

每层独立学习，实现渐进式剪枝（progressive pruning）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class LayerSpecificPruner(nn.Module):
    """多层剪枝管理器

    为多个LLM层创建独立的剪枝头，每个层学习不同的剪枝策略。

    参数:
        d_model: LLM hidden state维度（例如LLaMA-7B的4096）
        d_text: text embedding维度（通常与d_model相同）
        layer_indices: 要剪枝的层索引列表（例如[10, 20, 31]）
        d_internal: 内部处理维度
        num_heads: cross-attention头数
    """

    def __init__(
        self,
        d_model: int = 4096,
        d_text: int = 4096,
        layer_indices: List[int] = [10, 20, 31],
        d_internal: int = 512,
        num_heads: int = 4
    ):
        super().__init__()
        self.layer_indices = layer_indices

        # 为每个层创建独立的剪枝头
        self.pruners = nn.ModuleDict({
            str(layer_idx): VisionPrunerHead(
                d_vision=d_model,
                d_text=d_text,
                d_internal=d_internal,
                num_heads=num_heads
            )
            for layer_idx in layer_indices
        })

    def get_pruner(self, layer_idx: int) -> 'VisionPrunerHead':
        """获取指定层的剪枝头"""
        key = str(layer_idx)
        if key not in self.pruners:
            raise ValueError(f"No pruner for layer {layer_idx}. Available: {self.layer_indices}")
        return self.pruners[key]

    def get_all_layers(self) -> List[int]:
        """返回所有剪枝层的索引"""
        return self.layer_indices

    def set_temperature(self, temperature: float):
        """设置所有剪枝头的temperature"""
        for pruner in self.pruners.values():
            pruner.set_temperature(temperature)


class VisionPrunerHead(nn.Module):
    """单层Vision Token剪枝头

    架构：Cross-Attention + MLP
    - Vision tokens关注question embeddings（cross-attention）
    - 基于融合后的表示预测每个vision token的保留/丢弃决策

    参数:
        d_vision: vision token的hidden state维度
        d_text: text embedding维度
        d_internal: 内部处理维度
        num_heads: cross-attention头数
    """

    def __init__(
        self,
        d_vision: int = 4096,
        d_text: int = 4096,
        d_internal: int = 512,
        num_heads: int = 4
    ):
        super().__init__()
        self.d_internal = d_internal

        # === Feature投影 ===
        self.vision_proj = nn.Linear(d_vision, d_internal)
        self.text_proj = nn.Linear(d_text, d_internal)

        # === Cross-Attention: vision tokens关注question ===
        # 这使得剪枝决策能够基于问题内容
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_internal,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # === Mask预测头 ===
        # 输入: cross-attention后的vision features
        # 输出: 每个token的keep/drop logit
        self.mask_predictor = nn.Sequential(
            nn.Linear(d_internal, d_internal // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal // 2, 1)
        )

        # Temperature（外部动态更新）
        self.temperature = 1.0

    def forward(
        self,
        vision_hidden: torch.Tensor,
        question_embeddings: torch.Tensor,
        use_gumbel: bool = True
    ) -> torch.Tensor:
        """前向传播

        参数:
            vision_hidden: (batch, n_vision, d_vision) - 当前层的vision token hidden states
            question_embeddings: (batch, n_text, d_text) - question embeddings
            use_gumbel: bool - 是否使用Gumbel-Softmax（训练时True）

        返回:
            soft_mask: (batch, n_vision) - 每个token的保留概率，范围[0, 1]
        """
        llm_device = vision_hidden.device
        if self.vision_proj.weight.device != llm_device:
            self.to(llm_device)
        if question_embeddings.device != llm_device:
            question_embeddings = question_embeddings.to(llm_device)
        
        # === Step 1: 投影到内部维度 ===
        V = self.vision_proj(vision_hidden)         # (batch, n_vision, d_internal)
        Q = self.text_proj(question_embeddings)     # (batch, n_text, d_internal)

        # === Step 2: Cross-Attention - vision关注question ===
        # query=V (vision tokens), key=Q, value=Q (question)
        attended_V, attn_weights = self.cross_attn(
            query=V,
            key=Q,
            value=Q,
            need_weights=False
        )  # (batch, n_vision, d_internal)

        # === Step 3: 预测keep/drop logits ===
        keep_logits = self.mask_predictor(attended_V).squeeze(-1)  # (batch, n_vision)

        # === Step 4: Gumbel-Softmax（可微分的二分类） ===
        if use_gumbel and self.training:
            # 将二分类问题转换为[drop_logit, keep_logit]的2-way softmax
            stacked_logits = torch.stack([
                torch.zeros_like(keep_logits),  # drop的logit固定为0
                keep_logits                      # keep的logit为预测值
            ], dim=-1)  # (batch, n_vision, 2)

            # 为每个类别独立生成Gumbel噪声（标准Gumbel-Softmax实现）
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(stacked_logits) + 1e-8) + 1e-8)
            gumbel_logits = (stacked_logits + gumbel_noise) / self.temperature

            # Softmax + 提取keep概率
            probs = F.softmax(gumbel_logits, dim=-1)  # (batch, n_vision, 2)
            soft_mask = probs[..., 1]  # (batch, n_vision) - 提取P(keep)
        else:
            # 推理模式：使用sigmoid（确定性）
            soft_mask = torch.sigmoid(keep_logits / self.temperature)

        return soft_mask

    def set_temperature(self, temperature: float):
        """设置temperature"""
        self.temperature = temperature


class VisionPrunerHeadSimple(nn.Module):
    """简化版Pruner Head - 不使用Cross-Attention

    适用场景：
    - 计算资源受限
    - 不需要question-aware的剪枝
    - 只基于vision token自身特征判断重要性

    架构：MLP直接预测mask
    """

    def __init__(
        self,
        d_vision: int = 4096,
        d_internal: int = 512
    ):
        super().__init__()

        # === 简单的MLP ===
        self.mask_predictor = nn.Sequential(
            nn.Linear(d_vision, d_internal),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal, d_internal // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal // 2, 1)
        )

        self.temperature = 1.0

    def forward(
        self,
        vision_hidden: torch.Tensor,
        use_gumbel: bool = True
    ) -> torch.Tensor:
        """前向传播（不需要question_embeddings）

        参数:
            vision_hidden: (batch, n_vision, d_vision)
            use_gumbel: bool

        返回:
            soft_mask: (batch, n_vision)
        """
        # 直接从vision hidden预测keep logits
        keep_logits = self.mask_predictor(vision_hidden).squeeze(-1)

        # Gumbel-Softmax或Sigmoid
        if use_gumbel and self.training:
            stacked_logits = torch.stack([
                torch.zeros_like(keep_logits),
                keep_logits
            ], dim=-1)

            # 为每个类别独立生成Gumbel噪声（标准Gumbel-Softmax实现）
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(stacked_logits) + 1e-8) + 1e-8)
            gumbel_logits = (stacked_logits + gumbel_noise) / self.temperature

            probs = F.softmax(gumbel_logits, dim=-1)
            soft_mask = probs[..., 1]
        else:
            soft_mask = torch.sigmoid(keep_logits / self.temperature)

        return soft_mask

    def set_temperature(self, temperature: float):
        """设置temperature"""
        self.temperature = temperature
