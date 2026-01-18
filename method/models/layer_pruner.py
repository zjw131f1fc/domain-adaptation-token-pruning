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
        num_heads: int = 4,
        use_attn_residual: bool = False,
        attn_residual_weight: float = 0.5,
        learnable_attn_weight: bool = False
    ):
        super().__init__()
        self.layer_indices = layer_indices

        # 为每个层创建独立的剪枝头
        self.pruners = nn.ModuleDict({
            str(layer_idx): VisionPrunerHead(
                d_vision=d_model,
                d_text=d_text,
                d_internal=d_internal,
                num_heads=num_heads,
                use_attn_residual=use_attn_residual,
                attn_residual_weight=attn_residual_weight,
                learnable_attn_weight=learnable_attn_weight
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

    架构：Question→Vision Cross-Attention + MLP微调

    核心思想：
    - Question tokens 作为 Query，Vision tokens 作为 Key/Value
    - Attention weights 直接表示 "question 关注哪些 vision tokens"
    - 这是天然的重要性指标，比间接学习更有效
    - MLP 只做微调，主要依赖 attention weights

    参数:
        d_vision: vision token的hidden state维度
        d_text: text embedding维度
        d_internal: 内部处理维度
        num_heads: cross-attention头数
        use_attn_residual: 是否使用LLM内部的attention作为额外信号
        attn_residual_weight: LLM attention的权重
        learnable_attn_weight: 权重是否可学习
    """

    def __init__(
        self,
        d_vision: int = 4096,
        d_text: int = 4096,
        d_internal: int = 512,
        num_heads: int = 4,
        use_attn_residual: bool = False,
        attn_residual_weight: float = 0.5,
        learnable_attn_weight: bool = False
    ):
        super().__init__()
        self.d_internal = d_internal
        self.num_heads = num_heads
        self.use_attn_residual = use_attn_residual

        # === 1. 输入归一化层 ===
        self.vision_input_norm = nn.LayerNorm(d_vision)
        self.text_input_norm = nn.LayerNorm(d_text)

        # === 2. Feature投影 ===
        self.vision_proj = nn.Linear(d_vision, d_internal)
        self.text_proj = nn.Linear(d_text, d_internal)

        # === 3. 投影后归一化 ===
        self.vision_proj_norm = nn.LayerNorm(d_internal)
        self.text_proj_norm = nn.LayerNorm(d_internal)

        # === 4. Cross-Attention: Question → Vision ===
        # Query = Question, Key/Value = Vision
        # Attention weights 直接表示每个 vision token 的重要性
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_internal,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # === 5. 重要性微调 MLP ===
        # 输入: attention-based importance (1维)
        # 输出: 微调后的 keep logit
        self.importance_norm = nn.LayerNorm(1)
        self.refine_mlp = nn.Sequential(
            nn.Linear(1, d_internal // 4),
            nn.LayerNorm(d_internal // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal // 4, 1)
        )

        # 可学习的 attention 权重（控制 attention importance 的贡献）
        self.attn_importance_weight = nn.Parameter(torch.tensor(1.0))

        # 初始化：让 MLP 输出接近 0，主要依赖 attention
        nn.init.zeros_(self.refine_mlp[-1].weight)
        nn.init.zeros_(self.refine_mlp[-1].bias)

        # === 6. LLM Attention Residual配置（可选） ===
        if self.use_attn_residual:
            if learnable_attn_weight:
                self.llm_attn_weight = nn.Parameter(torch.tensor(attn_residual_weight))
            else:
                self.register_buffer('llm_attn_weight', torch.tensor(attn_residual_weight))

        # Temperature（外部动态更新）
        self.temperature = 1.0

    def forward(
        self,
        vision_hidden: torch.Tensor,
        question_embeddings: torch.Tensor,
        use_gumbel: bool = True,
        text_to_vision_attn: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播

        参数:
            vision_hidden: (batch, n_vision, d_vision) - 当前层的vision token hidden states
            question_embeddings: (batch, n_text, d_text) - question embeddings
            use_gumbel: bool - 是否使用Gumbel-Softmax（训练时True）
            text_to_vision_attn: (batch, n_vision) - 可选的LLM内部text→vision attention
            key_padding_mask: (batch, n_vision) - True表示要mask掉的位置（padding）
                             注意：现在是 vision 的 padding mask，因为 vision 是 Key

        返回:
            soft_mask: (batch, n_vision) - 每个token的保留概率，范围[0, 1]
        """
        llm_device = vision_hidden.device
        if self.vision_proj.weight.device != llm_device:
            self.to(llm_device)
        if question_embeddings.device != llm_device:
            question_embeddings = question_embeddings.to(llm_device)

        # === Step 1: 输入归一化 ===
        vision_normed = self.vision_input_norm(vision_hidden)  # (batch, n_vision, d_vision)
        text_normed = self.text_input_norm(question_embeddings)  # (batch, n_text, d_text)

        # === Step 2: 投影到内部维度 + 归一化 ===
        V = self.vision_proj(vision_normed)  # (batch, n_vision, d_internal)
        V = self.vision_proj_norm(V)

        Q = self.text_proj(text_normed)  # (batch, n_text, d_internal)
        Q = self.text_proj_norm(Q)

        # === Step 3: Cross-Attention - Question → Vision ===
        # Query = Q (question), Key = V, Value = V (vision)
        # attn_weights: (batch, n_text, n_vision) - 每个 question token 对每个 vision token 的注意力
        _, attn_weights = self.cross_attn(
            query=Q,      # question tokens 作为 query
            key=V,        # vision tokens 作为 key
            value=V,      # vision tokens 作为 value
            key_padding_mask=key_padding_mask,  # vision 的 padding mask
            need_weights=True,
            average_attn_weights=True  # 返回 head 平均后的权重
        )  # attn_weights: (batch, n_text, n_vision)

        # === Step 4: 计算每个 vision token 的重要性 ===
        # 对所有 question tokens 求平均，得到每个 vision token 被关注的程度
        importance = attn_weights.mean(dim=1)  # (batch, n_vision)

        # === Step 5: MLP 微调 ===
        importance_input = importance.unsqueeze(-1)  # (batch, n_vision, 1)
        importance_input = self.importance_norm(importance_input)
        mlp_adjustment = self.refine_mlp(importance_input).squeeze(-1)  # (batch, n_vision)

        # 组合：attention importance + MLP adjustment
        # importance 范围是 [0, 1]（softmax 输出），需要转换到 logit 空间
        # 使用 log-odds 转换：logit = log(p / (1-p))，但需要避免数值问题
        importance_clamped = torch.clamp(importance, min=1e-6, max=1-1e-6)
        importance_logit = torch.log(importance_clamped / (1 - importance_clamped))

        keep_logits = self.attn_importance_weight * importance_logit + mlp_adjustment

        # === Step 5.5: LLM Attention Residual（可选） ===
        if self.use_attn_residual and text_to_vision_attn is not None:
            text_to_vision_attn = text_to_vision_attn.to(device=keep_logits.device, dtype=keep_logits.dtype)
            # LLM attention 也转换到 logit 空间
            llm_attn_clamped = torch.clamp(text_to_vision_attn, min=1e-6, max=1-1e-6)
            llm_attn_logit = torch.log(llm_attn_clamped / (1 - llm_attn_clamped))
            keep_logits = keep_logits + self.llm_attn_weight * llm_attn_logit

        # === Step 6: Gumbel-Softmax ===
        keep_logits = torch.clamp(keep_logits, min=-10.0, max=10.0)
        stacked_logits = torch.stack([
            torch.zeros_like(keep_logits),  # drop logit = 0
            keep_logits                      # keep logit
        ], dim=-1)  # (batch, n_vision, 2)

        if use_gumbel and self.training:
            y = F.gumbel_softmax(stacked_logits, tau=self.temperature, hard=True, dim=-1)
            soft_mask = y[..., 1]
        else:
            probs = F.softmax(stacked_logits / self.temperature, dim=-1)
            soft_mask = (probs[..., 1] > 0.5).float()

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

        # Gumbel-Softmax: 将二分类问题转换为[drop_logit, keep_logit]的2-way softmax
        stacked_logits = torch.stack([
            torch.zeros_like(keep_logits),  # drop的logit固定为0
            keep_logits                      # keep的logit为预测值
        ], dim=-1)

        if use_gumbel and self.training:
            # 训练模式：使用 PyTorch 的 Gumbel-Softmax 配合 hard=True
            y_soft = F.gumbel_softmax(stacked_logits, tau=self.temperature, hard=True, dim=-1)
            soft_mask = y_soft[..., 1]
        else:
            # 推理模式：确定性 argmax
            probs = F.softmax(stacked_logits / self.temperature, dim=-1)
            soft_mask = (probs[..., 1] > 0.5).float()

        return soft_mask

    def set_temperature(self, temperature: float):
        """设置temperature"""
        self.temperature = temperature
