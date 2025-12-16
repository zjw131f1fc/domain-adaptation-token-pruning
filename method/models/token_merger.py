"""Learnable Token Merger - Vision Token合并模块

实现可训练的视觉token合并，用于在LLM输入前减少vision token数量。

核心技术：
1. Importance Scoring: 学习每个token的重要性
2. Gumbel-Top-K: 可微分的top-k选择（保留重要tokens作为cluster centers）
3. Soft Assignment: 通过学习的Q/K投影计算相似度，将所有tokens软分配到cluster centers
4. Temperature Annealing: 训练初期软分配（探索），后期硬分配（确定性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class LearnableTokenMerger(nn.Module):
    """可训练的Vision Token合并器

    工作流程：
    1. Importance Scorer预测每个token的重要性分数
    2. 使用Gumbel-Top-K选择要保留的tokens（作为cluster centers）
    3. 计算所有tokens到cluster centers的相似度（通过Q/K投影）
    4. 通过temperature控制的softmax得到软分配权重
    5. 加权聚合：每个cluster center = 周围tokens的加权和

    参数:
        d_model: vision特征维度（例如CLIP ViT-L/14的1024）
        num_heads: 多头注意力的头数（用于Q/K投影）
        merge_ratio: 保留的token比例（0.5表示576→288）
    """

    def __init__(
        self,
        d_model: int = 1024,
        num_heads: int = 4,
        merge_ratio: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        self.merge_ratio = merge_ratio
        self.num_heads = num_heads

        # === 1. Importance Scorer: 预测每个token的重要性 ===
        # 使用小型MLP：d_model -> d_model//2 -> 1
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

        # === 2. Query/Key投影: 用于计算token之间的相似度 ===
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)

        # === 3. Temperature（外部动态更新） ===
        self.temperature = 1.0

    def forward(
        self,
        vision_features: torch.Tensor,
        use_gumbel: bool = True
    ) -> Dict[str, torch.Tensor]:
        """前向传播

        参数:
            vision_features: (batch, N, d_model) - CLIP输出的vision features，N=576
            use_gumbel: bool - 是否使用Gumbel noise（训练时True，推理时False）

        返回:
            {
                'merged_features': (batch, M, d_model) - 合并后的features，M ≈ N * merge_ratio
                'merge_indices': (batch, M) - 保留的token索引
                'merge_weights': (batch, N, M) - 软分配矩阵（每行和为1）
                'importance_logits': (batch, N) - 重要性分数（用于稀疏损失）
            }
        """
        batch, N, d = vision_features.shape
        M = int(N * self.merge_ratio)  # 目标保留的token数量

        # === Step 1: 计算每个token的重要性分数 ===
        importance_logits = self.importance_scorer(vision_features).squeeze(-1)  # (batch, N)

        # === Step 2: 使用Gumbel-Top-K选择要保留的tokens ===
        if use_gumbel and self.training:
            # Gumbel-Max技巧：添加噪声使采样可微分
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(importance_logits) + 1e-8) + 1e-8)
            perturbed_logits = importance_logits + gumbel_noise
            _, top_k_indices = torch.topk(perturbed_logits, k=M, dim=-1)  # (batch, M)
        else:
            # 推理模式：确定性top-K
            _, top_k_indices = torch.topk(importance_logits, k=M, dim=-1)

        # === Step 3: 提取保留的tokens作为cluster centers ===
        # 使用gather提取选中的tokens
        cluster_centers = torch.gather(
            vision_features, 1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, d)
        )  # (batch, M, d)

        # === Step 4: 计算所有tokens到cluster centers的相似度 ===
        Q = self.q_proj(vision_features)    # (batch, N, d) - 所有tokens作为query
        K = self.k_proj(cluster_centers)    # (batch, M, d) - cluster centers作为key

        # Scaled dot-product similarity
        similarity = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d)  # (batch, N, M)

        # === Step 5: 软分配（temperature控制的softmax） ===
        # 每个token分配到所有cluster centers的概率分布
        merge_weights = F.softmax(similarity / self.temperature, dim=-1)  # (batch, N, M)

        # === Step 6: 软合并 - 将所有tokens加权聚合到cluster centers ===
        # merge_weights.T @ vision_features: 每个cluster center = 周围tokens的加权和
        merged_features = torch.matmul(
            merge_weights.transpose(-2, -1),  # (batch, M, N)
            vision_features                    # (batch, N, d)
        )  # (batch, M, d)

        return {
            'merged_features': merged_features,
            'merge_indices': top_k_indices,
            'merge_weights': merge_weights,
            'importance_logits': importance_logits
        }

    def set_temperature(self, temperature: float):
        """设置temperature（外部调用，用于annealing）"""
        self.temperature = temperature


class LearnableTokenMergerV2(nn.Module):
    """增强版Token Merger - 添加Question-Aware机制

    与V1的区别：
    - 使用Cross-Attention让vision tokens关注question
    - Question-aware的重要性评分
    - 更适合VQA等需要问题引导的任务

    参数:
        d_vision: vision特征维度
        d_text: text embedding维度
        d_internal: 内部处理维度
        num_heads: 多头注意力头数
        merge_ratio: 保留比例
    """

    def __init__(
        self,
        d_vision: int = 1024,
        d_text: int = 4096,
        d_internal: int = 512,
        num_heads: int = 4,
        merge_ratio: float = 0.5
    ):
        super().__init__()
        self.d_internal = d_internal
        self.merge_ratio = merge_ratio

        # === Feature投影到统一维度 ===
        self.vision_proj = nn.Linear(d_vision, d_internal)
        self.text_proj = nn.Linear(d_text, d_internal)

        # === Cross-Attention: vision关注question ===
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_internal,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # === Importance Scorer（基于cross-attention后的features） ===
        self.importance_scorer = nn.Sequential(
            nn.Linear(d_internal, d_internal // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal // 2, 1)
        )

        # === Q/K投影用于token合并 ===
        self.q_proj = nn.Linear(d_internal, d_internal)
        self.k_proj = nn.Linear(d_internal, d_internal)

        # === 投影回原始vision维度 ===
        self.output_proj = nn.Linear(d_internal, d_vision)

        self.temperature = 1.0

    def forward(
        self,
        vision_features: torch.Tensor,
        question_embeddings: torch.Tensor,
        use_gumbel: bool = True
    ) -> Dict[str, torch.Tensor]:
        """前向传播（Question-Aware版本）

        参数:
            vision_features: (batch, N, d_vision)
            question_embeddings: (batch, n_text, d_text)
            use_gumbel: bool
        """
        batch, N, _ = vision_features.shape
        M = int(N * self.merge_ratio)

        # === Step 1: 投影到统一维度 ===
        V = self.vision_proj(vision_features)         # (batch, N, d_internal)
        Q = self.text_proj(question_embeddings)       # (batch, n_text, d_internal)

        # === Step 2: Cross-Attention - vision关注question ===
        attended_V, _ = self.cross_attn(
            query=V,
            key=Q,
            value=Q,
            need_weights=False
        )  # (batch, N, d_internal)

        # === Step 3: 基于question-aware features计算重要性 ===
        importance_logits = self.importance_scorer(attended_V).squeeze(-1)  # (batch, N)

        # === Step 4-6: 与V1相同的Gumbel-Top-K + Soft Merge流程 ===
        if use_gumbel and self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(importance_logits) + 1e-8) + 1e-8)
            perturbed_logits = importance_logits + gumbel_noise
            _, top_k_indices = torch.topk(perturbed_logits, k=M, dim=-1)
        else:
            _, top_k_indices = torch.topk(importance_logits, k=M, dim=-1)

        cluster_centers = torch.gather(
            attended_V, 1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, self.d_internal)
        )

        Q_merge = self.q_proj(attended_V)
        K_merge = self.k_proj(cluster_centers)

        similarity = torch.matmul(Q_merge, K_merge.transpose(-2, -1)) / math.sqrt(self.d_internal)
        merge_weights = F.softmax(similarity / self.temperature, dim=-1)

        merged_internal = torch.matmul(
            merge_weights.transpose(-2, -1),
            attended_V
        )

        # === Step 7: 投影回原始vision维度 ===
        merged_features = self.output_proj(merged_internal)  # (batch, M, d_vision)

        return {
            'merged_features': merged_features,
            'merge_indices': top_k_indices,
            'merge_weights': merge_weights,
            'importance_logits': importance_logits
        }

    def set_temperature(self, temperature: float):
        """设置temperature"""
        self.temperature = temperature
