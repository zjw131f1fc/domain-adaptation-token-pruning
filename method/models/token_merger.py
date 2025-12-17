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


class LearnableTokenMergerV3(nn.Module):
    """固定输出M个tokens的可学习池化Merger（无top-k采样）

    核心改进：
    1. 预定义M个可学习查询向量（learnable pooling slots）
    2. Question-aware：用问题嵌入调制这些查询
    3. Cross-attention池化：查询关注所有vision tokens
    4. 输出固定M个tokens，全程可导，无需Gumbel/top-k
    5. 多层LayerNorm稳定训练

    优势：
    - 训练/推理行为完全一致（无随机性）
    - 梯度流畅通（无top-k断点）
    - 输出维度固定，下游兼容性好
    - LayerNorm保证训练稳定性

    参数:
        d_vision: vision特征维度（1024 for CLIP ViT-L/14）
        d_text: text embedding维度（4096 for LLaMA-7B）
        d_internal: 内部处理维度（建议256-512）
        num_heads: 多头注意力头数
        merge_ratio: 保留比例（用于计算M）
        use_question: 是否使用question调制查询向量
    """

    def __init__(
        self,
        d_vision: int = 1024,
        d_text: int = 4096,
        d_internal: int = 512,
        num_heads: int = 8,
        merge_ratio: float = 0.5,
        use_question: bool = True
    ):
        super().__init__()
        self.d_vision = d_vision
        self.d_internal = d_internal
        self.merge_ratio = merge_ratio
        self.use_question = use_question
        self.num_heads = num_heads

        # 假设N=576（CLIP ViT-L/14输出）
        # 动态计算M（在forward中根据实际N计算）
        self.M = None  # 在第一次forward时确定

        # === 1. 基准查询向量（M个可学习槽位） ===
        # 延迟初始化，在第一次forward时根据N*merge_ratio确定
        self.register_buffer('pool_queries', None)  # 将在forward中初始化为 (M, d_internal)

        # === 2. 输入归一化层 ===
        self.vision_input_norm = nn.LayerNorm(d_vision)  # vision输入归一化

        # === 3. Question调制器（将question信息融入查询） ===
        if self.use_question:
            self.text_input_norm = nn.LayerNorm(d_text)  # text输入归一化
            self.text_proj = nn.Linear(d_text, d_internal)
            self.text_proj_norm = nn.LayerNorm(d_internal)  # text投影后归一化
            # Question条件化：生成查询的偏置/调制
            self.query_modulator = nn.Sequential(
                nn.Linear(d_internal, d_internal),
                nn.GELU(),  # GELU比ReLU更平滑
                nn.Dropout(0.1),
                nn.Linear(d_internal, d_internal)
            )

        # === 4. Vision投影 ===
        self.vision_proj = nn.Linear(d_vision, d_internal)
        self.vision_proj_norm = nn.LayerNorm(d_internal)  # vision投影后归一化

        # === 5. Query归一化（attention前） ===
        self.query_norm = nn.LayerNorm(d_internal)

        # === 6. Cross-Attention：查询关注所有vision tokens ===
        self.pool_attention = nn.MultiheadAttention(
            embed_dim=d_internal,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        # === 7. Attention后归一化（Post-LN风格） ===
        self.post_attn_norm = nn.LayerNorm(d_internal)

        # === 8. FFN层（增加表达能力） ===
        self.ffn = nn.Sequential(
            nn.Linear(d_internal, d_internal * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_internal * 4, d_internal),
            nn.Dropout(0.1)
        )
        self.post_ffn_norm = nn.LayerNorm(d_internal)

        # === 9. 输出投影（回到vision维度） ===
        self.output_proj = nn.Linear(d_internal, d_vision)
        self.output_norm = nn.LayerNorm(d_vision)  # 最终输出归一化

        # 关键：初始化输出投影为接近零，让残差连接主导初始输出
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # === 10. Temperature（可选，用于控制attention的锐度） ===
        self.temperature = 1.0

        # === 11. 残差缩放因子（可学习，初始化为小值让残差主导） ===
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

    def _init_pool_queries(self, N: int, device: torch.device):
        """初始化池化查询向量（第一次forward时调用）"""
        if self.M is None:
            self.M = max(1, int(N * self.merge_ratio))
            # 初始化为标准正态分布
            pool_queries = torch.randn(self.M, self.d_internal, device=device)
            pool_queries = pool_queries / math.sqrt(self.d_internal)  # Xavier初始化
            self.register_buffer('pool_queries', pool_queries)

    def forward(
        self,
        vision_features: torch.Tensor,
        question_embeddings: Optional[torch.Tensor] = None,
        use_gumbel: bool = False  # 保留接口兼容性，但不使用
    ) -> Dict[str, torch.Tensor]:
        """前向传播

        参数:
            vision_features: (batch, N, d_vision) - CLIP输出
            question_embeddings: (batch, n_text, d_text) - 可选，用于question-aware
            use_gumbel: bool - 保留兼容性，但此版本不使用

        返回:
            {
                'merged_features': (batch, M, d_vision) - 固定M个tokens
                'merge_indices': None - 无显式索引（全软池化）
                'merge_weights': (batch, M, N) - 池化权重矩阵
                'importance_logits': None - 无显式重要性分数
            }
        """
        batch, N, _ = vision_features.shape
        device = vision_features.device

        # === Step 1: 初始化池化查询（第一次调用） ===
        if self.pool_queries is None:
            self._init_pool_queries(N, device)

        # === Step 2: Vision输入归一化 + 投影 ===
        vision_normed = self.vision_input_norm(vision_features)  # (batch, N, d_vision)
        V_proj = self.vision_proj(vision_normed)  # (batch, N, d_internal)
        V_proj = self.vision_proj_norm(V_proj)  # 投影后归一化

        # === Step 3: 准备查询向量 ===
        # 基准查询 (M, d_internal) -> (batch, M, d_internal)
        queries = self.pool_queries.unsqueeze(0).expand(batch, -1, -1)

        # Question调制（如果启用）
        if self.use_question and question_embeddings is not None:
            # 输入归一化 + 投影question到内部维度
            Q_text_normed = self.text_input_norm(question_embeddings)  # (batch, n_text, d_text)
            Q_text = self.text_proj(Q_text_normed)  # (batch, n_text, d_internal)
            Q_text = self.text_proj_norm(Q_text)  # 投影后归一化

            # 均值池化：提取全局question表示
            Q_global = Q_text.mean(dim=1)  # (batch, d_internal)

            # 生成查询调制向量
            query_bias = self.query_modulator(Q_global)  # (batch, d_internal)

            # 调制查询：queries = base_queries + question_bias
            queries = queries + query_bias.unsqueeze(1)  # (batch, M, d_internal)

        # === Step 4: Query归一化（attention前） ===
        queries = self.query_norm(queries)  # (batch, M, d_internal)

        # === Step 5: Cross-Attention池化 ===
        # queries关注所有vision tokens
        pooled, attn_weights = self.pool_attention(
            query=queries,        # (batch, M, d_internal) - 作为query
            key=V_proj,           # (batch, N, d_internal) - vision作为key
            value=V_proj,         # (batch, N, d_internal) - vision作为value
            need_weights=True,
            average_attn_weights=True  # 平均所有头的权重
        )  # pooled: (batch, M, d_internal), attn_weights: (batch, M, N)

        # === Step 6: Post-Attention归一化 ===
        pooled = self.post_attn_norm(pooled)  # (batch, M, d_internal)

        # === Step 7: FFN + 残差连接 ===
        ffn_out = self.ffn(pooled)
        pooled = self.post_ffn_norm(pooled + ffn_out)  # 残差 + 归一化

        # === Step 8: 投影回vision维度 ===
        transformed = self.output_proj(pooled)  # (batch, M, d_vision)

        # === Step 9: 残差连接 - 直接从原始vision聚合 ===
        # 使用 attention weights 从原始 vision_features 聚合（跳过所有变换）
        # attn_weights: (batch, M, N), vision_features: (batch, N, d_vision)
        # 结果: (batch, M, d_vision) - 原始特征的加权平均
        residual = torch.bmm(attn_weights, vision_features)  # 直接聚合原始特征

        # 合并：transformed（学习的变换）+ residual（原始信息）
        # 初始化时 transformed ≈ 0，所以输出 ≈ residual（原始特征的聚合）
        merged_features = self.residual_scale * transformed + residual

        # 不做 output_norm，保持原始 CLIP 特征分布

        # === Step 10: 返回结果（兼容原有接口） ===
        return {
            'merged_features': merged_features,
            'merge_indices': None,  # 无显式索引（软池化）
            'merge_weights': attn_weights,  # (batch, M, N) - 权重矩阵
            'importance_logits': None  # 无显式重要性分数
        }

    def set_temperature(self, temperature: float):
        """设置temperature（保留接口兼容性）"""
        self.temperature = temperature
