"""Attention Consistency Pruning - Layer Discriminator

判别单个 answer token 的 attention 聚合结果是 real 还是 fake。

设计理念 v2：
- 保留 head 结构信息，不直接 flatten
- Per-head 特征提取 + head 间交互 + 加权聚合
- 每个 answer token 独立判别
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class LayerDiscriminator(nn.Module):
    """单层 Attention 聚合结果判别器 (v3: 保留 head 结构 + concat)

    判别单个 answer token 从前面 positions 聚合的 h 是 real 还是 fake。

    结构：
    1. Per-head projection: (heads, head_dim) -> (heads, d_head)
    2. Head interaction: self-attention 学习 head 间关系
    3. Concat: 拼接所有 head 的特征（保留完整信息）
    4. Output MLP: 最终判别

    参数:
        num_heads: attention 头数
        head_dim: 每个头的维度
        d_head: per-head 投影后的维度
        d_hidden: 输出 MLP 的隐藏层维度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        d_head: int = 64,
        d_hidden: int = 128,
        dropout: float = 0.1,
        use_spectral_norm: bool = False  # 保留参数兼容性
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.d_head = d_head

        # 1. Per-head projection (参数共享)
        self.head_proj = nn.Sequential(
            nn.Linear(head_dim, d_head),
            nn.LayerNorm(d_head),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. Head interaction: 简单的 self-attention
        self.head_attn = nn.MultiheadAttention(
            embed_dim=d_head,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.head_norm = nn.LayerNorm(d_head)

        # 3. Output MLP (输入是 concat 后的 num_heads * d_head 维)
        concat_dim = num_heads * d_head
        self.output_mlp = nn.Sequential(
            nn.Linear(concat_dim, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1)
        )

    def reset_parameters(self):
        """重新初始化网络参数"""
        # Reset head projection
        for module in self.head_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Reset head attention
        nn.init.xavier_uniform_(self.head_attn.in_proj_weight)
        nn.init.xavier_uniform_(self.head_attn.out_proj.weight)
        nn.init.zeros_(self.head_attn.in_proj_bias)
        nn.init.zeros_(self.head_attn.out_proj.bias)
        nn.init.ones_(self.head_norm.weight)
        nn.init.zeros_(self.head_norm.bias)

        # Reset output MLP
        for module in self.output_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """判别 attention 聚合结果

        参数:
            h: (batch, heads, head_dim) - 单个 answer token 的聚合结果
               或 (batch, heads, n_ans, head_dim) - 多个 answer tokens

        返回:
            logit: (batch,) 或 (batch, n_ans) - real/fake 判断的 logit
        """
        if h.dim() == 3:
            # 单个 answer token: (batch, heads, head_dim)
            return self._forward_single(h)
        elif h.dim() == 4:
            # 多个 answer tokens: (batch, heads, n_ans, head_dim)
            batch, heads, n_ans, head_dim = h.shape
            # 转换为 (batch * n_ans, heads, head_dim)
            h_reshaped = h.permute(0, 2, 1, 3).reshape(batch * n_ans, heads, head_dim)
            logits = self._forward_single(h_reshaped)  # (batch * n_ans,)
            return logits.view(batch, n_ans)  # (batch, n_ans)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {h.dim()}D")

    def _forward_single(self, h: torch.Tensor) -> torch.Tensor:
        """处理单个 answer token

        参数:
            h: (batch, heads, head_dim)

        返回:
            logit: (batch,)
        """
        batch_size = h.shape[0]

        # 1. Per-head projection: (batch, heads, head_dim) -> (batch, heads, d_head)
        h_proj = self.head_proj(h)

        # 2. Head interaction: self-attention
        # (batch, heads, d_head) as sequence of heads
        h_attn, _ = self.head_attn(h_proj, h_proj, h_proj)
        h_attn = self.head_norm(h_attn + h_proj)  # residual connection

        # 3. Concat all heads: (batch, heads, d_head) -> (batch, heads * d_head)
        h_concat = h_attn.flatten(1)

        # 4. Output MLP
        logit = self.output_mlp(h_concat).squeeze(-1)  # (batch,)

        return logit

    def forward_batch_answers(
        self,
        h: torch.Tensor,
        reduce: str = 'mean'
    ) -> torch.Tensor:
        """对多个 answer tokens 的聚合结果进行判别

        参数:
            h: (batch, heads, n_ans, head_dim) - 多个 answer tokens 的聚合结果
            reduce: 如何聚合多个 answer tokens 的判别结果
                   'mean': 平均
                   'sum': 求和
                   'none': 不聚合

        返回:
            logit: (batch,) 或 (batch, n_ans) - 判别结果
        """
        logits = self.forward(h)  # (batch, n_ans)

        if reduce == 'mean':
            return logits.mean(dim=-1)  # (batch,)
        elif reduce == 'sum':
            return logits.sum(dim=-1)  # (batch,)
        elif reduce == 'none':
            return logits  # (batch, n_ans)
        else:
            raise ValueError(f"Unknown reduce mode: {reduce}")


class LayerDiscriminatorManager(nn.Module):
    """多层判别器管理器

    管理多个层的 LayerDiscriminator，提供统一接口。

    参数:
        layer_indices: 要判别的层索引列表（与 pruning_layers 相同）
        num_heads: attention 头数
        head_dim: 每个头的维度
        d_head: per-head 投影后的维度
        d_hidden: 输出 MLP 的隐藏层维度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        layer_indices: list,
        num_heads: int,
        head_dim: int,
        d_head: int = 64,
        d_hidden: int = 128,
        dropout: float = 0.1,
        use_spectral_norm: bool = False  # 保留参数兼容性
    ):
        super().__init__()
        self.layer_indices = layer_indices

        # 为每层创建独立的判别器
        self.discriminators = nn.ModuleDict({
            str(idx): LayerDiscriminator(
                num_heads=num_heads,
                head_dim=head_dim,
                d_head=d_head,
                d_hidden=d_hidden,
                dropout=dropout
            )
            for idx in layer_indices
        })

    def get_discriminator(self, layer_idx: int) -> LayerDiscriminator:
        """获取指定层的判别器"""
        key = str(layer_idx)
        if key not in self.discriminators:
            raise ValueError(f"No discriminator for layer {layer_idx}. Available: {self.layer_indices}")
        return self.discriminators[key]

    def reinit_all(self):
        """重新初始化所有判别器的参数"""
        for disc in self.discriminators.values():
            disc.reset_parameters()

    def reinit_layer(self, layer_idx: int):
        """重新初始化指定层的判别器参数"""
        key = str(layer_idx)
        if key in self.discriminators:
            self.discriminators[key].reset_parameters()

    def compute_disc_loss(
        self,
        h_real_dict: dict,
        h_fake_dict: dict,
    ) -> torch.Tensor:
        """计算所有层的判别器损失

        参数:
            h_real_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]} - 每层每个样本的真实聚合结果
            h_fake_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]} - 每层每个样本的剪枝后聚合结果

        返回:
            disc_loss: 判别器总损失
        """
        total_loss = 0
        for layer_idx in self.layer_indices:
            key = str(layer_idx)
            h_real_list = h_real_dict[layer_idx]
            h_fake_list = h_fake_dict[layer_idx]

            disc = self.discriminators[key]

            # 逐样本计算
            for h_real, h_fake in zip(h_real_list, h_fake_list):
                # h_real/h_fake: (heads, n_ans, head_dim) -> (1, heads, n_ans, head_dim)
                h_real = h_real.unsqueeze(0)
                h_fake = h_fake.unsqueeze(0)

                # 真实样本应该被判为 1
                real_pred = disc.forward_batch_answers(h_real, reduce='mean')
                real_loss = F.binary_cross_entropy_with_logits(
                    real_pred, torch.ones_like(real_pred)
                )

                # 假样本应该被判为 0（注意：h_fake 要 detach）
                fake_pred = disc.forward_batch_answers(h_fake.detach(), reduce='mean')
                fake_loss = F.binary_cross_entropy_with_logits(
                    fake_pred, torch.zeros_like(fake_pred)
                )

                total_loss = total_loss + real_loss + fake_loss

        # 除以样本数和层数
        n_samples = len(h_real_list)
        n_layers = len(self.layer_indices)
        return total_loss / (n_samples * n_layers)

    def compute_adv_loss(
        self,
        h_fake_dict: dict,
    ) -> torch.Tensor:
        """计算所有层的对抗损失（用于训练 Pruner）

        参数:
            h_fake_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]} - 每层每个样本的剪枝后聚合结果

        返回:
            adv_loss: 对抗损失
        """
        total_loss = 0
        n_samples = 0
        for layer_idx in self.layer_indices:
            key = str(layer_idx)
            h_fake_list = h_fake_dict[layer_idx]

            disc = self.discriminators[key]

            # 逐样本计算
            for h_fake in h_fake_list:
                # h_fake: (heads, n_ans, head_dim) -> (1, heads, n_ans, head_dim)
                h_fake = h_fake.unsqueeze(0)

                # Pruner 的目标：让 fake 被判为 real (1)
                fake_pred = disc.forward_batch_answers(h_fake, reduce='mean')
                adv_loss = F.binary_cross_entropy_with_logits(
                    fake_pred, torch.ones_like(fake_pred)
                )

                total_loss = total_loss + adv_loss

            n_samples = len(h_fake_list)

        # 除以样本数和层数
        n_layers = len(self.layer_indices)
        return total_loss / (n_samples * n_layers)

    def compute_accuracy(
        self,
        h_real_dict: dict,
        h_fake_dict: dict,
    ) -> dict:
        """计算判别器准确率（用于监控）

        参数:
            h_real_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]}
            h_fake_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]}

        返回:
            accuracy_dict: {
                'overall': float,
                'real_acc': float,
                'fake_acc': float,
                'per_layer': {layer_idx: (real_acc, fake_acc)}
            }
        """
        all_real_correct = 0
        all_fake_correct = 0
        all_real_total = 0
        all_fake_total = 0

        per_layer = {}

        with torch.no_grad():
            for layer_idx in self.layer_indices:
                key = str(layer_idx)
                h_real_list = h_real_dict[layer_idx]
                h_fake_list = h_fake_dict[layer_idx]

                disc = self.discriminators[key]

                layer_real_correct = 0
                layer_fake_correct = 0
                layer_real_total = 0
                layer_fake_total = 0

                # 逐样本计算
                for h_real, h_fake in zip(h_real_list, h_fake_list):
                    h_real = h_real.unsqueeze(0)
                    h_fake = h_fake.unsqueeze(0)

                    real_pred = disc.forward_batch_answers(h_real, reduce='mean')
                    fake_pred = disc.forward_batch_answers(h_fake, reduce='mean')

                    layer_real_correct += (real_pred > 0).sum().item()
                    layer_fake_correct += (fake_pred < 0).sum().item()
                    layer_real_total += real_pred.numel()
                    layer_fake_total += fake_pred.numel()

                per_layer[layer_idx] = (
                    layer_real_correct / layer_real_total if layer_real_total > 0 else 0,
                    layer_fake_correct / layer_fake_total if layer_fake_total > 0 else 0
                )

                all_real_correct += layer_real_correct
                all_fake_correct += layer_fake_correct
                all_real_total += layer_real_total
                all_fake_total += layer_fake_total

        real_acc = all_real_correct / all_real_total if all_real_total > 0 else 0
        fake_acc = all_fake_correct / all_fake_total if all_fake_total > 0 else 0
        overall = (all_real_correct + all_fake_correct) / (all_real_total + all_fake_total) \
            if (all_real_total + all_fake_total) > 0 else 0

        return {
            'overall': overall,
            'real_acc': real_acc,
            'fake_acc': fake_acc,
            'per_layer': per_layer
        }
