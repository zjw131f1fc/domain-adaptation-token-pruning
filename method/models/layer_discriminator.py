"""Attention Consistency Pruning - Layer Discriminator

判别单个 answer token 的 attention 聚合结果是 real 还是 fake。

设计理念：
- 每个 answer token 独立判别（粒度更细，信号更强）
- 输入是 attention 聚合后的结果 h = Σ attn[i] * V[i]
- 不是判别 hidden states，而是判别 attention 聚合结果
- 轻量级网络，避免判别器过强
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LayerDiscriminator(nn.Module):
    """单层 Attention 聚合结果判别器

    判别单个 answer token 从前面 positions 聚合的 h 是 real 还是 fake。

    参数:
        num_heads: attention 头数
        head_dim: 每个头的维度
        d_hidden: MLP 隐藏层维度
        dropout: Dropout 比例
        use_spectral_norm: 是否使用谱归一化（提高训练稳定性）
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        d_hidden: int = 256,
        dropout: float = 0.1,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.input_dim = num_heads * head_dim  # 32 * 128 = 4096

        # 简化的网络结构: 4096 -> 512 -> 256 -> 1
        linear1 = nn.Linear(self.input_dim, 512)
        linear2 = nn.Linear(512, d_hidden)
        linear_out = nn.Linear(d_hidden, 1)

        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
            linear2 = nn.utils.spectral_norm(linear2)
            linear_out = nn.utils.spectral_norm(linear_out)

        self.net = nn.Sequential(
            linear1,
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            linear2,
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            linear_out
        )

    def reset_parameters(self):
        """重新初始化网络参数（当判别器过强时调用）"""
        for module in self.net.modules():
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
            h_flat = h.view(h.shape[0], -1)  # (batch, heads * head_dim)
            return self.net(h_flat).squeeze(-1)  # (batch,)
        elif h.dim() == 4:
            # 多个 answer tokens: (batch, heads, n_ans, head_dim)
            batch, heads, n_ans, head_dim = h.shape
            # Reshape: (batch, heads, n_ans, head_dim) -> (batch, n_ans, heads, head_dim)
            h = h.permute(0, 2, 1, 3)
            # Flatten: (batch, n_ans, heads * head_dim)
            h_flat = h.reshape(batch, n_ans, -1)
            # 对每个 answer token 判别: (batch, n_ans, 1) -> (batch, n_ans)
            return self.net(h_flat).squeeze(-1)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got {h.dim()}D")

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
        d_hidden: MLP 隐藏层维度
        dropout: Dropout 比例
        use_spectral_norm: 是否使用谱归一化
    """

    def __init__(
        self,
        layer_indices: list,
        num_heads: int,
        head_dim: int,
        d_hidden: int = 256,
        dropout: float = 0.1,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        self.layer_indices = layer_indices

        # 为每层创建独立的判别器
        self.discriminators = nn.ModuleDict({
            str(idx): LayerDiscriminator(
                num_heads=num_heads,
                head_dim=head_dim,
                d_hidden=d_hidden,
                dropout=dropout,
                use_spectral_norm=use_spectral_norm
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
        """重新初始化所有判别器的参数（当判别器过强时调用）"""
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
        loss_type: str = 'bce',
        gp_weight: float = 10.0,
    ) -> torch.Tensor:
        """计算所有层的判别器损失

        参数:
            h_real_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]} - 每层每个样本的真实聚合结果
            h_fake_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]} - 每层每个样本的剪枝后聚合结果
            loss_type: 损失类型 ('bce', 'wgan', 'hinge')
            gp_weight: WGAN-GP 梯度惩罚权重

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
                h_fake = h_fake.unsqueeze(0).detach()

                real_pred = disc.forward_batch_answers(h_real, reduce='mean')
                fake_pred = disc.forward_batch_answers(h_fake, reduce='mean')

                if loss_type == 'wgan':
                    # WGAN loss: max E[D(real)] - E[D(fake)]
                    wgan_loss = -real_pred.mean() + fake_pred.mean()
                    gp = self._gradient_penalty(disc, h_real, h_fake)
                    total_loss = total_loss + wgan_loss + gp_weight * gp
                elif loss_type == 'hinge':
                    # Hinge loss: E[max(0, 1-D(real))] + E[max(0, 1+D(fake))]
                    real_loss = F.relu(1.0 - real_pred).mean()
                    fake_loss = F.relu(1.0 + fake_pred).mean()
                    total_loss = total_loss + real_loss + fake_loss
                else:
                    # 标准 GAN loss (BCE)
                    real_loss = F.binary_cross_entropy_with_logits(
                        real_pred, torch.ones_like(real_pred)
                    )
                    fake_loss = F.binary_cross_entropy_with_logits(
                        fake_pred, torch.zeros_like(fake_pred)
                    )
                    total_loss = total_loss + real_loss + fake_loss

        # 除以样本数和层数
        n_samples = len(h_real_list)
        n_layers = len(self.layer_indices)
        return total_loss / (n_samples * n_layers)

    def _gradient_penalty(
        self,
        disc: LayerDiscriminator,
        h_real: torch.Tensor,
        h_fake: torch.Tensor,
    ) -> torch.Tensor:
        """计算 WGAN-GP 的梯度惩罚

        参数:
            disc: 判别器
            h_real: (1, heads, n_ans, head_dim) - 真实样本
            h_fake: (1, heads, n_ans, head_dim) - 假样本

        返回:
            gradient_penalty: 梯度惩罚值
        """
        batch_size = h_real.size(0)
        # 随机插值系数
        alpha = torch.rand(batch_size, 1, 1, 1, device=h_real.device, dtype=h_real.dtype)
        # 插值样本
        interpolated = alpha * h_real + (1 - alpha) * h_fake
        interpolated.requires_grad_(True)

        # 判别器输出
        d_interpolated = disc.forward_batch_answers(interpolated, reduce='mean')

        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
        )[0]

        # 梯度范数
        gradients = gradients.reshape(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        # 惩罚项: (||grad|| - 1)^2
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

    def compute_adv_loss(
        self,
        h_fake_dict: dict,
        loss_type: str = 'bce',
    ) -> torch.Tensor:
        """计算所有层的对抗损失（用于训练 Pruner）

        参数:
            h_fake_dict: {layer_idx: List[(heads, n_ans_i, head_dim)]} - 每层每个样本的剪枝后聚合结果
            loss_type: 损失类型 ('bce', 'wgan', 'hinge')

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

                fake_pred = disc.forward_batch_answers(h_fake, reduce='mean')
                if loss_type in ('wgan', 'hinge'):
                    # WGAN/Hinge: Pruner 的目标是最大化 D(fake)，即最小化 -D(fake)
                    adv_loss = -fake_pred.mean()
                else:
                    # 标准 GAN: Pruner 的目标是让 fake 被判为 real (1)
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
