"""Vision Token Pruning - Unified Model for HF Trainer

将整个训练流程封装到一个模型中，支持HF Trainer + FSDP多卡训练。

核心设计:
1. 所有组件(backbone, token_merger, layer_pruners, discriminator)封装在一个模型中
2. 训练逻辑从train_step移到模型的forward中
3. 支持单个optimizer + parameter groups（不同学习率）
4. GAN训练逻辑通过控制requires_grad实现
5. 返回单个总loss给HF Trainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from typing import Dict, Any, List, Optional, Union
from collections import defaultdict
from PIL import Image

from .pruning_wrapper import create_llava_with_pruning, LLaVAWithPruning
from ..utils import (
    weighted_pool_text_hidden_states,
    add_position_aware_noise_to_pooled,
    update_temperature_for_all,
    get_current_sparsity_weight,
    replace_vision_tokens_in_embeddings,
    extract_text_hidden_states,
    compute_task_loss
)


class VisionTokenPruningModel(nn.Module):
    """Vision Token Pruning统一模型

    封装所有组件和训练逻辑,支持HF Trainer。

    组件:
        - backbone: LLaVAWithPruning (带集成pruning的LLaVA)
        - token_merger: TokenMerger (可选,LLM输入前合并tokens)
        - layer_pruners: LayerSpecificPruner (LLM内部分层剪枝)
        - discriminator: Discriminator (GAN判别器)

    训练流程:
        1. Preprocess: 处理images/questions/answers
        2. Token Merge: (可选) 合并vision tokens
        3. Fake Forward: 带pruning的forward
        4. Real Forward: 不带pruning的forward
        5. Discriminator: 判别real vs fake
        6. Loss: 综合loss计算

    参数:
        config: 配置对象
        backbone: 原始backbone模型
        token_merger: TokenMerger实例（可选）
        layer_pruners: LayerSpecificPruner实例
        discriminator: Discriminator实例
    """

    def __init__(
        self,
        config: Any,
        backbone: nn.Module,
        layer_pruners: nn.Module,
        discriminator: nn.Module,
        token_merger: Optional[nn.Module] = None
    ):
        super().__init__()

        self.config = config
        self.method_config = config.method_settings if hasattr(config, 'method_settings') else config['method_settings']

        # 创建带pruning的backbone
        self.backbone = create_llava_with_pruning(backbone, layer_pruners, config)

        # 保存原始backbone（用于real forward）
        self.backbone_original = backbone

        # 其他组件
        self.token_merger = token_merger
        self.layer_pruners = layer_pruners
        self.discriminator = discriminator

        # 训练状态
        self.global_step = 0
        self.total_steps = 1000  # 会被trainer更新

        # 配置
        self.enable_token_merger = self.method_config.get('enable_token_merger', False)
        self.disc_target_layers = self.method_config['disc_target_layers']
        self.amp_enabled = self.method_config.get('amp_enabled', False)
        amp_dtype_str = self.method_config.get('amp_dtype', 'bfloat16')
        self.amp_dtype = torch.float16 if amp_dtype_str == 'float16' else torch.bfloat16

        print(f"✓ VisionTokenPruningModel initialized")
        print(f"  - Token Merger: {'Enabled' if self.enable_token_merger else 'Disabled'}")
        print(f"  - Disc Target Layers: {self.disc_target_layers}")
        print(f"  - AMP: {self.amp_enabled} ({amp_dtype_str})")

    def _load_images(self, images: List[Union[str, Image.Image]]) -> List[Image.Image]:
        """加载图片（如果是文件路径则打开）

        参数:
            images: 图片路径列表或PIL Image列表

        返回:
            PIL Image列表
        """
        loaded_images = []
        for img in images:
            if isinstance(img, str):
                # 文件路径，需要加载
                loaded_images.append(Image.open(img).convert('RGB'))
            elif isinstance(img, Image.Image):
                # 已经是PIL Image
                loaded_images.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        return loaded_images

    def forward(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """统一的forward pass

        参数:
            batch: List of samples, each with 'image', 'question', 'answer'

        返回:
            Dict with:
                - loss: 总loss (required by HF Trainer)
                - **metrics: 各种统计指标
        """
        device = next(self.parameters()).device

        # 提取batch数据
        images = [sample["image"] for sample in batch]
        questions = [sample["question"] for sample in batch]
        answers = [sample["answer"] for sample in batch]

        # 加载图片（如果是文件路径）
        images = self._load_images(images)

        # ===== Phase 1: Preprocess =====
        with torch.no_grad():
            emb_info = self.backbone_original.preprocess_batch(images, questions, answers)

        original_embeddings = emb_info['embeddings']
        original_vision_pos = emb_info['vision_token_positions']
        answer_pos = emb_info['answer_token_positions']
        vision_features_raw = emb_info['raw_vision_features']

        # ===== Phase 2: Token Merge (optional) =====
        with autocast('cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
            if self.enable_token_merger and self.token_merger is not None:
                merged_embeddings, new_vision_pos, new_attention_mask = \
                    self._apply_token_merge(emb_info, vision_features_raw, original_embeddings, original_vision_pos)
            else:
                # 不使用merger，直接投影vision features
                vision_projected = self.backbone_original.model.multi_modal_projector(vision_features_raw)
                merged_embeddings, new_vision_pos, new_attention_mask = replace_vision_tokens_in_embeddings(
                    original_embeddings,
                    original_vision_pos,
                    vision_projected,
                    emb_info['attention_mask']
                )

        # 提取question embeddings (用于pruners)
        question_embeddings = self._extract_question_embeddings(
            merged_embeddings, new_vision_pos, answer_pos, original_vision_pos
        )

        # ===== Phase 3: Fake Forward (with pruning) =====
        mask_collector = []
        self.backbone.enable_pruning(
            new_vision_pos,
            question_embeddings,
            use_attn_residual=self.method_config.get('use_attn_residual', False),
            mask_collector=mask_collector
        )

        try:
            with autocast('cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
                result_fake = self.backbone(
                    inputs_embeds=merged_embeddings,
                    attention_mask=new_attention_mask,
                    output_hidden_states=True
                )
        finally:
            self.backbone.disable_pruning()

        # 提取fake hidden states
        fake_hidden_list = [
            extract_text_hidden_states(result_fake.hidden_states[layer_idx], new_vision_pos)
            for layer_idx in self.disc_target_layers
        ]

        # ===== Phase 4: Real Forward (no pruning) =====
        with torch.no_grad():
            with autocast('cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
                result_real = self.backbone_original(
                    inputs_embeds=original_embeddings,
                    attention_mask=emb_info['attention_mask'],
                    output_hidden_states=True
                )

        # 提取real hidden states
        real_hidden_list = [
            extract_text_hidden_states(result_real.hidden_states[layer_idx], original_vision_pos)
            for layer_idx in self.disc_target_layers
        ]

        # ===== Phase 5: Discriminator =====
        disc_loss, gen_adv_loss, metrics = self._compute_discriminator_losses(
            fake_hidden_list, real_hidden_list
        )

        # ===== Phase 6: Task Loss =====
        task_loss = compute_task_loss(
            result_fake.logits,
            answer_pos,
            answers,
            self.backbone_original.processor
        )

        # ===== Phase 7: Sparsity Loss =====
        sparsity_loss, sparsity_metrics = self._compute_sparsity_loss(mask_collector, device)
        metrics.update(sparsity_metrics)

        # ===== Phase 8: 组合Loss =====
        total_loss, loss_weights = self._combine_losses(
            task_loss, gen_adv_loss, disc_loss, sparsity_loss
        )

        # 更新metrics
        metrics.update({
            'raw_task_loss': task_loss.item(),
            'raw_gen_adv_loss': gen_adv_loss.item(),
            'raw_disc_loss': disc_loss.item(),
            **loss_weights
        })

        # 清理显存
        self._cleanup(merged_embeddings, result_fake, result_real,
                      fake_hidden_list, real_hidden_list, mask_collector, batch)

        # 返回给HF Trainer
        return {
            'loss': total_loss,
            **metrics  # 其他指标会被HF Trainer记录
        }

    def _apply_token_merge(self, emb_info, vision_features_raw, original_embeddings, original_vision_pos):
        """应用token merge"""
        # 提取question embeddings for merger
        v_start, v_end = original_vision_pos[0, 0].item(), original_vision_pos[0, 1].item()
        answer_start_abs = emb_info['answer_token_positions'][0, 0].item()
        if answer_start_abs < 0:
            answer_start_abs = original_embeddings.shape[1] + answer_start_abs

        question_emb_for_merger = original_embeddings[:, v_end+1:answer_start_abs, :]

        # Merge
        self.token_merger.train()
        merger_type = self.config.method_settings.merger_type if hasattr(self.config.method_settings, 'merger_type') \
            else self.config['method_settings']['merger_type']

        if merger_type in ["question_aware", "fixed_pooling"]:
            merge_result = self.token_merger(vision_features_raw, question_emb_for_merger, use_gumbel=True)
        else:
            merge_result = self.token_merger(vision_features_raw, use_gumbel=True)

        merged_vision = merge_result['merged_features']

        # 投影
        merged_vision = self.backbone_original.model.multi_modal_projector(merged_vision)

        # 替换
        return replace_vision_tokens_in_embeddings(
            original_embeddings,
            original_vision_pos,
            merged_vision,
            emb_info['attention_mask']
        )

    def _extract_question_embeddings(self, merged_embeddings, new_vision_pos, answer_pos, original_vision_pos):
        """提取question embeddings"""
        num_removed_tokens = (original_vision_pos[0, 1] - original_vision_pos[0, 0] + 1) - \
                             (new_vision_pos[0, 1] - new_vision_pos[0, 0] + 1)
        answer_start_abs = answer_pos[0, 0].item()
        if answer_start_abs < 0:
            answer_start_abs = merged_embeddings.shape[1] + answer_start_abs
        answer_start_merged = answer_start_abs - num_removed_tokens.item()

        return merged_embeddings[:, new_vision_pos[0, 1].item()+1:answer_start_merged, :]

    def _compute_discriminator_losses(self, fake_hidden_list, real_hidden_list):
        """计算discriminator相关losses"""
        # Pool hidden states
        pool_config = {
            'start_weight': self.method_config.get('disc_pool_start_weight', 0.4),
            'end_weight': self.method_config.get('disc_pool_end_weight', 1.0),
            'noise_scale_start': self.method_config.get('disc_noise_scale_start', 0.05),
            'noise_scale_end': self.method_config.get('disc_noise_scale_end', 0.01)
        }

        fake_pooled = weighted_pool_text_hidden_states(fake_hidden_list, training=True, **pool_config)
        real_pooled = weighted_pool_text_hidden_states(real_hidden_list, training=True, **pool_config)

        # Add noise
        noise_scale = self.method_config.get('disc_noise_scale', 0.0)
        fake_pooled = add_position_aware_noise_to_pooled(fake_pooled, noise_scale, training=True)
        real_pooled = add_position_aware_noise_to_pooled(real_pooled, noise_scale, training=True)

        # Prepare for discriminator
        fake_for_disc = [h.unsqueeze(1) for h in fake_pooled]
        real_for_disc = [h.unsqueeze(1) for h in real_pooled]

        # Generator adversarial loss (discriminator frozen)
        self._freeze_discriminator()
        with autocast('cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
            fake_pred_for_gen = self.discriminator(fake_for_disc)
        self._unfreeze_discriminator()

        gen_adv_loss = F.binary_cross_entropy(
            fake_pred_for_gen, torch.ones_like(fake_pred_for_gen), reduction='mean'
        )

        # Discriminator loss
        self.discriminator.train()
        with autocast('cuda', enabled=self.amp_enabled, dtype=self.amp_dtype):
            real_pred = self.discriminator(real_for_disc)
            fake_pred_for_disc = self.discriminator([h.detach() for h in fake_for_disc])

        disc_real_loss = F.binary_cross_entropy(real_pred, torch.ones_like(real_pred), reduction='mean')
        disc_fake_loss = F.binary_cross_entropy(fake_pred_for_disc, torch.zeros_like(fake_pred_for_disc), reduction='mean')
        disc_loss = disc_real_loss + disc_fake_loss

        # Metrics
        metrics = {
            'disc_real_acc': (real_pred > 0.5).float().mean().item(),
            'disc_fake_acc': (fake_pred_for_disc < 0.5).float().mean().item()
        }

        return disc_loss, gen_adv_loss, metrics

    def _compute_sparsity_loss(self, mask_collector, device):
        """计算sparsity loss"""
        if len(mask_collector) == 0:
            return torch.tensor(0.0, device=device), {}

        use_token_num_target = self.method_config.get('use_token_num_target', True)
        sparsity_loss_only_on_excess = self.method_config.get('sparsity_loss_only_on_excess', False)

        n_vision = mask_collector[0].shape[1]

        if use_token_num_target:
            target_avg_token_num = self.method_config.get('target_token_num', 128)
        else:
            target_sparsity = self.method_config.get('target_sparsity', 0.5)
            target_avg_token_num = n_vision * (1.0 - target_sparsity)

        # 每层的保留率
        kept_ratios = [mask.mean().to(device) for mask in mask_collector]
        tokens_per_layer = [n_vision * ratio for ratio in kept_ratios]
        avg_tokens = torch.stack(tokens_per_layer).mean()

        # Sparsity loss
        if sparsity_loss_only_on_excess:
            excess = torch.relu(avg_tokens - target_avg_token_num)
            sparsity_loss = excess.pow(2)
        else:
            sparsity_loss = (avg_tokens - target_avg_token_num).pow(2)

        # Binarization loss
        binarization_loss = torch.tensor(0.0, device=device)
        for mask in mask_collector:
            binary_term = (mask * (1 - mask)).mean()
            variance_term = mask.var()
            binarization_loss = binarization_loss + (binary_term - 0.5 * variance_term)
        binarization_loss = binarization_loss / len(mask_collector)

        # Metrics
        metrics = {
            'avg_tokens': avg_tokens.item(),
            'target_avg_tokens': target_avg_token_num,
            'raw_sparsity_loss': sparsity_loss.item(),
            'raw_binarization_loss': binarization_loss.item()
        }

        # 添加每层的统计
        pruning_layers = self.layer_pruners.get_all_layers()
        for idx, (mask, tokens) in enumerate(zip(mask_collector, tokens_per_layer)):
            layer_num = pruning_layers[idx]
            metrics[f'L{layer_num}_kept_ratio'] = mask.mean().item()
            metrics[f'L{layer_num}_tokens'] = tokens.item()

        return sparsity_loss, metrics

    def _combine_losses(self, task_loss, gen_adv_loss, disc_loss, sparsity_loss):
        """组合所有losses"""
        # 动态权重
        progress = self.global_step / max(self.total_steps, 1)

        task_weight = self._get_loss_weight('task_loss_weight', progress)
        adv_weight = self._get_loss_weight('adv_loss_weight', progress)
        sparsity_weight = get_current_sparsity_weight(self.config, self.global_step, self.total_steps)
        binarization_weight = self.method_config.get('binarization_loss_weight', 0.0)

        # 组合
        total_loss = (
            task_weight * task_loss +
            adv_weight * gen_adv_loss +
            sparsity_weight * sparsity_loss +
            disc_loss  # discriminator loss不加权
        )

        weights = {
            'current_task_weight': task_weight,
            'current_adv_weight': adv_weight,
            'current_sparsity_weight': sparsity_weight
        }

        return total_loss, weights

    def _get_loss_weight(self, weight_name, progress):
        """获取动态loss权重（支持warmup）"""
        weight_end = self.method_config.get(weight_name)
        weight_start = self.method_config.get(f'{weight_name}_start', None)
        warmup_ratio = self.method_config.get('loss_weight_warmup_ratio', 0.0)

        if warmup_ratio > 0 and progress < warmup_ratio and weight_start is not None:
            warmup_progress = progress / warmup_ratio
            cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2
            return weight_start + (weight_end - weight_start) * cosine_factor
        else:
            return weight_end

    def _freeze_discriminator(self):
        """Freeze discriminator (for generator training)"""
        for p in self.discriminator.parameters():
            p.requires_grad = False

    def _unfreeze_discriminator(self):
        """Unfreeze discriminator"""
        for p in self.discriminator.parameters():
            p.requires_grad = True

    def _cleanup(self, *tensors_and_lists):
        """清理显存"""
        for item in tensors_and_lists:
            del item
        torch.cuda.empty_cache()

    def set_training_progress(self, global_step: int, total_steps: int):
        """由trainer调用,更新训练进度"""
        self.global_step = global_step
        self.total_steps = total_steps

        # 更新temperature
        if self.token_merger is not None and self.enable_token_merger:
            update_temperature_for_all(
                self.token_merger, self.layer_pruners,
                self.config, global_step, total_steps
            )
        else:
            # 只更新layer pruners
            self._update_pruners_temperature(global_step, total_steps)

    def _update_pruners_temperature(self, current_step, total_steps):
        """更新pruners的temperature"""
        temperature = self.method_config.get('temperature', 1.0)
        temperature_min = self.method_config.get('temperature_min', 0.1)
        anneal_rate = self.method_config.get('temperature_anneal_rate', 0.5)

        progress = current_step / total_steps
        if progress < anneal_rate:
            current_temp = temperature - (progress / anneal_rate) * (temperature - temperature_min)
        else:
            current_temp = temperature_min

        for layer_idx in self.layer_pruners.get_all_layers():
            pruner = self.layer_pruners.get_pruner(layer_idx)
            pruner.set_temperature(current_temp)

    def create_optimizer_groups(self) -> List[Dict[str, Any]]:
        """创建parameter groups（不同组件用不同学习率）

        返回:
            List of dicts for torch.optim.Optimizer
        """
        groups = []

        # Token Merger (如果启用)
        if self.enable_token_merger and self.token_merger is not None:
            merger_lr = self.config.trainer_settings.dl_settings.optimizers.token_merger.lr \
                if hasattr(self.config.trainer_settings.dl_settings.optimizers.token_merger, 'lr') \
                else self.config['trainer_settings']['dl_settings']['optimizers']['token_merger']['lr']
            groups.append({
                'params': self.token_merger.parameters(),
                'lr': merger_lr
            })

        # Layer Pruners
        pruner_lr = self.config.trainer_settings.dl_settings.optimizers.layer_pruners.lr \
            if hasattr(self.config.trainer_settings.dl_settings.optimizers.layer_pruners, 'lr') \
            else self.config['trainer_settings']['dl_settings']['optimizers']['layer_pruners']['lr']
        groups.append({
            'params': self.layer_pruners.parameters(),
            'lr': pruner_lr
        })

        # Discriminator
        disc_lr = self.config.trainer_settings.dl_settings.optimizers.discriminator.lr \
            if hasattr(self.config.trainer_settings.dl_settings.optimizers.discriminator, 'lr') \
            else self.config['trainer_settings']['dl_settings']['optimizers']['discriminator']['lr']
        groups.append({
            'params': self.discriminator.parameters(),
            'lr': disc_lr
        })

        print(f"✓ Created optimizer groups:")
        for i, group in enumerate(groups):
            print(f"  Group {i}: {len(list(group['params']))} params, lr={group['lr']}")

        return groups
