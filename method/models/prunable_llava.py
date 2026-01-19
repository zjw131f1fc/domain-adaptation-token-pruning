"""Attention Consistency Pruning - Prunable LLaVA Model

可剪枝的 LLaVA 模型，继承自 transformers 的 LlavaForConditionalGeneration。

核心改动：
1. 替换特定层为 PrunableLlamaDecoderLayer
2. 重写 forward 方法以传递剪枝参数和收集剪枝信息
3. 提供训练和推理两种模式
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass

from transformers import LlavaForConditionalGeneration, LlavaConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer

from .layer_pruner_acp import LayerPruner, LayerPrunerManager
from .layer_discriminator import LayerDiscriminator, LayerDiscriminatorManager
from .prunable_llama_layer import PrunableLlamaDecoderLayer


@dataclass
class PrunableLlavaOutput:
    """可剪枝 LLaVA 的输出"""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    image_hidden_states: Optional[torch.Tensor] = None
    # 剪枝信息
    pruning_infos: Optional[Dict[int, Dict]] = None


class PrunableLlavaForConditionalGeneration(nn.Module):
    """可剪枝的 LLaVA 模型

    通过替换特定层的 DecoderLayer 为 PrunableLlamaDecoderLayer 实现剪枝。

    参数:
        base_model: 基础的 LlavaForConditionalGeneration 模型
        pruning_layers: 要剪枝的层索引列表
        pruner_d_internal: Pruner 内部维度
        disc_d_hidden: Discriminator 隐藏层维度
        temperature: 初始 Gumbel-Softmax 温度
        dropout: Dropout 比例
    """

    def __init__(
        self,
        base_model: LlavaForConditionalGeneration,
        pruning_layers: List[int] = [4, 14, 24],
        pruner_d_internal: int = 128,
        pruner_n_heads: int = 4,
        disc_d_hidden: int = 256,
        temperature: float = 1.0,
        dropout: float = 0.1,
        disc_use_spectral_norm: bool = False
    ):
        super().__init__()

        # 保存基础模型
        self.base_model = base_model
        self.config = base_model.config
        self.pruning_layers = pruning_layers

        # 获取 LLM 配置
        llm_config = self.config.text_config
        self.num_heads = llm_config.num_attention_heads
        self.head_dim = llm_config.hidden_size // self.num_heads
        self.hidden_size = llm_config.hidden_size

        # 创建 Pruners（CrossAttentionPruner 需要 d_model）
        self.pruner_manager = LayerPrunerManager(
            layer_indices=pruning_layers,
            d_model=self.hidden_size,
            d_internal=pruner_d_internal,
            n_heads=pruner_n_heads,
            temperature=temperature,
            dropout=dropout
        )

        # 创建 Discriminators
        self.disc_manager = LayerDiscriminatorManager(
            layer_indices=pruning_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            d_hidden=disc_d_hidden,
            dropout=dropout,
            use_spectral_norm=disc_use_spectral_norm
        )

        # 替换剪枝层
        self._replace_pruning_layers()

    def _replace_pruning_layers(self):
        """替换特定层为可剪枝层"""
        llm = self.base_model.model.language_model

        for layer_idx in self.pruning_layers:
            original_layer = llm.layers[layer_idx]
            pruner = self.pruner_manager.get_pruner(layer_idx)
            discriminator = self.disc_manager.get_discriminator(layer_idx)

            # 将 pruner 和 discriminator 移动到与原始层相同的设备和 dtype
            # 注意：直接在原对象上调用 to()，不要重新赋值
            layer_param = next(original_layer.parameters())
            layer_device = layer_param.device
            layer_dtype = layer_param.dtype
            pruner.to(device=layer_device, dtype=layer_dtype)
            discriminator.to(device=layer_device, dtype=layer_dtype)

            llm.layers[layer_idx] = PrunableLlamaDecoderLayer(
                original_layer=original_layer,
                layer_idx=layer_idx,
                pruner=pruner,
                discriminator=discriminator
            )

    def _restore_original_layers(self):
        """还原为原始层（用于保存模型等场景）"""
        llm = self.base_model.model.language_model

        for layer_idx in self.pruning_layers:
            prunable_layer = llm.layers[layer_idx]
            if isinstance(prunable_layer, PrunableLlamaDecoderLayer):
                llm.layers[layer_idx] = prunable_layer.original_layer

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        pruning_layers: List[int] = [4, 14, 24],
        pruner_d_internal: int = 128,
        disc_d_hidden: int = 256,
        temperature: float = 1.0,
        dropout: float = 0.1,
        disc_use_spectral_norm: bool = False,
        **kwargs
    ):
        """从预训练模型创建可剪枝模型

        参数:
            pretrained_model_name_or_path: HuggingFace 模型路径或本地路径
            pruning_layers: 要剪枝的层索引
            其他参数同 __init__
        """
        # 加载基础模型
        base_model = LlavaForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs
        )

        # 创建可剪枝模型
        return cls(
            base_model=base_model,
            pruning_layers=pruning_layers,
            pruner_d_internal=pruner_d_internal,
            disc_d_hidden=disc_d_hidden,
            temperature=temperature,
            dropout=dropout,
            disc_use_spectral_norm=disc_use_spectral_norm
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # === 剪枝参数 ===
        vision_start: Optional[int] = None,
        vision_end: Optional[int] = None,
        question_starts: Optional[list] = None,
        question_ends: Optional[list] = None,
        answer_starts: Optional[list] = None,
        answer_ends: Optional[list] = None,
        return_pruning_info: bool = True,
        **kwargs
    ) -> PrunableLlavaOutput:
        """前向传播

        参数:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            attention_mask: 注意力掩码
            labels: 标签（用于计算 loss）
            vision_start/end: vision tokens 的位置范围
            question_start/end: question tokens 的位置范围
            answer_start/end: answer tokens 的位置范围
            return_pruning_info: 是否返回剪枝信息

        返回:
            PrunableLlavaOutput
        """
        # 获取 image features 并准备 inputs_embeds
        model = self.base_model.model
        llm = model.language_model

        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy

        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # 准备 LLaMA forward 的参数
        batch_size, seq_len, _ = inputs_embeds.shape

        use_cache = kwargs.get('use_cache', False)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=llm.config)

        cache_position = kwargs.get('cache_position', None)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + seq_len, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 创建 causal mask
        causal_mask = create_causal_mask(
            config=llm.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        # Position embeddings
        hidden_states = inputs_embeds
        position_embeddings = llm.rotary_emb(hidden_states, position_ids)

        # === 遍历所有层 ===
        pruning_infos = {}

        for layer_idx, decoder_layer in enumerate(llm.layers):
            if isinstance(decoder_layer, PrunableLlamaDecoderLayer) and return_pruning_info:
                # 剪枝层
                hidden_states, pruning_info = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    vision_start=vision_start,
                    vision_end=vision_end,
                    question_starts=question_starts,
                    question_ends=question_ends,
                    answer_starts=answer_starts,
                    answer_ends=answer_ends,
                    return_pruning_info=True,
                )
                if pruning_info is not None:
                    pruning_infos[layer_idx] = pruning_info
            else:
                # 非剪枝层或不需要返回剪枝信息
                if isinstance(decoder_layer, PrunableLlamaDecoderLayer):
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        vision_start=vision_start,
                        vision_end=vision_end,
                        question_starts=question_starts,
                        question_ends=question_ends,
                        answer_starts=answer_starts,
                        answer_ends=answer_ends,
                        return_pruning_info=False,
                    )
                else:
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

        # Final LayerNorm
        hidden_states = llm.norm(hidden_states)

        # LM Head
        logits = self.base_model.lm_head(hidden_states)

        # 计算 loss
        loss = None
        if labels is not None:
            loss = self.base_model.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size
            )

        return PrunableLlavaOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            image_hidden_states=image_features if pixel_values is not None else None,
            pruning_infos=pruning_infos if return_pruning_info else None,
        )

    def set_temperature(self, temperature: float):
        """设置所有 pruner 的温度"""
        self.pruner_manager.set_temperature(temperature)

    def get_pruner_parameters(self):
        """获取所有 pruner 的参数"""
        return self.pruner_manager.parameters()

    def get_discriminator_parameters(self):
        """获取所有 discriminator 的参数"""
        return self.disc_manager.parameters()

    def freeze_base_model(self):
        """冻结基础模型参数（但保持 pruner 和 discriminator 可训练）"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 重新启用 pruner 和 discriminator 的梯度
        # （因为它们已经被添加到 llm.layers 中，会被上面的循环冻结）
        for param in self.pruner_manager.parameters():
            param.requires_grad = True
        for param in self.disc_manager.parameters():
            param.requires_grad = True

    def unfreeze_base_model(self):
        """解冻基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = True

    def compute_losses(
        self,
        output: PrunableLlavaOutput,
        target_token_num: int,
        n_vision: int,
        task_weight: float = 1.0,
        adv_weight: float = 0.5,
        sparsity_weight: float = 0.2
    ) -> Dict[str, torch.Tensor]:
        """计算所有损失

        参数:
            output: forward 的输出
            target_token_num: 目标保留的 token 数量
            n_vision: 总的 vision token 数量
            task_weight: task loss 权重
            adv_weight: adversarial loss 权重
            sparsity_weight: sparsity loss 权重

        返回:
            losses: {
                'task_loss': tensor,
                'adv_loss': tensor,
                'disc_loss': tensor,
                'sparsity_loss': tensor,
                'pruner_total': tensor,  # task + adv + sparsity
                'kept_ratio': tensor,    # 平均保留率
            }
        """
        losses = {}

        # 1. Task loss（已经在 forward 中计算）
        task_loss = output.loss if output.loss is not None else torch.tensor(0.0)
        losses['task_loss'] = task_loss

        if output.pruning_infos is None or len(output.pruning_infos) == 0:
            # 没有剪枝信息，返回只有 task loss
            losses['adv_loss'] = torch.tensor(0.0)
            losses['disc_loss'] = torch.tensor(0.0)
            losses['sparsity_loss'] = torch.tensor(0.0)
            losses['pruner_total'] = task_loss * task_weight
            losses['kept_ratio'] = torch.tensor(1.0)
            return losses

        # 收集 h_real 和 h_fake
        h_real_dict = {idx: info['h_real'] for idx, info in output.pruning_infos.items()}
        h_fake_dict = {idx: info['h_fake'] for idx, info in output.pruning_infos.items()}

        # 2. Adversarial loss（Pruner 的目标：让 fake 被判为 real）
        adv_loss = self.disc_manager.compute_adv_loss(h_fake_dict)
        losses['adv_loss'] = adv_loss

        # 3. Discriminator loss
        disc_loss = self.disc_manager.compute_disc_loss(h_real_dict, h_fake_dict)
        losses['disc_loss'] = disc_loss

        # 4. Sparsity loss
        target_ratio = target_token_num / n_vision
        sparsity_loss = torch.tensor(0.0, device=task_loss.device)
        total_kept_ratio = 0

        for layer_idx, info in output.pruning_infos.items():
            hard_mask = info['hard_mask']  # (batch, n_vision)
            kept_ratio = hard_mask.mean()
            total_kept_ratio += kept_ratio.item()
            sparsity_loss = sparsity_loss + torch.abs(kept_ratio - target_ratio)

        sparsity_loss = sparsity_loss / len(output.pruning_infos)
        losses['sparsity_loss'] = sparsity_loss
        losses['kept_ratio'] = torch.tensor(total_kept_ratio / len(output.pruning_infos))

        # 5. Pruner 总损失
        pruner_total = task_loss * task_weight + adv_loss * adv_weight + sparsity_loss * sparsity_weight
        losses['pruner_total'] = pruner_total

        return losses

    def train(self, mode: bool = True):
        """设置训练模式"""
        super().train(mode)
        self.base_model.train(mode)
        return self

    def eval(self):
        """设置评估模式"""
        return self.train(False)

    def to(self, device):
        """移动到指定设备"""
        super().to(device)
        self.base_model.to(device)
        return self

    @property
    def device(self):
        """获取模型设备"""
        return next(self.parameters()).device

    def generate(self, *args, **kwargs):
        """生成文本（直接调用基础模型）"""
        return self.base_model.generate(*args, **kwargs)

    # ========== 推理剪枝模式 ==========

    def compute_pruning_masks(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_start: int = None,
        vision_end: int = None,
    ) -> Dict[int, torch.Tensor]:
        """预计算剪枝 masks

        在 generate 之前调用，为每个剪枝层计算 hard_mask。

        参数:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            attention_mask: 注意力掩码
            vision_start: vision tokens 起始位置
            vision_end: vision tokens 结束位置

        返回:
            masks: {layer_idx: hard_mask} 字典
        """
        self.eval()
        model = self.base_model.model
        llm = model.language_model

        # 获取 image features 并准备 inputs_embeds
        vision_feature_layer = self.config.vision_feature_layer
        vision_feature_select_strategy = self.config.vision_feature_select_strategy

        inputs_embeds = model.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            image_features = model.get_image_features(
                pixel_values=pixel_values,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_features = torch.cat(image_features, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            special_image_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # 自动检测 vision tokens 位置
        if vision_start is None:
            vision_start = 1  # 跳过 BOS
        if vision_end is None:
            vision_end = vision_start + 576  # LLaVA 1.5 默认 576 tokens

        batch_size, seq_len, _ = inputs_embeds.shape

        # 准备 position embeddings
        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        position_embeddings = llm.rotary_emb(inputs_embeds, position_ids)

        # 遍历到第一个剪枝层，获取 hidden states
        hidden_states = inputs_embeds
        masks = {}

        for layer_idx, decoder_layer in enumerate(llm.layers):
            if layer_idx in self.pruning_layers:
                # 在剪枝层，提取 vision hidden 并计算 mask
                hidden_normed = decoder_layer.original_layer.input_layernorm(hidden_states)
                vision_hidden = hidden_normed[:, vision_start:vision_end, :]

                pruner = self.pruner_manager.get_pruner(layer_idx)
                with torch.no_grad():
                    hard_mask, _ = pruner.forward_full(vision_hidden)
                masks[layer_idx] = hard_mask

            # 继续前向传播（使用原始层，不剪枝）
            if isinstance(decoder_layer, PrunableLlamaDecoderLayer):
                hidden_states = decoder_layer.original_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                )

        return masks

    def enable_inference_pruning(
        self,
        masks: Dict[int, torch.Tensor],
        vision_start: int,
        vision_end: int
    ):
        """启用推理剪枝模式

        参数:
            masks: 预计算的 {layer_idx: hard_mask} 字典
            vision_start: vision tokens 起始位置
            vision_end: vision tokens 结束位置
        """
        llm = self.base_model.model.language_model

        for layer_idx in self.pruning_layers:
            decoder_layer = llm.layers[layer_idx]
            if isinstance(decoder_layer, PrunableLlamaDecoderLayer):
                decoder_layer.enable_inference_pruning(vision_start, vision_end)
                if layer_idx in masks:
                    decoder_layer.set_cached_mask(masks[layer_idx])

    def disable_inference_pruning(self):
        """禁用推理剪枝模式"""
        llm = self.base_model.model.language_model

        for layer_idx in self.pruning_layers:
            decoder_layer = llm.layers[layer_idx]
            if isinstance(decoder_layer, PrunableLlamaDecoderLayer):
                decoder_layer.disable_inference_pruning()

    def generate_with_pruning(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_start: int = None,
        vision_end: int = None,
        **generate_kwargs
    ) -> torch.LongTensor:
        """带剪枝的生成

        参数:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            attention_mask: 注意力掩码
            vision_start: vision tokens 起始位置（可选，自动检测）
            vision_end: vision tokens 结束位置（可选，自动检测）
            **generate_kwargs: 传递给 generate 的其他参数

        返回:
            output_ids: 生成的 token IDs
        """
        # 自动检测 vision tokens 位置
        if vision_start is None:
            vision_start = 1
        if vision_end is None:
            vision_end = vision_start + 576

        # 1. 预计算 masks
        masks = self.compute_pruning_masks(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            vision_start=vision_start,
            vision_end=vision_end,
        )

        # 2. 启用推理剪枝
        self.enable_inference_pruning(masks, vision_start, vision_end)

        try:
            # 3. 生成
            output_ids = self.base_model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                **generate_kwargs
            )
        finally:
            # 4. 禁用推理剪枝
            self.disable_inference_pruning()

        return output_ids

    def get_kept_ratio_from_masks(self, masks: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """从 masks 计算保留率统计（使用加权平均，与训练时一致）

        参数:
            masks: {layer_idx: hard_mask} 字典

        返回:
            stats: 包含各层和平均保留率的字典
        """
        stats = {}

        if not masks:
            return stats

        # 获取 LLM 总层数
        total_layers = len(self.base_model.model.language_model.layers)
        pruning_layers = sorted(masks.keys())

        # 加权平均：每个剪枝层的 mask 影响该层及之后的所有层
        weighted_kept = 0.0

        for i, layer_idx in enumerate(pruning_layers):
            # 剪枝层之前的层数（100% 保留）
            if i == 0:
                n_layers_before = layer_idx
                weighted_kept += n_layers_before * 1.0

            # 该剪枝层影响的层数
            if i < len(pruning_layers) - 1:
                n_affected = pruning_layers[i + 1] - layer_idx
            else:
                n_affected = total_layers - layer_idx

            ratio = masks[layer_idx].float().mean().item()
            weighted_kept += n_affected * ratio
            stats[f'L{layer_idx}_kept'] = ratio

        # LLM 平均每层的保留比例
        stats['avg_kept_ratio'] = weighted_kept / total_layers

        return stats

