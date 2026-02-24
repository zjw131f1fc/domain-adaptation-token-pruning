"""Attention Consistency Pruning - Prunable Qwen2-VL Model

可剪枝的 Qwen2-VL 模型，类似于 PrunableLlavaForConditionalGeneration。

核心改动：
1. 替换特定层为 PrunableQwen2VLDecoderLayer
2. 重写 forward 方法以传递剪枝参数和收集剪枝信息
3. 提供训练和推理两种模式

与 LLaVA 的主要差异：
1. 模型结构：model.language_model.layers（而非 base_model.model.language_model.layers）
2. Vision Token 位置：通过 image_token_id 检测
3. Position Embedding：3D M-RoPE
4. 支持 video 输入（本实现暂不支持）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass

from transformers import Qwen2VLForConditionalGeneration, Qwen2VLConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLDecoderLayer,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from .layer_pruner_acp import LayerPruner, LayerPrunerManager
from .layer_discriminator import LayerDiscriminator, LayerDiscriminatorManager
from .prunable_qwen2vl_layer import PrunableQwen2VLDecoderLayer
from .adapter import AdapterManager, SeparatedAdapterManager


@dataclass
class PrunableQwen2VLOutput:
    """可剪枝 Qwen2-VL 的输出"""
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    image_hidden_states: Optional[torch.Tensor] = None
    # 剪枝信息
    pruning_infos: Optional[Dict[int, Dict]] = None
    # 物理删除后调整的位置
    adjusted_answer_starts: Optional[List[int]] = None
    adjusted_answer_ends: Optional[List[int]] = None
    # Qwen2-VL 特有
    rope_deltas: Optional[torch.LongTensor] = None


class PrunableQwen2VLForConditionalGeneration(nn.Module):
    """可剪枝的 Qwen2-VL 模型

    通过替换特定层的 DecoderLayer 为 PrunableQwen2VLDecoderLayer 实现剪枝。

    参数:
        base_model: 基础的 Qwen2VLForConditionalGeneration 模型
        pruning_layers: 要剪枝的层索引列表
        pruner_d_internal: Pruner 内部维度
        disc_d_hidden: Discriminator 隐藏层维度
        temperature: 初始 Gumbel-Softmax 温度
        dropout: Dropout 比例
        n_vision: Vision token 数量（默认 576）
    """

    def __init__(
        self,
        base_model: Qwen2VLForConditionalGeneration,
        pruning_layers: List[int] = [4, 14, 24],
        pruner_d_internal: int = 128,
        pruner_n_heads: int = 4,
        pruner_n_queries: int = 4,
        pruner_query_dropout: float = 0.0,
        disc_d_hidden: int = 256,
        adapter_bottleneck: int = None,
        adapter_type: str = 'lightweight',
        use_separated_adapters: bool = False,
        vision_adapter_bottleneck: int = 256,
        text_adapter_bottleneck: int = 256,
        generator_adapter_bottleneck: int = 512,
        mask_encoder_type: str = 'attention',
        temperature: float = 1.0,
        dropout: float = 0.1,
        adapter_dropout: float = 0.15,
        disc_use_spectral_norm: bool = False,
        use_gumbel_noise: bool = True,
        pruning_threshold: float = 0.5,
        use_question_condition: bool = False,
        n_vision: int = 576,  # Vision token 数量
    ):
        super().__init__()

        # 保存基础模型
        self.base_model = base_model
        self.config = base_model.config
        self.pruning_layers = pruning_layers
        self.use_question_condition = use_question_condition
        self.n_vision = n_vision

        # 获取 LLM 配置
        llm_config = self.config.text_config
        self.num_heads = llm_config.num_attention_heads
        self.head_dim = llm_config.hidden_size // self.num_heads
        self.hidden_size = llm_config.hidden_size

        # 创建 Pruners
        self.pruner_manager = LayerPrunerManager(
            layer_indices=pruning_layers,
            d_model=self.hidden_size,
            d_internal=pruner_d_internal,
            n_heads=pruner_n_heads,
            n_queries=pruner_n_queries,
            temperature=temperature,
            dropout=dropout,
            query_dropout=pruner_query_dropout,
            use_gumbel_noise=use_gumbel_noise,
            pruning_threshold=pruning_threshold,
            use_question_condition=use_question_condition,
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

        # 创建 Adapters
        self.use_separated_adapters = use_separated_adapters
        if use_separated_adapters:
            self.separated_adapter_manager = SeparatedAdapterManager(
                layer_indices=pruning_layers,
                hidden_size=self.hidden_size,
                vision_bottleneck_dim=vision_adapter_bottleneck,
                text_bottleneck_dim=text_adapter_bottleneck,
                answer_bottleneck_dim=generator_adapter_bottleneck,
                n_vision=n_vision,
                dropout=adapter_dropout,
                mask_encoder_type=mask_encoder_type,
            )
            self.adapter_manager = None
        else:
            self.adapter_manager = AdapterManager(
                layer_indices=pruning_layers,
                hidden_size=self.hidden_size,
                bottleneck_dim=adapter_bottleneck,
                adapter_type=adapter_type,
                n_vision=n_vision,
                mask_encoder_type=mask_encoder_type,
                dropout=adapter_dropout,
            )
            self.separated_adapter_manager = None

        # 替换所有层为 PrunableQwen2VLDecoderLayer
        self._replace_all_layers()

    def _replace_all_layers(self):
        """替换所有层为 PrunableQwen2VLDecoderLayer"""
        llm = self.base_model.model.language_model
        num_layers = len(llm.layers)

        for layer_idx in range(num_layers):
            original_layer = llm.layers[layer_idx]

            # 跳过已经是 PrunableQwen2VLDecoderLayer 的层
            if isinstance(original_layer, PrunableQwen2VLDecoderLayer):
                continue

            # 获取设备和 dtype
            layer_param = next(original_layer.parameters())
            layer_device = layer_param.device
            layer_dtype = layer_param.dtype

            if layer_idx in self.pruning_layers:
                # 剪枝层：有 pruner, discriminator, adapter
                pruner = self.pruner_manager.get_pruner(layer_idx)
                discriminator = self.disc_manager.get_discriminator(layer_idx)
                pruner.to(device=layer_device, dtype=layer_dtype)
                discriminator.to(device=layer_device, dtype=layer_dtype)

                if self.use_separated_adapters:
                    separated_adapters = self.separated_adapter_manager.get_adapters(layer_idx)
                    for adapter in separated_adapters:
                        adapter.to(device=layer_device, dtype=layer_dtype)
                    adapter = None
                else:
                    adapter = self.adapter_manager.get_adapter(layer_idx)
                    adapter.to(device=layer_device, dtype=layer_dtype)
                    separated_adapters = None

                llm.layers[layer_idx] = PrunableQwen2VLDecoderLayer(
                    original_layer=original_layer,
                    layer_idx=layer_idx,
                    pruner=pruner,
                    discriminator=discriminator,
                    adapter=adapter,
                    separated_adapters=separated_adapters
                )
            else:
                # 非剪枝层：没有 pruner
                llm.layers[layer_idx] = PrunableQwen2VLDecoderLayer(
                    original_layer=original_layer,
                    layer_idx=layer_idx,
                    pruner=None,
                    discriminator=None,
                    adapter=None,
                    separated_adapters=None
                )

    def _restore_original_layers(self):
        """还原为原始层"""
        llm = self.base_model.model.language_model

        for layer_idx in range(len(llm.layers)):
            prunable_layer = llm.layers[layer_idx]
            if isinstance(prunable_layer, PrunableQwen2VLDecoderLayer):
                llm.layers[layer_idx] = prunable_layer.original_layer

    def _find_vision_token_positions(
        self,
        input_ids: torch.LongTensor,
    ) -> Tuple[int, int]:
        """找到 vision tokens 的位置

        Qwen2-VL 使用 image_token_id 标记 vision tokens。

        Returns:
            vision_start: vision tokens 起始位置
            vision_end: vision tokens 结束位置
        """
        image_token_id = self.config.image_token_id

        # 找到第一个和最后一个 image token
        image_mask = input_ids[0] == image_token_id
        image_positions = image_mask.nonzero(as_tuple=True)[0]

        if len(image_positions) == 0:
            # 没有图像 token
            return None, None

        vision_start = image_positions[0].item()
        vision_end = image_positions[-1].item() + 1

        return vision_start, vision_end

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        # === 剪枝参数 ===
        vision_start: Optional[int] = None,
        vision_end: Optional[int] = None,
        question_starts: Optional[list] = None,
        question_ends: Optional[list] = None,
        answer_starts: Optional[list] = None,
        answer_ends: Optional[list] = None,
        return_pruning_info: bool = True,
        detach_h_fake_for_adv: bool = False,
        **kwargs
    ) -> PrunableQwen2VLOutput:
        """前向传播（训练时使用 post-softmax masking，与推理对齐）"""
        model = self.base_model.model
        llm = model.language_model

        # 获取 inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        # 处理图像
        image_embeds = None
        if pixel_values is not None:
            image_embeds = model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 自动检测 vision token 位置
        if vision_start is None or vision_end is None:
            vision_start, vision_end = self._find_vision_token_positions(input_ids)

        batch_size, orig_seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        use_cache = kwargs.get('use_cache', False)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=llm.config)

        # === 累积 mask ===
        n_vision_orig = vision_end - vision_start if (vision_start is not None and vision_end is not None) else 0
        if n_vision_orig > 0:
            cumulative_mask = torch.ones(
                batch_size, n_vision_orig,
                device=device,
                dtype=dtype
            )
        else:
            cumulative_mask = None

        hidden_states = inputs_embeds

        # === 计算 position_ids 和 position_embeddings ===
        if position_ids is None:
            if model.rope_deltas is None:
                position_ids, rope_deltas = model.get_rope_index(
                    input_ids, image_grid_thw, None, attention_mask
                )
                model.rope_deltas = rope_deltas
            else:
                position_ids = torch.arange(orig_seq_len, device=device)
                position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
                delta = model.rope_deltas.to(device)
                position_ids = position_ids + delta

        # 计算 position embeddings
        position_embeddings = llm.rotary_emb(hidden_states, position_ids)

        # 构建 causal mask
        min_val = torch.finfo(dtype).min
        causal_mask = torch.triu(
            torch.full((orig_seq_len, orig_seq_len), min_val, device=device, dtype=dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # === 遍历所有层 ===
        pruning_infos = {}

        for layer_idx, decoder_layer in enumerate(llm.layers):
            if isinstance(decoder_layer, PrunableQwen2VLDecoderLayer) and return_pruning_info:
                hidden_states, pruning_info = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=None,
                    position_embeddings=position_embeddings,
                    vision_start=vision_start,
                    vision_end=vision_end,
                    question_starts=question_starts,
                    question_ends=question_ends,
                    answer_starts=answer_starts,
                    answer_ends=answer_ends,
                    return_pruning_info=True,
                    cumulative_vision_mask=cumulative_mask,
                    detach_h_fake_for_adv=detach_h_fake_for_adv,
                )
                if pruning_info is not None:
                    pruning_infos[layer_idx] = pruning_info
                    if 'cumulative_mask' in pruning_info:
                        cumulative_mask = pruning_info['cumulative_mask'].clone()
            else:
                output = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=torch.arange(orig_seq_len, device=device),
                    position_embeddings=position_embeddings,
                )
                hidden_states = output[0] if isinstance(output, tuple) else output

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

        return PrunableQwen2VLOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=None,
            attentions=None,
            image_hidden_states=image_embeds,
            pruning_infos=pruning_infos if return_pruning_info else None,
            adjusted_answer_starts=answer_starts,
            adjusted_answer_ends=answer_ends,
            rope_deltas=model.rope_deltas,
        )

    def set_temperature(self, temperature: float):
        """设置所有 pruner 的温度"""
        self.pruner_manager.set_temperature(temperature)

    def set_use_gumbel_noise(self, use_gumbel_noise: bool):
        """设置所有 pruner 是否使用 Gumbel noise"""
        self.pruner_manager.set_use_gumbel_noise(use_gumbel_noise)

    def set_pruning_threshold(self, threshold: float):
        """设置所有 pruner 的剪枝阈值"""
        self.pruner_manager.set_pruning_threshold(threshold)

    def get_pruner_parameters(self):
        """获取所有 pruner 的参数"""
        return self.pruner_manager.parameters()

    def get_discriminator_parameters(self):
        """获取所有 discriminator 的参数"""
        return self.disc_manager.parameters()

    def get_adapter_parameters(self):
        """获取所有 adapter 的参数"""
        if self.use_separated_adapters:
            return self.separated_adapter_manager.parameters()
        else:
            return self.adapter_manager.parameters()

    def freeze_base_model(self):
        """冻结基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 重新启用 pruner, discriminator, adapter 的梯度
        for param in self.pruner_manager.parameters():
            param.requires_grad = True
        for param in self.disc_manager.parameters():
            param.requires_grad = True
        for param in self.get_adapter_parameters():
            param.requires_grad = True

    def unfreeze_base_model(self):
        """解冻基础模型参数"""
        for param in self.base_model.parameters():
            param.requires_grad = True

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
        """生成文本（直接调用基础模型，不带剪枝）"""
        return self.base_model.generate(*args, **kwargs)

    @torch.no_grad()
    def generate_with_pruning(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        vision_start: Optional[int] = None,
        vision_end: Optional[int] = None,
        question_starts: Optional[list] = None,
        question_ends: Optional[list] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        return_dict_in_generate: bool = False,
        **kwargs
    ):
        """带剪枝的生成（使用 post-softmax masking）

        在 prefill 阶段计算剪枝 mask，在 decode 阶段应用 mask 生成 tokens。

        Args:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            attention_mask: 注意力 mask
            image_grid_thw: 图像网格信息
            vision_start/end: vision token 位置
            question_starts/ends: question token 位置
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（0 表示 greedy）
            top_p: nucleus sampling 参数
            eos_token_id: 结束 token ID
            pad_token_id: padding token ID
            return_dict_in_generate: 是否返回详细信息

        Returns:
            generated_ids: 生成的 token IDs
            或 dict 包含 generated_ids 和 pruning_stats
        """
        model = self.base_model.model
        llm = model.language_model

        batch_size = input_ids.shape[0]
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # 获取 eos_token_id
        if eos_token_id is None:
            eos_token_id = self.config.text_config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.text_config.pad_token_id or eos_token_id

        # === Prefill 阶段 ===
        # 获取 inputs_embeds
        inputs_embeds = model.get_input_embeddings()(input_ids)

        # 处理图像
        if pixel_values is not None:
            image_embeds = model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 自动检测 vision token 位置
        if vision_start is None or vision_end is None:
            vision_start, vision_end = self._find_vision_token_positions(input_ids)

        orig_seq_len = inputs_embeds.shape[1]
        n_vision = vision_end - vision_start if (vision_start is not None and vision_end is not None) else 0

        # 初始化累积 mask
        if n_vision > 0:
            cumulative_mask = torch.ones(batch_size, n_vision, device=device, dtype=dtype)
        else:
            cumulative_mask = None

        # 计算 position_ids
        model.rope_deltas = None  # 重置
        position_ids, rope_deltas = model.get_rope_index(
            input_ids, image_grid_thw, None, attention_mask
        )
        model.rope_deltas = rope_deltas

        # 计算 position embeddings
        position_embeddings = llm.rotary_emb(inputs_embeds, position_ids)

        # 构建 causal mask
        min_val = torch.finfo(dtype).min
        causal_mask = torch.triu(
            torch.full((orig_seq_len, orig_seq_len), min_val, device=device, dtype=dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        # 初始化 KV cache
        past_key_values = DynamicCache(config=llm.config)

        hidden_states = inputs_embeds
        pruning_stats = {}

        # === Prefill: 遍历所有层 ===
        for layer_idx, decoder_layer in enumerate(llm.layers):
            if isinstance(decoder_layer, PrunableQwen2VLDecoderLayer):
                # 对于剪枝层，需要计算 mask
                if decoder_layer.is_pruning_layer and cumulative_mask is not None:
                    hidden_states, pruning_info = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=torch.arange(orig_seq_len, device=device),
                        position_embeddings=position_embeddings,
                        vision_start=vision_start,
                        vision_end=vision_end,
                        question_starts=question_starts,
                        question_ends=question_ends,
                        answer_starts=question_ends,  # prefill 时 answer 还没开始
                        answer_ends=question_ends,
                        return_pruning_info=True,
                        cumulative_vision_mask=cumulative_mask,
                    )
                    if pruning_info is not None:
                        cumulative_mask = pruning_info['cumulative_mask'].clone()
                        current_mask = pruning_info['current_mask']
                        kept_ratio = current_mask.float().mean().item()
                        pruning_stats[layer_idx] = {
                            'kept_ratio': kept_ratio,
                            'cumulative_kept_ratio': cumulative_mask.float().mean().item(),
                        }
                else:
                    # 非剪枝层，应用累积 mask
                    hidden_states, _ = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=torch.arange(orig_seq_len, device=device),
                        position_embeddings=position_embeddings,
                        vision_start=vision_start,
                        vision_end=vision_end,
                        return_pruning_info=True,
                        cumulative_vision_mask=cumulative_mask,
                    )
            else:
                output = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=torch.arange(orig_seq_len, device=device),
                    position_embeddings=position_embeddings,
                )
                hidden_states = output[0] if isinstance(output, tuple) else output

        # Final LayerNorm
        hidden_states = llm.norm(hidden_states)

        # 获取下一个 token
        logits = self.base_model.lm_head(hidden_states[:, -1:, :])

        # === Decode 阶段 ===
        generated_ids = input_ids.clone()
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_new_tokens):
            # 采样下一个 token
            if temperature == 0:
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum_probs - sorted_probs > top_p
                    sorted_probs[mask] = 0.0
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = sorted_indices.gather(-1, torch.multinomial(sorted_probs, 1))
                else:
                    next_token = torch.multinomial(probs, 1)

            # 更新 generated_ids
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)

            # 检查是否结束
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break

            # 准备下一步的输入
            next_embeds = model.get_input_embeddings()(next_token)
            cur_seq_len = generated_ids.shape[1]

            # 更新 position_ids
            next_position_ids = torch.full(
                (3, batch_size, 1),
                cur_seq_len - 1,
                device=device,
                dtype=torch.long
            )
            if model.rope_deltas is not None:
                next_position_ids = next_position_ids + model.rope_deltas.unsqueeze(0)

            # 计算 position embeddings
            next_position_embeddings = llm.rotary_emb(next_embeds, next_position_ids)

            # Decode: 遍历所有层
            hidden_states = next_embeds
            cache_position = torch.tensor([cur_seq_len - 1], device=device)

            for layer_idx, decoder_layer in enumerate(llm.layers):
                if isinstance(decoder_layer, PrunableQwen2VLDecoderLayer):
                    # Decode 阶段不需要计算新的 mask，直接使用原始层
                    output = decoder_layer.original_layer(
                        hidden_states,
                        attention_mask=None,
                        position_ids=next_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=next_position_embeddings,
                    )
                else:
                    output = decoder_layer(
                        hidden_states,
                        attention_mask=None,
                        position_ids=next_position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        cache_position=cache_position,
                        position_embeddings=next_position_embeddings,
                    )
                hidden_states = output[0] if isinstance(output, tuple) else output

            # Final LayerNorm
            hidden_states = llm.norm(hidden_states)

            # 获取下一个 token 的 logits
            logits = self.base_model.lm_head(hidden_states)

        if return_dict_in_generate:
            return {
                'sequences': generated_ids,
                'pruning_stats': pruning_stats,
                'final_cumulative_mask': cumulative_mask,
                'final_kept_ratio': cumulative_mask.float().mean().item() if cumulative_mask is not None else 1.0,
            }

        return generated_ids
