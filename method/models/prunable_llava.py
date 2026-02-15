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
from transformers.models.llama.modeling_llama import LlamaModel, LlamaDecoderLayer

from .layer_pruner_acp import LayerPruner, LayerPrunerManager
from .layer_discriminator import LayerDiscriminator, LayerDiscriminatorManager
from .prunable_llama_layer import PrunableLlamaDecoderLayer
from .adapter import AdapterManager, SeparatedAdapterManager


def build_vision_pruning_attention_mask(
    cumulative_vision_mask: torch.Tensor,
    vision_start: int,
    vision_end: int,
    seq_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """构建 4D attention mask，屏蔽被剪掉的 vision tokens。

    Args:
        cumulative_vision_mask: (batch, n_vision), 1=保留, 0=已被剪掉
        vision_start: vision tokens 在序列中的起始位置
        vision_end: vision tokens 在序列中的结束位置
        seq_len: 序列总长度
        dtype: tensor dtype
        device: tensor device

    Returns:
        attention_mask: (batch, 1, seq_len, seq_len)
            0.0 = 参与 attention, min_val = 不参与
    """
    batch_size = cumulative_vision_mask.shape[0]
    min_val = torch.finfo(dtype).min

    # 创建 causal mask: (1, 1, seq_len, seq_len)
    causal_mask = torch.triu(
        torch.full((seq_len, seq_len), min_val, device=device, dtype=dtype),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # 创建 vision pruning mask: (batch, 1, 1, n_vision)
    # 0 -> min_val (不参与), 1 -> 0 (参与)
    n_vision = vision_end - vision_start
    vision_mask = (1 - cumulative_vision_mask).unsqueeze(1).unsqueeze(2) * min_val

    # 合并到完整 mask: (batch, 1, seq_len, seq_len)
    attn_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len).clone()
    attn_mask[:, :, :, vision_start:vision_end] = attn_mask[:, :, :, vision_start:vision_end] + vision_mask

    return attn_mask


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
    # 物理删除后调整的位置（用于 compute_task_loss）
    adjusted_answer_starts: Optional[List[int]] = None
    adjusted_answer_ends: Optional[List[int]] = None


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
        adapter_bottleneck: int = None,  # adapter 瓶颈维度，None 则为 hidden_size // 4
        adapter_type: str = 'lightweight',    # adapter 类型: 'simple' 或 'lightweight'
        use_separated_adapters: bool = False,  # 是否使用分离式 Adapter
        vision_adapter_bottleneck: int = 256,  # vision adapter 瓶颈维度
        text_adapter_bottleneck: int = 256,    # text adapter 瓶颈维度
        generator_adapter_bottleneck: int = 512,  # generator adapter 瓶颈维度
        temperature: float = 1.0,
        dropout: float = 0.1,
        disc_use_spectral_norm: bool = False,
        use_gumbel_noise: bool = True,  # 是否使用 Gumbel noise
        pruning_threshold: float = 0.5,  # sigmoid 后的剪枝阈值
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
            dropout=dropout,
            use_gumbel_noise=use_gumbel_noise,
            pruning_threshold=pruning_threshold,
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
                n_vision=576,
                dropout=dropout,
            )
            self.adapter_manager = None
        else:
            self.adapter_manager = AdapterManager(
                layer_indices=pruning_layers,
                hidden_size=self.hidden_size,
                bottleneck_dim=adapter_bottleneck,
                adapter_type=adapter_type,
                n_vision=576,  # LLaVA 1.5 的 vision token 数量
            )
            self.separated_adapter_manager = None

        # 替换所有层为 PrunableLlamaDecoderLayer（剪枝层有 pruner，非剪枝层没有）
        self._replace_all_layers()

    def _replace_all_layers(self):
        """替换所有层为 PrunableLlamaDecoderLayer

        剪枝层：有 pruner, discriminator, adapter
        非剪枝层：没有 pruner，但可以应用 cumulative_mask（post-softmax masking）
        """
        llm = self.base_model.model.language_model
        num_layers = len(llm.layers)

        for layer_idx in range(num_layers):
            original_layer = llm.layers[layer_idx]

            # 跳过已经是 PrunableLlamaDecoderLayer 的层
            if isinstance(original_layer, PrunableLlamaDecoderLayer):
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

                llm.layers[layer_idx] = PrunableLlamaDecoderLayer(
                    original_layer=original_layer,
                    layer_idx=layer_idx,
                    pruner=pruner,
                    discriminator=discriminator,
                    adapter=adapter,
                    separated_adapters=separated_adapters
                )
            else:
                # 非剪枝层：没有 pruner，但可以应用 cumulative_mask
                llm.layers[layer_idx] = PrunableLlamaDecoderLayer(
                    original_layer=original_layer,
                    layer_idx=layer_idx,
                    pruner=None,
                    discriminator=None,
                    adapter=None,
                    separated_adapters=None
                )

    def _restore_original_layers(self):
        """还原为原始层（用于保存模型等场景）"""
        llm = self.base_model.model.language_model

        for layer_idx in range(len(llm.layers)):
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
        detach_h_fake_for_adv: bool = False,  # 是否 detach h_fake（阻止 adv_loss 梯度流向 pruner）
        **kwargs
    ) -> PrunableLlavaOutput:
        """前向传播（训练时物理删除 vision tokens，与推理完全对齐）

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
            # 兼容不同版本的 transformers：可能返回 list/tuple 或单个 tensor
            if isinstance(image_features, (list, tuple)):
                image_features = torch.cat(image_features, dim=0)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # 准备 LLaMA forward 的参数
        batch_size, orig_seq_len, _ = inputs_embeds.shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        use_cache = kwargs.get('use_cache', False)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=llm.config)

        # === 累积 mask：追踪哪些 vision tokens 被剪掉 ===
        n_vision_orig = vision_end - vision_start if (vision_start is not None and vision_end is not None) else 0
        if n_vision_orig > 0:
            cumulative_mask = torch.ones(
                batch_size, n_vision_orig,
                device=device,
                dtype=dtype
            )
        else:
            cumulative_mask = None

        # === 当前状态（不做物理删除，位置保持不变）===
        hidden_states = inputs_embeds

        # === 遍历所有层 ===
        pruning_infos = {}

        # 构建 causal mask（所有层共用，不包含 vision pruning）
        min_val = torch.finfo(dtype).min
        causal_mask = torch.triu(
            torch.full((orig_seq_len, orig_seq_len), min_val, device=device, dtype=dtype),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

        for layer_idx, decoder_layer in enumerate(llm.layers):
            # position_ids 保持不变（不做物理删除）
            position_ids = torch.arange(orig_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            position_embeddings = llm.rotary_emb(hidden_states, position_ids)

            if isinstance(decoder_layer, PrunableLlamaDecoderLayer) and return_pruning_info:
                # PrunableLlamaDecoderLayer（剪枝层或非剪枝层）
                # 使用 causal mask，vision pruning 通过 post-softmax masking 实现
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
                    # 用新的累积 mask 更新
                    if 'cumulative_mask' in pruning_info:
                        cumulative_mask = pruning_info['cumulative_mask'].clone()

            else:
                # 非 PrunableLlamaDecoderLayer（不应该发生，因为所有层都被替换了）
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=torch.arange(orig_seq_len, device=device),
                    position_embeddings=position_embeddings,
                )


        # Final LayerNorm
        hidden_states = llm.norm(hidden_states)

        # LM Head
        logits = self.base_model.lm_head(hidden_states)

        # 计算 loss（序列长度不变，不需要调整 labels）
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
            adjusted_answer_starts=answer_starts,  # 位置不变
            adjusted_answer_ends=answer_ends,
        )

    def set_temperature(self, temperature: float):
        """设置所有 pruner 的温度"""
        self.pruner_manager.set_temperature(temperature)

    def set_use_gumbel_noise(self, use_gumbel_noise: bool):
        """设置所有 pruner 是否使用 Gumbel noise"""
        self.pruner_manager.set_use_gumbel_noise(use_gumbel_noise)

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
        """冻结基础模型参数（但保持 pruner, discriminator, adapter 可训练）"""
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 重新启用 pruner, discriminator, adapter 的梯度
        # （因为它们已经被添加到 llm.layers 中，会被上面的循环冻结）
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
            cumulative_mask = info['cumulative_mask']  # (batch, n_vision)
            kept_ratio = cumulative_mask.mean()
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

    def get_kept_ratio_from_masks(
        self,
        masks: Dict[int, torch.Tensor],
        original_n_vision: int = 576
    ) -> Dict[str, float]:
        """从 masks 计算保留率统计（使用累积公式，与训练时一致）

        剪枝是累积的：L4 剪枝后，L14 在剩余 tokens 上再剪枝。
        所有 kept ratio 都以原始 vision token 数量为分母。

        参数:
            masks: {layer_idx: (hard_mask, n_kept_absolute)} 字典
                   hard_mask 用于计算相对比例，n_kept_absolute 是绝对保留数量
            original_n_vision: 原始 vision token 数量（默认 576）

        返回:
            stats: 包含各层和平均保留率的字典
        """
        stats = {}

        if not masks:
            return stats

        # 获取 LLM 总层数
        total_layers = len(self.base_model.model.language_model.layers)
        pruning_layers = sorted(masks.keys())

        # 累积计算
        weighted_kept = 0.0
        cumulative_kept = original_n_vision  # 当前累积保留的 token 数量

        for i, layer_idx in enumerate(pruning_layers):
            # 剪枝层之前的层数
            if i == 0:
                n_layers_before = layer_idx
                weighted_kept += n_layers_before * 1.0  # 第一个剪枝层之前是 100%

            # 获取该层的 mask 信息
            mask_info = masks[layer_idx]
            if isinstance(mask_info, tuple):
                if len(mask_info) == 3:
                    hard_mask, n_kept_absolute, _ = mask_info  # 新格式：包含 padded_mask
                else:
                    hard_mask, n_kept_absolute = mask_info  # 旧格式
            else:
                # 兼容旧格式：只有 hard_mask
                hard_mask = mask_info
                n_kept_absolute = (hard_mask > 0.5).sum().int().item()  # 用 >0.5 避免 bfloat16 sum 误差

            # 更新累积保留数量
            cumulative_kept = n_kept_absolute

            # 计算相对于原始 token 数量的保留率
            absolute_ratio = n_kept_absolute / original_n_vision
            stats[f'L{layer_idx}_kept'] = absolute_ratio
            stats[f'L{layer_idx}_n_kept'] = n_kept_absolute

            # 该剪枝层影响的层数
            if i < len(pruning_layers) - 1:
                n_affected = pruning_layers[i + 1] - layer_idx
            else:
                n_affected = total_layers - layer_idx

            weighted_kept += n_affected * absolute_ratio

        # LLM 平均每层的保留比例
        stats['avg_kept_ratio'] = weighted_kept / total_layers
        stats['original_n_vision'] = original_n_vision
        stats['final_n_kept'] = cumulative_kept

        return stats

    # ========== 硬剪枝推理模式 ==========

    def generate_with_hard_pruning(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        vision_start: int = None,
        vision_end: int = None,
        question_starts: List[int] = None,
        question_ends: List[int] = None,
        **generate_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, float]]:
        """带硬剪枝的生成 - 物理删除被剪掉的 tokens 以减少 FLOPS

        在 prefill 阶段：
        1. 遍历所有层，在剪枝层计算 mask 并物理删除 tokens
        2. 更新 position_ids（重新编号）
        3. 构建 KV cache
        4. 调用 generate 进行 decode

        参数:
            input_ids: 输入 token IDs
            pixel_values: 图像像素值
            attention_mask: 注意力掩码
            vision_start: vision tokens 起始位置
            vision_end: vision tokens 结束位置
            question_starts: 每个样本的 question 开始位置
            question_ends: 每个样本的 question 结束位置
            **generate_kwargs: 传递给 generate 的其他参数

        返回:
            output_ids: 生成的 token IDs
            kept_stats: 保留率统计
        """
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

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
            # 兼容不同版本的 transformers：可能返回 list/tuple 或单个 tensor
            if isinstance(image_features, (list, tuple)):
                image_features = torch.cat(image_features, dim=0)
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # 自动检测 vision tokens 位置
        if vision_start is None:
            vision_start = 1
        if vision_end is None:
            vision_end = vision_start + 576

        batch_size, seq_len, hidden_size = inputs_embeds.shape
        n_vision = vision_end - vision_start

        # 如果没有提供 question 位置，使用默认值
        if question_starts is None:
            question_starts = [vision_end] * batch_size
        if question_ends is None:
            question_ends = [seq_len] * batch_size

        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # === Prefill with hard pruning ===
        # 初始化
        hidden_states = inputs_embeds
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

        # 创建 KV cache
        past_key_values = DynamicCache()

        # 记录每层的 mask（用于统计）
        masks = {}

        # 当前 vision tokens 的位置和数量（会随着剪枝而变化）
        current_vision_start = vision_start
        current_vision_end = vision_end
        # 记录每个样本中哪些原始位置被保留了
        # kept_indices[i] = 原始序列中被保留的 token 索引列表
        kept_indices = [list(range(seq_len)) for _ in range(batch_size)]

        # 累积 vision mask（相对于原始 n_vision 个 token，用于 adapter）
        # 与训练时保持一致：adapter 接收的 mask 始终是原始 n_vision 维
        cumulative_vision_mask = torch.ones(batch_size, n_vision, device=device, dtype=dtype)

        for layer_idx, decoder_layer in enumerate(llm.layers):
            # 获取原始层
            if isinstance(decoder_layer, PrunableLlamaDecoderLayer):
                original_layer = decoder_layer.original_layer
            else:
                original_layer = decoder_layer

            # 当前序列长度
            current_seq_len = hidden_states.shape[1]

            # 计算 position embeddings（使用当前的 position_ids）
            position_embeddings = llm.rotary_emb(hidden_states, position_ids)

            # === 非剪枝层：直接使用原始层的 forward（与训练完全一致）===
            if layer_idx not in self.pruning_layers:
                # 使用原始层的 forward，确保数值计算与训练完全一致
                layer_outputs = original_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=torch.arange(current_seq_len, device=device),
                    position_embeddings=position_embeddings,
                )
                # 处理输出格式
                if isinstance(layer_outputs, tuple):
                    hidden_states = layer_outputs[0]
                else:
                    hidden_states = layer_outputs

                continue

            # === 剪枝层：手动计算 attention 以实现剪枝 ===
            attn = original_layer.self_attn

            # 获取配置
            num_heads = attn.config.num_attention_heads
            num_kv_heads = attn.config.num_key_value_heads
            head_dim = attn.head_dim
            num_kv_groups = num_heads // num_kv_heads

            cos, sin = position_embeddings

            # LayerNorm + Q/K/V 投影
            hidden_normed = original_layer.input_layernorm(hidden_states)

            # 保存 hidden_normed 用于后续 pruning（在 attention 计算之前）
            hidden_normed_for_pruning = hidden_normed

            query_states = attn.q_proj(hidden_normed)
            key_states = attn.k_proj(hidden_normed)
            value_states = attn.v_proj(hidden_normed)

            # 保存 query_states_flat 用于 adapter（在 reshape 之前）
            # 与训练时保持一致：adapter 接收 (batch, seq, hidden_size) 的 query
            query_states_flat = query_states

            # Reshape
            query_states = query_states.view(batch_size, current_seq_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(batch_size, current_seq_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, current_seq_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            # 存入 KV cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": torch.arange(current_seq_len, device=device)}
            key_states_cached, value_states_cached = past_key_values.update(
                key_states, value_states, layer_idx, cache_kwargs
            )

            # Repeat KV for GQA
            key_states_expanded = repeat_kv(key_states_cached, num_kv_groups)
            value_states_expanded = repeat_kv(value_states_cached, num_kv_groups)

            # 计算 attention
            attn_weights = torch.matmul(query_states, key_states_expanded.transpose(-2, -1)) * attn.scaling

            # 应用 causal mask
            causal_mask = torch.triu(
                torch.full((current_seq_len, current_seq_len), float('-inf'), device=device, dtype=dtype),
                diagonal=1
            )
            attn_weights = attn_weights + causal_mask

            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(dtype)

            # === 剪枝层处理 ===
            # Step 1: 计算 question->vision attention 和 hard_mask
            q2v_attn_list = []
            for i in range(batch_size):
                orig_q_start, orig_q_end = question_starts[i], question_ends[i]
                current_q_start = None
                current_q_end = None
                for new_idx, orig_idx in enumerate(kept_indices[i]):
                    if orig_idx == orig_q_start and current_q_start is None:
                        current_q_start = new_idx
                    if orig_idx == orig_q_end - 1:
                        current_q_end = new_idx + 1
                if current_q_start is None:
                    current_q_start = current_vision_end
                if current_q_end is None:
                    current_q_end = current_seq_len
                q2v_i = attn_weights[i, :, current_q_start:current_q_end, current_vision_start:current_vision_end]
                q2v_avg_i = q2v_i.mean(dim=(0, 1))
                q2v_attn_list.append(q2v_avg_i)
            q2v_attn_avg = torch.stack(q2v_attn_list, dim=0)

            vision_hidden = hidden_normed_for_pruning[:, current_vision_start:current_vision_end, :]

            # 计算已被剪掉的 tokens 数量（用于修正 baseline_mean）
            current_n_vision = current_vision_end - current_vision_start
            n_pruned_tokens = n_vision - current_n_vision

            pruner = self.pruner_manager.get_pruner(layer_idx)
            with torch.no_grad():
                hard_mask, _ = pruner.forward_full(
                    vision_hidden, q2v_attn_avg,
                    n_pruned_tokens=n_pruned_tokens
                )

            # Step 2: 用 mask 修改 attention weights（与训练一致）
            mask_expanded = hard_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_vision)

            # 创建完整的 mask（非 vision 部分为 1，vision 部分为 hard_mask）
            ones_before = torch.ones(batch_size, num_heads, current_seq_len, current_vision_start,
                                     device=device, dtype=dtype)
            ones_after = torch.ones(batch_size, num_heads, current_seq_len, current_seq_len - current_vision_end,
                                    device=device, dtype=dtype)
            mask_vision = mask_expanded.expand(-1, num_heads, current_seq_len, -1)
            full_mask = torch.cat([ones_before, mask_vision, ones_after], dim=-1)

            # 应用 mask 并重新归一化
            attn_weights_masked = attn_weights * full_mask
            attn_sum = attn_weights_masked.sum(dim=-1, keepdim=True)
            attn_weights_masked = (attn_weights_masked / (attn_sum + 1e-8)).to(dtype)

            # Step 3: 计算剪枝后的 attention output
            attn_output = torch.matmul(attn_weights_masked, value_states_expanded)

            attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, current_seq_len, hidden_size)

            # Step 4: 将当前层的 hard_mask scatter 到原始 n_vision 维度（与训练时一致）
            # 训练时使用 cumulative_vision_mask 的保留位置来 scatter，推理时也要一致
            # hard_mask: (batch, current_n_vision) -> scattered_mask: (batch, n_vision)
            scattered_mask = torch.zeros(batch_size, n_vision, device=device, dtype=dtype)
            for i in range(batch_size):
                kept_positions = cumulative_vision_mask[i].nonzero(as_tuple=True)[0]
                for j, pos in enumerate(kept_positions):
                    if j < hard_mask.shape[1]:
                        scattered_mask[i, pos] = hard_mask[i, j]

            # Step 5: 应用 Adapter（使用 scatter 后的 mask，与训练一致）
            if self.use_separated_adapters:
                # 分离式 Adapter：对 vision/text 分别处理（text 包含 question 和 generator）
                vision_adapter, text_adapter = self.separated_adapter_manager.get_adapters(layer_idx)
                adapted_output = attn_output.clone()

                # Vision tokens: [current_vision_start, current_vision_end)
                vision_slice = attn_output[:, current_vision_start:current_vision_end, :]
                vision_query = query_states_flat[:, current_vision_start:current_vision_end, :]
                adapted_vision = vision_adapter(vision_slice, mask=scattered_mask, query=vision_query)
                adapted_output[:, current_vision_start:current_vision_end, :] = adapted_vision

                # Text tokens (除最后一个): [current_vision_end, current_seq_len-1)
                if current_vision_end < current_seq_len - 1:
                    text_slice = attn_output[:, current_vision_end:current_seq_len-1, :]
                    text_query = query_states_flat[:, current_vision_end:current_seq_len-1, :]
                    adapted_text = text_adapter(text_slice, mask=scattered_mask, query=text_query)
                    adapted_output[:, current_vision_end:current_seq_len-1, :] = adapted_text

                # Generator token (最后一个，用于生成第一个 answer token): 使用 text_adapter
                gen_slice = attn_output[:, current_seq_len-1:current_seq_len, :]
                gen_query = query_states_flat[:, current_seq_len-1:current_seq_len, :]
                adapted_gen = text_adapter(gen_slice, mask=scattered_mask, query=gen_query)
                adapted_output[:, current_seq_len-1:current_seq_len, :] = adapted_gen

                attn_output = adapted_output
            else:
                adapter = self.adapter_manager.get_adapter(layer_idx)
                if adapter is not None:
                    attn_output = adapter(
                        attn_output,
                        mask=scattered_mask,
                        query=query_states_flat
                    )

            # Step 6: 更新累积 vision mask（用于后续层的物理删除）
            # 直接使用 scattered_mask 作为新的累积 mask
            cumulative_vision_mask = scattered_mask.clone()

            # 记录 mask 用于统计（保存 scattered_mask 供 Generate 阶段使用）
            n_kept_absolute = (hard_mask[0] > 0.5).sum().int().item()  # 用 >0.5 避免 bfloat16 sum 误差
            masks[layer_idx] = (hard_mask, n_kept_absolute, scattered_mask)

            attn_output = attn.o_proj(attn_output)

            # 残差连接
            hidden_states = hidden_states + attn_output

            # MLP
            residual = hidden_states
            hidden_states = original_layer.post_attention_layernorm(hidden_states)
            hidden_states = original_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            # === 硬剪枝（物理删除 tokens）- 现在只有剪枝层会到达这里 ===
            # 物理删除被剪掉的 vision tokens
            # 对于每个样本，根据 hard_mask 选择要保留的 tokens
            new_hidden_states_list = []
            new_position_ids_list = []
            new_kept_indices_list = []

            for i in range(batch_size):
                # 当前样本的 mask
                sample_mask = hard_mask[i]  # (current_n_vision,)
                kept_vision_indices = sample_mask.nonzero(as_tuple=True)[0]  # 保留的 vision token 的相对索引

                # 构建新的序列：[前部分] + [保留的 vision tokens] + [后部分]
                # 前部分：position 0 到 current_vision_start
                # 后部分：current_vision_end 到末尾

                before_vision = hidden_states[i, :current_vision_start, :]  # (vision_start, hidden)
                kept_vision = hidden_states[i, current_vision_start:current_vision_end, :][kept_vision_indices]  # (n_kept, hidden)
                after_vision = hidden_states[i, current_vision_end:, :]  # (rest, hidden)

                new_hidden = torch.cat([before_vision, kept_vision, after_vision], dim=0)
                new_hidden_states_list.append(new_hidden)

                # 更新 kept_indices（记录哪些原始位置被保留）
                old_kept = kept_indices[i]
                new_kept = (
                    old_kept[:current_vision_start] +
                    [old_kept[current_vision_start + j] for j in kept_vision_indices.tolist()] +
                    old_kept[current_vision_end:]
                )
                new_kept_indices_list.append(new_kept)

                # 更新 position_ids（保持原位置，不重新编号）
                new_pos_ids = torch.tensor(new_kept, device=device, dtype=torch.long)
                new_position_ids_list.append(new_pos_ids)

            # Pad to same length (batch 内可能长度不同)
            max_new_len = max(h.shape[0] for h in new_hidden_states_list)

            padded_hidden = torch.zeros(batch_size, max_new_len, hidden_size, device=device, dtype=dtype)
            padded_position_ids = torch.zeros(batch_size, max_new_len, device=device, dtype=torch.long)

            for i in range(batch_size):
                length = new_hidden_states_list[i].shape[0]
                padded_hidden[i, :length] = new_hidden_states_list[i]
                padded_position_ids[i, :length] = new_position_ids_list[i]

            hidden_states = padded_hidden
            position_ids = padded_position_ids
            kept_indices = new_kept_indices_list

            # 更新 vision 位置（batch_size=1，直接用第一个样本）
            n_kept = (hard_mask[0] > 0.5).sum().int().item()  # 用 >0.5 避免 bfloat16 sum 误差
            current_vision_end = current_vision_start + n_kept

            # 更新 KV cache（删除被剪掉的 tokens）
            # 需要重新构建 cache
            new_cache = DynamicCache()
            for l_idx in range(layer_idx + 1):
                # 使用 layers[l_idx].keys 和 layers[l_idx].values 访问
                old_k = past_key_values.layers[l_idx].keys  # (batch, heads, old_seq, head_dim)
                old_v = past_key_values.layers[l_idx].values

                new_k_list = []
                new_v_list = []

                for i in range(batch_size):
                    sample_mask = hard_mask[i]
                    kept_vision_indices = sample_mask.nonzero(as_tuple=True)[0]

                    # 同样的逻辑：保留 before + kept_vision + after
                    before_k = old_k[i, :, :current_vision_start, :]
                    kept_k = old_k[i, :, current_vision_start:current_vision_start + len(sample_mask), :][:, kept_vision_indices, :]
                    after_k = old_k[i, :, current_vision_start + len(sample_mask):, :]
                    new_k = torch.cat([before_k, kept_k, after_k], dim=1)
                    new_k_list.append(new_k)

                    before_v = old_v[i, :, :current_vision_start, :]
                    kept_v = old_v[i, :, current_vision_start:current_vision_start + len(sample_mask), :][:, kept_vision_indices, :]
                    after_v = old_v[i, :, current_vision_start + len(sample_mask):, :]
                    new_v = torch.cat([before_v, kept_v, after_v], dim=1)
                    new_v_list.append(new_v)

                # Pad KV cache
                max_kv_len = max(k.shape[1] for k in new_k_list)
                padded_k = torch.zeros(batch_size, old_k.shape[1], max_kv_len, head_dim, device=device, dtype=dtype)
                padded_v = torch.zeros(batch_size, old_v.shape[1], max_kv_len, head_dim, device=device, dtype=dtype)

                for i in range(batch_size):
                    length = new_k_list[i].shape[1]
                    padded_k[i, :, :length, :] = new_k_list[i]
                    padded_v[i, :, :length, :] = new_v_list[i]

                # 使用 update 方法添加到新 cache
                new_cache.update(padded_k, padded_v, l_idx, {})

            past_key_values = new_cache

        # Final LayerNorm
        hidden_states = llm.norm(hidden_states)

        # 计算保留率统计（使用原始 vision token 数量作为分母）
        kept_stats = self.get_kept_ratio_from_masks(masks, original_n_vision=n_vision)

        # === Decode 阶段 ===
        # 获取最后一个 token 的 logits
        logits = self.base_model.lm_head(hidden_states[:, -1:, :])

        # 使用 generate 进行后续生成
        # 注意：需要传入已经构建好的 past_key_values
        max_new_tokens = generate_kwargs.pop('max_new_tokens', 32)
        debug_generate = generate_kwargs.pop('debug_generate', False)

        # 获取 attention 配置（用于 decode 阶段）
        first_layer = llm.layers[0]
        if isinstance(first_layer, PrunableLlamaDecoderLayer):
            first_layer = first_layer.original_layer
        first_attn = first_layer.self_attn
        num_heads = first_attn.config.num_attention_heads
        num_kv_heads = first_attn.config.num_key_value_heads
        head_dim = first_attn.head_dim
        num_kv_groups = num_heads // num_kv_heads

        # 获取 EOS token ID（LLaVA 模型可能在 text_config 中）
        eos_token_id = self.base_model.config.eos_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.base_model.config, 'text_config', {})
            if hasattr(eos_token_id, 'eos_token_id'):
                eos_token_id = eos_token_id.eos_token_id
            else:
                eos_token_id = 2  # 默认 LLaMA EOS token ID
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0]

        # 获取下一个 token
        next_token = logits.argmax(dim=-1)  # (batch, 1)

        # Debug: 打印第一个生成的 token
        if debug_generate:
            token_id = next_token[0, 0].item()
            print(f"[DEBUG generate] step 1: token_id={token_id}, eos_token_id={eos_token_id}, is_eos={token_id == eos_token_id}")

        generated_tokens = [next_token]

        # 检查第一个 token 是否是 EOS
        if eos_token_id is not None:
            if (next_token == eos_token_id).all():
                generated = torch.cat(generated_tokens, dim=1)
                output_ids = torch.cat([input_ids, generated], dim=1)
                return output_ids, kept_stats

        for _ in range(max_new_tokens - 1):
            # 准备输入
            current_pos = past_key_values.get_seq_length()
            cache_position = torch.tensor([current_pos], device=device)
            new_position_ids = torch.tensor([[current_pos]], device=device).expand(batch_size, -1)

            # 获取 embedding
            next_embeds = model.get_input_embeddings()(next_token)

            # Position embeddings
            position_embeddings = llm.rotary_emb(next_embeds, new_position_ids)

            # Forward through all layers
            hidden_states = next_embeds

            for layer_idx, decoder_layer in enumerate(llm.layers):
                if isinstance(decoder_layer, PrunableLlamaDecoderLayer):
                    original_layer = decoder_layer.original_layer
                else:
                    original_layer = decoder_layer

                # 获取 attention 模块
                attn = original_layer.self_attn

                # LayerNorm
                hidden_normed = original_layer.input_layernorm(hidden_states)

                # Q/K/V 投影
                query_states_gen = attn.q_proj(hidden_normed)
                key_states_gen = attn.k_proj(hidden_normed)
                value_states_gen = attn.v_proj(hidden_normed)

                # 保存 query_states_flat 用于 adapter
                query_states_flat_gen = query_states_gen

                # Reshape
                query_states_gen = query_states_gen.view(batch_size, 1, num_heads, head_dim).transpose(1, 2)
                key_states_gen = key_states_gen.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)
                value_states_gen = value_states_gen.view(batch_size, 1, num_kv_heads, head_dim).transpose(1, 2)

                # Apply RoPE
                cos, sin = position_embeddings
                query_states_gen, key_states_gen = apply_rotary_pos_emb(query_states_gen, key_states_gen, cos, sin)

                # 更新 KV cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states_gen, value_states_gen = past_key_values.update(
                    key_states_gen, value_states_gen, layer_idx, cache_kwargs
                )

                # Repeat KV for GQA
                key_states_gen = repeat_kv(key_states_gen, num_kv_groups)
                value_states_gen = repeat_kv(value_states_gen, num_kv_groups)

                # 计算 attention
                attn_weights_gen = torch.matmul(query_states_gen, key_states_gen.transpose(-2, -1)) * attn.scaling
                attn_weights_gen = F.softmax(attn_weights_gen, dim=-1, dtype=torch.float32).to(dtype)

                # 计算 attention output
                attn_output_gen = torch.matmul(attn_weights_gen, value_states_gen)
                attn_output_gen = attn_output_gen.transpose(1, 2).contiguous().reshape(batch_size, 1, hidden_size)

                # 在剪枝层应用 Adapter（使用 Prefill 阶段保存的 padded_mask）
                if layer_idx in self.pruning_layers:
                    # 从 masks 中获取 scattered_mask（与训练时一致的 scatter 格式）
                    _, _, scattered_mask = masks[layer_idx]
                    if self.use_separated_adapters:
                        # 分离式 Adapter：Generate 阶段使用 text_adapter
                        _, text_adapter = self.separated_adapter_manager.get_adapters(layer_idx)
                        attn_output_gen = text_adapter(
                            attn_output_gen,
                            mask=scattered_mask,
                            query=query_states_flat_gen
                        )
                    else:
                        adapter = self.adapter_manager.get_adapter(layer_idx)
                        if adapter is not None:
                            attn_output_gen = adapter(
                                attn_output_gen,
                                mask=scattered_mask,
                                query=query_states_flat_gen  # Decode 阶段的 query
                            )

                attn_output_gen = attn.o_proj(attn_output_gen)

                # 残差连接
                hidden_states = hidden_states + attn_output_gen

                # MLP
                residual = hidden_states
                hidden_states = original_layer.post_attention_layernorm(hidden_states)
                hidden_states = original_layer.mlp(hidden_states)
                hidden_states = residual + hidden_states

            hidden_states = llm.norm(hidden_states)
            logits = self.base_model.lm_head(hidden_states)
            next_token = logits.argmax(dim=-1)

            generated_tokens.append(next_token)

            # Debug: 打印生成的 token
            if debug_generate:
                token_id = next_token[0, 0].item()
                print(f"[DEBUG generate] step {len(generated_tokens)}: token_id={token_id}")

            # 检查是否生成了 EOS（使用前面已获取的 eos_token_id）
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        # 拼接生成的 tokens
        generated = torch.cat(generated_tokens, dim=1)

        # 拼接原始输入和生成的 tokens
        # 注意：input_ids 是原始长度，但我们实际处理的是剪枝后的序列
        # 返回时需要返回原始 input_ids + generated
        output_ids = torch.cat([input_ids, generated], dim=1)

        return output_ids, kept_stats

