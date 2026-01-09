"""LLaVA Model with Integrated Pruning

通过继承并覆写LLaVA模型的decoder layers，将pruning逻辑直接集成到模型中，
替代外部hook机制，提供更好的FSDP兼容性。
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import warnings


class PruningLayerWrapper(nn.Module):
    """包装LLaMA Decoder Layer，注入pruning逻辑

    这个wrapper在原始layer的forward后应用vision token pruning。
    支持batch化处理和可选的attention residual。

    参数:
        original_layer: 原始的LlamaDecoderLayer
        pruner: VisionPrunerHead实例
        layer_idx: 层索引（用于调试）
    """

    def __init__(
        self,
        original_layer: LlamaDecoderLayer,
        pruner: nn.Module,
        layer_idx: int
    ):
        super().__init__()
        self.original_layer = original_layer
        self.pruner = pruner
        self.layer_idx = layer_idx

        # 这些将在forward时由parent model设置
        self.vision_positions = None  # (batch_size, 2) or (2,)
        self.question_embeddings = None  # (batch_size, q_len, d)
        self.use_attn_residual = False
        self.collect_masks = False
        self.mask_collector = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """Forward pass with pruning

        流程:
        1. 调用原始layer的forward
        2. 如果配置了pruning context，应用pruning
        3. 返回修改后的hidden states
        """
        # 1. 原始layer forward
        outputs = self.original_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )

        layer_output = outputs[0]  # (batch_size, seq_len, hidden_size)

        # 2. 应用pruning（如果配置了context）
        if self.vision_positions is not None and self.question_embeddings is not None:
            layer_output = self._apply_pruning(layer_output, attention_mask)

        # 3. 返回修改后的outputs
        return (layer_output,) + outputs[1:]

    def _apply_pruning(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """应用pruning到vision tokens

        参数:
            hidden_states: (batch_size, seq_len, hidden_size)
            attention_mask: (batch_size, seq_len)

        返回:
            修改后的hidden_states，vision部分被pruned
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device

        # 处理vision_positions的格式
        if self.vision_positions.dim() == 1:
            # (2,) -> (batch_size, 2)
            v_start = self.vision_positions[0].item()
            v_end = self.vision_positions[1].item()
            vision_pos_batch = self.vision_positions.unsqueeze(0).expand(batch_size, -1)
        else:
            # (batch_size, 2)
            v_start = self.vision_positions[0, 0].item()
            v_end = self.vision_positions[0, 1].item()
            vision_pos_batch = self.vision_positions

            # 验证batch内vision位置一致性
            if not torch.all(vision_pos_batch == vision_pos_batch[0]):
                warnings.warn(
                    f"Layer {self.layer_idx}: vision positions not uniform across batch, "
                    f"using first sample's position"
                )

        # 提取vision tokens: (batch_size, n_vision, hidden_size)
        vision_hidden = hidden_states[:, v_start:v_end+1, :].contiguous()
        n_vision = vision_hidden.shape[1]

        # 调用pruner获取soft mask
        # pruner返回: {'soft_mask': (batch, n_vision), ...}
        pruner_output = self.pruner(
            vision_hidden,
            self.question_embeddings,
            use_gumbel=self.training  # 训练时用Gumbel，推理时用确定性
        )
        soft_mask = pruner_output['soft_mask']  # (batch_size, n_vision)

        # 收集mask（用于sparsity loss计算）
        if self.collect_masks and self.mask_collector is not None:
            self.mask_collector.append(soft_mask.detach())

        # 应用mask: element-wise multiply
        # soft_mask: (batch, n_vision, 1)
        masked_vision = vision_hidden * soft_mask.unsqueeze(-1)

        # 替换回原始hidden_states
        hidden_states = hidden_states.clone()
        hidden_states[:, v_start:v_end+1, :] = masked_vision

        return hidden_states

    def set_pruning_context(
        self,
        vision_positions: torch.Tensor,
        question_embeddings: torch.Tensor,
        use_attn_residual: bool = False,
        collect_masks: bool = False,
        mask_collector: Optional[List] = None
    ):
        """设置pruning上下文（在forward前调用）"""
        self.vision_positions = vision_positions
        self.question_embeddings = question_embeddings
        self.use_attn_residual = use_attn_residual
        self.collect_masks = collect_masks
        self.mask_collector = mask_collector

    def clear_pruning_context(self):
        """清除pruning上下文（在forward后调用）"""
        self.vision_positions = None
        self.question_embeddings = None
        self.collect_masks = False
        self.mask_collector = None


class LLaVAWithPruning(nn.Module):
    """LLaVA模型 + 集成的Layer-wise Pruning

    通过替换指定的decoder layers为PruningLayerWrapper，实现无需外部hook的pruning。

    参数:
        original_model: 原始的LlavaForConditionalGeneration实例
        layer_pruners: LayerSpecificPruner实例
        config: 配置对象（包含pruning设置）
    """

    def __init__(
        self,
        original_model: nn.Module,
        layer_pruners: nn.Module,
        config: Any
    ):
        super().__init__()

        # 直接使用原始模型的所有属性
        # 这样可以保持完整的LLaVA功能（vision_tower, projector等）
        for key, value in original_model.__dict__.items():
            if not key.startswith('_'):
                setattr(self, key, value)

        self.layer_pruners = layer_pruners
        self.pruning_config = config

        # Cache the decoder layers reference (必须在 _wrap_decoder_layers 之前初始化)
        self._decoder_layers = None

        # 包装需要pruning的layers
        self._wrap_decoder_layers()

        # Pruning状态
        self.pruning_enabled = False
        self.current_vision_pos = None
        self.current_question_emb = None
        self.current_mask_collector = None

    def _get_decoder_layers(self):
        """获取decoder layers的统一方法"""
        if self._decoder_layers is not None:
            return self._decoder_layers

        language_model = self.model.language_model

        # LlamaModel 直接有 .layers 属性
        if hasattr(language_model, 'layers'):
            self._decoder_layers = language_model.layers
        elif hasattr(language_model, 'model') and hasattr(language_model.model, 'layers'):
            self._decoder_layers = language_model.model.layers
        else:
            raise AttributeError(
                f"Cannot find decoder layers in language_model (type: {type(language_model)}). "
                f"Available attributes: {dir(language_model)}"
            )

        return self._decoder_layers

    def _wrap_decoder_layers(self):
        """将指定的decoder layers替换为PruningLayerWrapper"""
        # Debug: 打印模型结构
        print(f"🔍 Debug: Checking model structure...")
        print(f"  - self.model type: {type(self.model)}")
        print(f"  - self.model.language_model type: {type(self.model.language_model)}")

        decoder_layers = self._get_decoder_layers()
        print(f"  - Found {len(decoder_layers)} decoder layers")

        for layer_idx in self.layer_pruners.get_all_layers():
            if layer_idx >= len(decoder_layers):
                raise ValueError(
                    f"Layer index {layer_idx} out of range. "
                    f"Model has {len(decoder_layers)} layers."
                )

            original_layer = decoder_layers[layer_idx]
            pruner = self.layer_pruners.get_pruner(layer_idx)

            # 创建wrapper
            wrapper = PruningLayerWrapper(original_layer, pruner, layer_idx)

            # 替换
            decoder_layers[layer_idx] = wrapper

        print(f"✓ Wrapped {len(self.layer_pruners.get_all_layers())} decoder layers with pruning")

    def enable_pruning(
        self,
        vision_positions: torch.Tensor,
        question_embeddings: torch.Tensor,
        use_attn_residual: bool = False,
        mask_collector: Optional[List] = None
    ):
        """启用pruning并设置上下文

        参数:
            vision_positions: (batch_size, 2) 或 (2,)
            question_embeddings: (batch_size, q_len, d)
            use_attn_residual: 是否使用attention residual
            mask_collector: 用于收集masks的列表
        """
        self.pruning_enabled = True
        self.current_vision_pos = vision_positions
        self.current_question_emb = question_embeddings
        self.current_mask_collector = mask_collector

        # 配置所有wrapped layers
        decoder_layers = self._get_decoder_layers()
        for layer_idx in self.layer_pruners.get_all_layers():
            wrapper = decoder_layers[layer_idx]
            if isinstance(wrapper, PruningLayerWrapper):
                wrapper.set_pruning_context(
                    vision_positions,
                    question_embeddings,
                    use_attn_residual,
                    collect_masks=(mask_collector is not None),
                    mask_collector=mask_collector
                )

    def disable_pruning(self):
        """禁用pruning并清除上下文"""
        self.pruning_enabled = False
        self.current_vision_pos = None
        self.current_question_emb = None
        self.current_mask_collector = None

        # 清除所有wrapped layers的context
        decoder_layers = self._get_decoder_layers()
        for layer_idx in self.layer_pruners.get_all_layers():
            wrapper = decoder_layers[layer_idx]
            if isinstance(wrapper, PruningLayerWrapper):
                wrapper.clear_pruning_context()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Forward pass（保持LLaVA原始接口）

        如果启用了pruning，会自动应用到wrapped layers。
        """
        # 调用原始模型的forward
        # 由于我们已经替换了decoder layers，pruning会自动发生
        return self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

    def preprocess_batch(self, images, questions, answers):
        """预处理batch（保持原有接口）

        这个方法应该在原始model上已经实现
        """
        if hasattr(self.model, 'preprocess_batch'):
            return self.model.preprocess_batch(images, questions, answers)
        else:
            raise NotImplementedError(
                "preprocess_batch not found. Make sure your backbone implements this method."
            )


def create_llava_with_pruning(
    backbone_model: nn.Module,
    layer_pruners: nn.Module,
    config: Any
) -> LLaVAWithPruning:
    """工厂函数：创建带pruning的LLaVA模型

    参数:
        backbone_model: 原始backbone（通常是从loader加载的）
        layer_pruners: LayerSpecificPruner实例
        config: 配置对象

    返回:
        LLaVAWithPruning实例
    """
    # 如果backbone已经被包装过，先解包
    if isinstance(backbone_model, LLaVAWithPruning):
        print("Warning: backbone already wrapped, using original model")
        # TODO: 提取original model

    return LLaVAWithPruning(backbone_model, layer_pruners, config)
