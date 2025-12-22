"""LLaVA-1.5-7B Backbone 实现。

实现 LLaVA-1.5 模型的功能：
- 统一的生成方法（支持 text 和 embedding 两种模式）
- 获取完整 embedding 和 vision token 位置
- 支持层间 hook 的 forward 方法
- 支持 vision tower 结构
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple, Callable, List
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from ...base.mllm import BaseMLLMBackbone


class LLaVAMLLMBackbone(BaseMLLMBackbone):
    """LLaVA-1.5 MLLM Backbone 实现。
    
    LLaVA 使用 vision tower 结构:
    - 图像通过 vision tower (CLIP ViT) 编码为 patch embeddings
    - 通过 multi-modal projector 投影到 LLM 隐空间
    - 与文本 token 拼接后送入 LLM (Vicuna/LLaMA)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 LLaVA Backbone。

        参数:
            config: Config 实例，包含 backbone_settings 等配置
        """
        super().__init__(config)
        self.device = self.config.global_settings["device"]  # type: ignore
        self.mllm_cfg = self.backbone_cfg["mllm_settings"]
        self.image_max_size = self.mllm_cfg["image_max_size"]
        
        self._load_model()
        
        # 获取模型实际所在的设备（当使用 device_map 时）
        if self.model is not None:
            model_device = next(self.model.parameters()).device
            self.device = str(model_device)
        
        # 统一输出设备：当使用 device_map 分布在多个 GPU 时，将输出统一移到第一个参数所在的设备
        # 这样可以避免跨设备梯度计算的问题，同时支持 CPU 调试
        self.output_device = torch.device(self.device)

    def _load_model(self):
        """加载 LLaVA-1.5 模型和处理器。"""
        # 优先使用 config 中的 logger，保证日志格式一致
        logger = getattr(self.config, "logger", None) or logging.getLogger(__name__)
        
        # 从配置获取模型 ID
        model_id = self.backbone_cfg.get("model_id", None)
        if model_id is None:
            # 如果未指定，根据 name 映射
            model_name = self.model_name
            model_map = {
                "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
                "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
            }
            model_id = model_map[model_name]
        
        logger.info(f"Loading LLaVA model: {model_id} (this may take a while)...")
        device_map = self.mllm_cfg["device_map"]

        # 根据设备选择 dtype：GPU 用 float16，CPU 用 float32
        if device_map == "cpu" or self.device == "cpu":
            model_dtype = torch.float32
            device_map = "cpu"
        else:
            model_dtype = torch.float32

        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=model_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        # 设置 pad_token（LLaVA 使用 unk_token 作为 pad_token）
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.unk_token

        # 打印模型信息
        num_layers = len(self.model.language_model.layers)
        hidden_size = self.model.config.text_config.hidden_size
        vocab_size = self.model.config.text_config.vocab_size

        logger.info(f"[LLaVA] Model: {model_id}")
        logger.info(f"[LLaVA] LLM Layers: {num_layers} | Hidden: {hidden_size} | Vocab: {vocab_size}")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """根据配置调整图像大小。

        参数:
            image: 原始图像

        返回:
            调整后的图像
        """
        # 检查是否启用统一尺寸
        enable_true_batch = self.mllm_cfg.get("enable_true_batch", False)

        if enable_true_batch:
            # 启用batch化：强制调整为统一的正方形尺寸
            unified_size = self.mllm_cfg.get("unified_image_size", 336)
            return image.resize((unified_size, unified_size), Image.Resampling.LANCZOS)

        # 原有逻辑：等比例缩放到不超过 image_max_size
        if self.image_max_size is None:
            return image

        width, height = image.size
        max_dim = max(width, height)

        if max_dim <= self.image_max_size:
            return image

        # 等比例缩放
        scale = self.image_max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def _build_prompt(self, question: str, answer: Optional[str] = None) -> str:
        """构建 LLaVA 风格的对话 prompt。

        LLaVA 1.5 使用 Vicuna 格式:
        USER: <image>\n{question}
        ASSISTANT: {answer}

        参数:
            question: 问题文本
            answer: 答案文本（可选）

        返回:
            格式化的 prompt 字符串
        """
        if answer is None:
            # 生成模式：不包含答案，等待模型生成
            prompt = f"USER: <image>\n{question}\nASSISTANT:"
        else:
            # 训练/评估模式：包含完整对话
            prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
        return prompt

    def _get_vision_token_positions(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: List[torch.Tensor]
    ) -> Tuple[int, int, torch.Tensor]:
        """使用库提供的 get_placeholder_mask 方法获取 vision token 位置。

        参数:
            input_ids: (batch, seq_len)
            inputs_embeds: (batch, seq_len, hidden_dim)
            image_features: list of (num_patches, hidden_dim)

        返回:
            vision_start: int, vision token 起始位置
            vision_end: int, vision token 结束位置
            vision_indices: torch.Tensor, 所有 vision token 的索引
        """
        # 1. 获取 placeholder mask
        special_image_mask = self.model.model.get_placeholder_mask(
            input_ids, inputs_embeds, image_features
        )

        # 2. 压缩维度到 2D (batch, seq_len)
        mask_2d = special_image_mask.any(dim=-1)

        # 3. 提取 vision token 的位置索引
        vision_indices = torch.where(mask_2d[0])[0]

        if len(vision_indices) == 0:
            raise ValueError("未找到图像占位 token")

        vision_start = int(vision_indices[0])
        vision_end = int(vision_indices[-1])

        return vision_start, vision_end, vision_indices

    def generate(
        self,
        # === 方式1: text模式 ===
        image: Optional[Image.Image] = None,
        question: Optional[str] = None,
        # === 方式2: embedding模式 ===
        embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        # === Hook 参数 ===
        prefill_modifier: Optional[Callable] = None,
        hook_layer_idx: Optional[int] = None,
        # === 通用参数 ===
        max_new_tokens: int = 20,
        do_sample: bool = False,
        **generation_kwargs
    ) -> str:
        """统一的生成接口，支持两种输入模式和可选的 Prefill 修改器。

        模式1 - Text模式:
            generate(image=img, question="...", max_new_tokens=20)

        模式2 - Embedding模式:
            generate(embeddings=emb, attention_mask=mask, max_new_tokens=20)

        Prefill 修改器（可选）:
            用户提供一个函数，在 Prefill 阶段修改 hidden_states 和 attention_mask。
            Backbone 会自动包装成 hook，并在 Decode 阶段跳过。

        参数:
            image: PIL Image 对象（text模式）
            question: 问题文本（text模式）
            embeddings: 输入embeddings (batch, seq_len, hidden_dim)（embedding模式）
            attention_mask: attention mask（embedding模式）
            prefill_modifier: 用户自定义的 Prefill 修改函数，签名为:
                def modifier(hidden_states, attention_mask) -> Tuple[Tensor, Tensor]
                    参数:
                        hidden_states: (batch, seq_len, hidden_dim)
                        attention_mask: (batch, seq_len)
                    返回:
                        (new_hidden_states, new_attention_mask)
            hook_layer_idx: 在哪一层应用修改器 (0-31)
            max_new_tokens: 最大生成token数
            do_sample: 是否采样
            **generation_kwargs: 其他生成参数

        返回:
            生成的文本答案

        示例 - 剪枝场景:
            ```python
            def my_pruning_logic(hidden_states, attention_mask):
                # 只关心核心逻辑，不用判断 Prefill/Decode
                v_start, v_end = 10, 585  # vision token 位置
                vision_hidden = hidden_states[:, v_start:v_end+1, :]

                # 剪枝逻辑
                hard_mask = pruner(vision_hidden)
                kept_indices = torch.nonzero(hard_mask[0, :, 0]).squeeze(-1)
                pruned_vision = vision_hidden[:, kept_indices, :]

                # 重新拼接
                new_hidden = torch.cat([
                    hidden_states[:, :v_start, :],
                    pruned_vision,
                    hidden_states[:, v_end+1:, :]
                ], dim=1)

                new_mask = torch.cat([
                    attention_mask[:, :v_start],
                    torch.ones(1, len(kept_indices), device=attention_mask.device),
                    attention_mask[:, v_end+1:]
                ], dim=1)

                return new_hidden, new_mask

            # 使用（Backbone 自动处理 Prefill/Decode 判断）
            answer = backbone.generate(
                embeddings=emb,
                attention_mask=mask,
                prefill_modifier=my_pruning_logic,
                hook_layer_idx=10,
                max_new_tokens=20
            )
            ```
        """
        hook_handle = None

        # 如果提供了 prefill_modifier，包装成 hook
        if prefill_modifier is not None and hook_layer_idx is not None:
            def prefill_hook_wrapper(module, args):
                """自动判断 Prefill/Decode，只在 Prefill 阶段调用用户函数"""
                hidden_states = args[0]

                # Decode 阶段：直接跳过
                if hidden_states.shape[1] == 1:
                    return args

                # Prefill 阶段：调用用户提供的修改函数
                attention_mask = args[1] if len(args) > 1 else None

                # 调用用户函数
                new_hidden, new_mask = prefill_modifier(hidden_states, attention_mask)

                # 重新组装 args
                new_args = [new_hidden, new_mask] + list(args[2:])
                return tuple(new_args)

            # 注册 hook
            target_layer = self.model.model.language_model.layers[hook_layer_idx]
            hook_handle = target_layer.register_forward_pre_hook(prefill_hook_wrapper)

        try:
            # 自动判断输入类型
            if image is not None and question is not None:
                # text模式：调用processor
                image = self._resize_image(image)
                prompt = self._build_prompt(question)
                inputs = self.processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
                target_device = next(self.model.parameters()).device
                inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v
                          for k, v in inputs.items()}

                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                                    do_sample=do_sample, **generation_kwargs)
                input_length = inputs['input_ids'].shape[1]

            elif embeddings is not None and attention_mask is not None:
                # embedding模式：直接使用
                with torch.no_grad():
                    output_ids = self.model.generate(inputs_embeds=embeddings,
                                                    attention_mask=attention_mask,
                                                    max_new_tokens=max_new_tokens,
                                                    do_sample=do_sample, **generation_kwargs)
                input_length = 0
            else:
                raise ValueError("必须提供 (image, question) 或 (embeddings, attention_mask)")

            # 解码（仅新生成的部分）
            generated_ids = output_ids[:, input_length:]
            answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            return answer

        finally:
            # 确保 hook 被清理
            if hook_handle is not None:
                hook_handle.remove()

    def preprocess(
        self,
        image: Image.Image,
        question: str,
        answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """预处理输入，将图像和文本转换为 embedding 序列。

        LLaVA 结构说明:
        - 图像通过 vision_tower 编码为 (num_patches, vision_dim)
        - 通过 multi_modal_projector 投影为 (num_patches, hidden_dim)
        - 与文本 embedding 在 <image> 占位符位置拼接

        参数:
            image: PIL Image 对象
            question: 问题文本
            answer: 答案文本（可选，如果提供则返回 answer 位置）

        返回:
            包含以下键的字典:
            - embeddings: torch.Tensor, shape (1, seq_len, hidden_dim)
            - attention_mask: torch.Tensor, shape (1, seq_len)
            - vision_token_positions: Tuple[int, int]，(start, end) vision token 的位置
            - answer_token_positions: Tuple[int, int]，(start, end) answer token 的位置（如果提供 answer）
        """
        # 根据配置调整图像大小
        image = self._resize_image(image)
        
        prompt = self._build_prompt(question, answer)
        
        inputs = self.processor(
            text=prompt,
            images=image.convert("RGB"),
            return_tensors="pt"
        )
        
        # 移动到模型设备
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            input_ids = inputs['input_ids']
            pixel_values = inputs['pixel_values']
            
            # 1) 获取文本 token 的嵌入
            text_token_embeds = self.model.get_input_embeddings()(input_ids)  # (1, T_text, D)

            # 2) 只调用一次 vision_tower，同时获取 raw 和 projected features
            # 避免两次调用导致的数值不一致
            raw_vision_features = None
            image_token_embeds = None

            if hasattr(self.model, 'vision_tower') and hasattr(self.model, 'multi_modal_projector'):
                vision_tower = self.model.vision_tower
                projector = self.model.multi_modal_projector

                # 调用 vision_tower 获取原始 features（只调用一次！）
                vision_outputs = vision_tower(pixel_values, output_hidden_states=True)

                # 获取指定层的 hidden states（默认使用 vision_feature_layer）
                vision_feature_layer = self.model.config.vision_feature_layer
                if isinstance(vision_feature_layer, int):
                    selected_features = vision_outputs.hidden_states[vision_feature_layer]
                else:
                    # 多层拼接的情况
                    selected_features = torch.cat(
                        [vision_outputs.hidden_states[idx] for idx in vision_feature_layer],
                        dim=-1
                    )

                # 根据 vision_feature_select_strategy 决定是否去掉 CLS
                vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
                if vision_feature_select_strategy == "default":
                    selected_features = selected_features[:, 1:]  # 去掉 CLS token

                # raw_vision_features: 未投影的 CLIP 输出 (1, 576, 1024)
                raw_vision_features = selected_features

                # 通过 projector 投影到 LLM hidden dim (1, 576, 4096)
                image_token_embeds = projector(selected_features)
                image_token_embeds = image_token_embeds.to(text_token_embeds.dtype)
            else:
                # Fallback: 使用 get_image_features（会额外调用一次 vision_tower）
                image_features_list = self.model.get_image_features(pixel_values=pixel_values)
                image_token_embeds = torch.cat(image_features_list, dim=0).unsqueeze(0)
                image_token_embeds = image_token_embeds.to(text_token_embeds.dtype)
            
            # 4) 找到图像占位 token 的位置
            # LLaVA processor 会将 <image> 扩展为多个 image_token_id（通常是 576 个）
            image_token_id = self.model.config.image_token_index
            image_token_indices = torch.where(input_ids[0] == image_token_id)[0]
            
            if len(image_token_indices) == 0:
                raise ValueError("未找到图像占位 token (<image>)，请检查输入格式")
            
            # 获取 image token 区间的起始和结束位置
            img_token_start_idx = int(image_token_indices[0])
            img_token_end_idx = int(image_token_indices[-1])
            
            # 5) 构建完整序列: 文本前段 + 视觉token + 文本后段
            text_embeds_part1 = text_token_embeds[:, :img_token_start_idx, :]
            text_embeds_part2 = text_token_embeds[:, img_token_end_idx + 1:, :]
            
            full_embeddings = torch.cat([
                text_embeds_part1,
                image_token_embeds,
                text_embeds_part2
            ], dim=1)  # (1, T_full, D)
            
            # 构建对应的 attention mask
            num_vision_tokens = image_token_embeds.shape[1]
            vision_attention = torch.ones(
                (1, num_vision_tokens), 
                dtype=torch.long, 
                device=target_device
            )
            
            if 'attention_mask' in inputs:
                attention_part1 = inputs['attention_mask'][:, :img_token_start_idx]
                attention_part2 = inputs['attention_mask'][:, img_token_end_idx + 1:]
                full_attention_mask = torch.cat([
                    attention_part1,
                    vision_attention,
                    attention_part2
                ], dim=1)
            else:
                full_attention_mask = torch.ones(
                    (1, full_embeddings.shape[1]),
                    dtype=torch.long,
                    device=target_device
                )
            
            # vision token 在完整序列中的位置
            vision_start = text_embeds_part1.shape[1]
            vision_end = vision_start + num_vision_tokens - 1

            result = {
                "embeddings": full_embeddings.to(self.output_device),
                "attention_mask": full_attention_mask.to(self.output_device),
                "vision_token_positions": (vision_start, vision_end),
                "raw_vision_features": raw_vision_features.to(self.output_device) if raw_vision_features is not None else None,  # 添加未投影的vision features
            }

            # 如果提供了 answer，计算 answer token 位置
            if answer:
                # 构建不带 answer 的 prompt
                prompt_no_answer = self._build_prompt(question)

                inputs_no_answer = self.processor(
                    text=prompt_no_answer,
                    images=image.convert("RGB"),
                    return_tensors="pt"
                )

                # 计算 answer token 的长度差异
                total_added_len = input_ids.shape[1] - inputs_no_answer['input_ids'].shape[1]

                # 单独 tokenize answer
                answer_tokens = self.processor.tokenizer(
                    answer,
                    add_special_tokens=False,
                    return_tensors='pt'
                )['input_ids']
                answer_len = answer_tokens.shape[1]

                # 使用负索引定位 answer 位置（相对于完整序列末尾）
                answer_start = -total_added_len
                answer_end = answer_start + answer_len - 1

                result["answer_token_positions"] = (answer_start, answer_end)

        return result

    def preprocess_batch(
        self,
        images: List[Image.Image],
        questions: List[str],
        answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """批量预处理输入（真正的batch化）。

        注意：
        - 要求所有图片已统一尺寸（通过 _resize_image 实现）
        - 要求启用 enable_true_batch 配置
        - 所有样本的 vision token 数量必须相同

        参数:
            images: PIL Image 对象列表，batch_size 个
            questions: 问题文本列表，batch_size 个
            answers: 答案文本列表，batch_size 个（可选）

        返回:
            包含以下键的字典:
            - embeddings: torch.Tensor, shape (batch_size, seq_len, hidden_dim)
            - attention_mask: torch.Tensor, shape (batch_size, seq_len)
            - vision_token_positions: torch.Tensor, shape (batch_size, 2)，每行为 (start, end)
            - answer_token_positions: torch.Tensor, shape (batch_size, 2)（如果提供 answer）
            - raw_vision_features: torch.Tensor, shape (batch_size, n_vision_tokens, 1024)

        抛出:
            ValueError: 当未启用 enable_true_batch 或序列长度不一致时
        """
        # 检查配置
        enable_true_batch = self.mllm_cfg.get("enable_true_batch", False)
        if not enable_true_batch:
            raise ValueError(
                "preprocess_batch requires enable_true_batch=True in config. "
                "Please set backbone_settings.mllm_settings.enable_true_batch=true"
            )

        batch_size = len(images)
        if len(questions) != batch_size:
            raise ValueError(f"images and questions length mismatch: {batch_size} vs {len(questions)}")
        if answers is not None and len(answers) != batch_size:
            raise ValueError(f"images and answers length mismatch: {batch_size} vs {len(answers)}")

        # 1. 调整所有图片到统一尺寸
        resized_images = [self._resize_image(img) for img in images]

        # 2. 构建所有prompts
        if answers is not None:
            prompts = [self._build_prompt(q, a) for q, a in zip(questions, answers)]
        else:
            prompts = [self._build_prompt(q) for q in questions]

        # 3. Batch处理所有输入
        inputs = self.processor(
            text=prompts,
            images=[img.convert("RGB") for img in resized_images],
            return_tensors="pt",
            padding=True  # 自动padding到相同长度
        )

        # 移动到模型设备
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            input_ids = inputs['input_ids']  # (batch_size, seq_len)
            pixel_values = inputs['pixel_values']  # (batch_size, 3, H, W)

            # 4. 获取文本 token 的嵌入
            text_token_embeds = self.model.get_input_embeddings()(input_ids)  # (batch_size, seq_len, hidden_dim)

            # 5. 批量调用 vision_tower
            raw_vision_features = None
            image_token_embeds = None

            if hasattr(self.model, 'vision_tower') and hasattr(self.model, 'multi_modal_projector'):
                vision_tower = self.model.vision_tower
                projector = self.model.multi_modal_projector

                # Batch调用 vision_tower
                vision_outputs = vision_tower(pixel_values, output_hidden_states=True)

                # 获取指定层的 hidden states
                vision_feature_layer = self.model.config.vision_feature_layer
                if isinstance(vision_feature_layer, int):
                    selected_features = vision_outputs.hidden_states[vision_feature_layer]
                else:
                    selected_features = torch.cat(
                        [vision_outputs.hidden_states[idx] for idx in vision_feature_layer],
                        dim=-1
                    )

                # 根据策略决定是否去掉 CLS
                vision_feature_select_strategy = self.model.config.vision_feature_select_strategy
                if vision_feature_select_strategy == "default":
                    selected_features = selected_features[:, 1:]  # 去掉 CLS token

                # raw_vision_features: (batch_size, n_vision_tokens, 1024)
                raw_vision_features = selected_features

                # 通过 projector 投影
                image_token_embeds = projector(selected_features)  # (batch_size, n_vision_tokens, hidden_dim)
                image_token_embeds = image_token_embeds.to(text_token_embeds.dtype)
            else:
                raise ValueError("Batch mode requires direct access to vision_tower and multi_modal_projector")

            # 6. 找到每个样本的图像占位 token 位置
            image_token_id = self.model.config.image_token_index
            vision_positions_list = []

            for i in range(batch_size):
                image_token_indices = torch.where(input_ids[i] == image_token_id)[0]
                if len(image_token_indices) == 0:
                    raise ValueError(f"Sample {i}: 未找到图像占位 token (<image>)")

                img_token_start_idx = int(image_token_indices[0])
                img_token_end_idx = int(image_token_indices[-1])
                vision_positions_list.append((img_token_start_idx, img_token_end_idx))

            # 检查所有样本的vision token位置是否一致（batch化要求）
            first_start, first_end = vision_positions_list[0]
            for i, (start, end) in enumerate(vision_positions_list[1:], 1):
                if start != first_start or end != first_end:
                    raise ValueError(
                        f"Vision token positions not aligned across batch: "
                        f"sample 0 has ({first_start}, {first_end}), sample {i} has ({start}, {end}). "
                        f"This usually means text sequences have different lengths. "
                        f"Please ensure enable_true_batch mode uses unified image sizes."
                    )

            # 7. 批量构建完整序列
            img_token_start_idx, img_token_end_idx = first_start, first_end
            num_vision_tokens = image_token_embeds.shape[1]

            text_embeds_part1 = text_token_embeds[:, :img_token_start_idx, :]
            text_embeds_part2 = text_token_embeds[:, img_token_end_idx + 1:, :]

            full_embeddings = torch.cat([
                text_embeds_part1,
                image_token_embeds,
                text_embeds_part2
            ], dim=1)  # (batch_size, seq_len, hidden_dim)

            # 8. 构建对应的 attention mask
            vision_attention = torch.ones(
                (batch_size, num_vision_tokens),
                dtype=torch.long,
                device=target_device
            )

            if 'attention_mask' in inputs:
                attention_part1 = inputs['attention_mask'][:, :img_token_start_idx]
                attention_part2 = inputs['attention_mask'][:, img_token_end_idx + 1:]
                full_attention_mask = torch.cat([
                    attention_part1,
                    vision_attention,
                    attention_part2
                ], dim=1)
            else:
                full_attention_mask = torch.ones(
                    (batch_size, full_embeddings.shape[1]),
                    dtype=torch.long,
                    device=target_device
                )

            # 9. vision token 在完整序列中的位置（所有样本相同）
            vision_start = text_embeds_part1.shape[1]
            vision_end = vision_start + num_vision_tokens - 1

            # 转为 (batch_size, 2) 的tensor
            vision_positions_tensor = torch.tensor(
                [[vision_start, vision_end]] * batch_size,
                dtype=torch.long,
                device=self.output_device
            )

            result = {
                "embeddings": full_embeddings.to(self.output_device),
                "attention_mask": full_attention_mask.to(self.output_device),
                "vision_token_positions": vision_positions_tensor,  # (batch_size, 2)
                "raw_vision_features": raw_vision_features.to(self.output_device) if raw_vision_features is not None else None,
            }

            # 10. 如果提供了 answers，批量计算 answer token 位置
            if answers:
                answer_positions_list = []

                for i, (question, answer) in enumerate(zip(questions, answers)):
                    # 构建不带 answer 的 prompt
                    prompt_no_answer = self._build_prompt(question)

                    inputs_no_answer = self.processor(
                        text=prompt_no_answer,
                        images=resized_images[i].convert("RGB"),
                        return_tensors="pt"
                    )

                    # 计算 answer token 的长度差异
                    total_added_len = input_ids[i].shape[0] - inputs_no_answer['input_ids'].shape[1]

                    # 单独 tokenize answer
                    answer_tokens = self.processor.tokenizer(
                        answer,
                        add_special_tokens=False,
                        return_tensors='pt'
                    )['input_ids']
                    answer_len = answer_tokens.shape[1]

                    # 使用负索引定位 answer 位置
                    answer_start = -total_added_len
                    answer_end = answer_start + answer_len - 1

                    answer_positions_list.append((answer_start, answer_end))

                # 转为 (batch_size, 2) 的tensor
                answer_positions_tensor = torch.tensor(
                    answer_positions_list,
                    dtype=torch.long,
                    device=self.output_device
                )
                result["answer_token_positions"] = answer_positions_tensor

        return result

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_positions: Optional[Tuple[int, int]] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        # === 层修改器参数 ===
        layer_modifier: Optional[Callable] = None,
        hook_layer_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        """从 embeddings 直接 forward，支持层间修改器。

        参数:
            embeddings: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            vision_positions: (start, end) vision token 在序列中的位置（元信息，可选）
            output_hidden_states: 是否输出所有层的隐层状态
            output_attentions: 是否输出所有层的attention weights
            layer_modifier: 用户自定义的层修改函数，签名为:
                def modifier(hidden_states, attention_mask) -> Tuple[Tensor, Tensor]
                    参数:
                        hidden_states: (batch, seq_len, hidden_dim)
                        attention_mask: (batch, seq_len)
                    返回:
                        (new_hidden_states, new_attention_mask)
            hook_layer_idx: 在哪一层应用修改器 (0-31)

        返回:
            包含以下键的字典:
            - logits: torch.Tensor, 输出 logits (batch, seq_len, vocab_size)
            - all_hidden_states: tuple[torch.Tensor], 所有层的隐层状态（如果 output_hidden_states=True）
            - attentions: tuple[torch.Tensor], 所有层的attention weights（如果 output_attentions=True）
                每个元素形状为 (batch, num_heads, seq_len, seq_len)

        示例 - 剪枝场景:
            ```python
            def my_pruning_logic(hidden_states, attention_mask):
                # 只关心核心逻辑
                v_start, v_end = 10, 585
                vision_hidden = hidden_states[:, v_start:v_end+1, :]

                # 剪枝
                hard_mask = pruner(vision_hidden)
                kept_indices = torch.nonzero(hard_mask[0, :, 0]).squeeze(-1)
                pruned_vision = vision_hidden[:, kept_indices, :]

                # 重新拼接
                new_hidden = torch.cat([
                    hidden_states[:, :v_start, :],
                    pruned_vision,
                    hidden_states[:, v_end+1:, :]
                ], dim=1)

                new_mask = torch.cat([
                    attention_mask[:, :v_start],
                    torch.ones(1, len(kept_indices), device=attention_mask.device),
                    attention_mask[:, v_end+1:]
                ], dim=1)

                return new_hidden, new_mask

            result = backbone.forward(
                embeddings=emb,
                attention_mask=mask,
                layer_modifier=my_pruning_logic,
                hook_layer_idx=10
            )
            ```
        """
        hook_handle = None

        # 如果提供了 layer_modifier，包装成 hook
        if layer_modifier is not None and hook_layer_idx is not None:
            def hook_wrapper(module, args):
                """包装用户函数为 hook"""
                hidden_states = args[0]
                attention_mask = args[1] if len(args) > 1 else None

                # 调用用户函数
                new_hidden, new_mask = layer_modifier(hidden_states, attention_mask)

                # 重新组装 args
                return (new_hidden, new_mask, *args[2:])

            target_layer = self.model.model.language_model.layers[hook_layer_idx]
            hook_handle = target_layer.register_forward_pre_hook(hook_wrapper)

        try:
            # 调用 language_model（hook 会自动执行）
            outputs = self.model.model.language_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True
            )

            # 使用 lm_head 计算 logits
            logits = self.model.lm_head(outputs.last_hidden_state)

            result = {
                'logits': logits.to(self.output_device)
            }

            if output_hidden_states:
                result['all_hidden_states'] = tuple(
                    h.to(self.output_device) for h in outputs.hidden_states
                )

            if output_attentions:
                result['attentions'] = tuple(
                    attn.to(self.output_device) for attn in outputs.attentions
                )

            return result

        finally:
            # 确保 hook 被清理
            if hook_handle is not None:
                hook_handle.remove()
