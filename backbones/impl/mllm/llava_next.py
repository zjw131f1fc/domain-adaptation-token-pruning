"""LLaVA-NeXT (1.6) Backbone 实现。

实现 LLaVA-NeXT 模型的功能：
- 一般生成方法
- 基于 embedding 的生成方法
- 获取完整 embedding 和 vision token 位置
- 支持动态高分辨率图像处理

LLaVA-NeXT 与 LLaVA 1.5 的主要区别：
- 使用 LlavaNextForConditionalGeneration 和 LlavaNextProcessor
- 支持动态高分辨率图像（会将图像分割成多个patch）
- 改进的视觉指令微调数据集
"""

import torch
import logging
from typing import Dict, Any, Optional, Tuple
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from ...base.mllm import BaseMLLMBackbone


class LLaVANextMLLMBackbone(BaseMLLMBackbone):
    """LLaVA-NeXT (1.6) MLLM Backbone 实现。
    
    LLaVA-NeXT 使用动态高分辨率图像处理:
    - 图像会被分割成多个 patch 并分别处理
    - 支持更高的图像分辨率
    - 通过 multi-modal projector 投影到 LLM 隐空间
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 LLaVA-NeXT Backbone。

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
        self.output_device = torch.device(self.device)

    def _load_model(self):
        """加载 LLaVA-NeXT 模型和处理器。"""
        logger = logging.getLogger(__name__)
        
        # 从配置获取模型 ID
        model_id = self.backbone_cfg.get("model_id", None)
        if model_id is None:
            # 如果未指定，根据 name 映射
            model_name = self.model_name
            model_map = {
                "llava-1.6-vicuna-7b": "llava-hf/llava-v1.6-vicuna-7b-hf",
                "llava-1.6-vicuna-13b": "llava-hf/llava-v1.6-vicuna-13b-hf",
                "llava-1.6-mistral-7b": "llava-hf/llava-v1.6-mistral-7b-hf",
            }
            model_id = model_map[model_name]
        
        logger.info(f"Loading LLaVA-NeXT model: {model_id} (this may take a while)...")
        device_map = self.mllm_cfg["device_map"]
        
        # 根据设备选择 dtype：GPU 用 float16，CPU 用 float32
        if device_map == "cpu" or self.device == "cpu":
            model_dtype = torch.float32
            device_map = "cpu"
        else:
            model_dtype = torch.float16
        
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=model_dtype,
        )
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        
        # 设置 pad_token（LLaVA-NeXT 使用 unk_token 作为 pad_token）
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.unk_token
        
        logger.info("LLaVA-NeXT model loaded successfully.")

    def _resize_image(self, image: Image.Image) -> Image.Image:
        """根据配置调整图像大小。
        
        参数:
            image: 原始图像
            
        返回:
            调整后的图像
        """
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
        """构建 LLaVA-NeXT 风格的对话 prompt。
        
        LLaVA-NeXT 使用 Vicuna 格式:
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

    def generate(
        self,
        image: Image.Image,
        question: str,
        max_new_tokens: int = 20
    ) -> str:
        """一般的生成方法。

        参数:
            image: PIL Image 对象
            question: 问题文本
            max_new_tokens: 最大生成 token 数

        返回:
            生成的文本答案
        """
        # 根据配置调整图像大小
        image = self._resize_image(image)
        
        prompt = self._build_prompt(question)
        
        inputs = self.processor(
            text=prompt,
            images=image.convert("RGB"),
            return_tensors="pt"
        )
        
        # 移动到模型设备
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        # 只解码新生成的 token
        input_length = inputs['input_ids'].shape[1]
        generated_ids = output_ids[:, input_length:]
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return answer

    def generate_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 20
    ) -> str:
        """利用 embedding 进行生成。

        参数:
            embeddings: 输入的 embedding 张量，shape 为 (batch_size, seq_len, hidden_dim)
            attention_mask: attention mask，shape 为 (batch_size, seq_len)
            max_new_tokens: 最大生成 token 数

        返回:
            生成的完整对话文本
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        answer = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return answer

    def get_embeddings(
        self,
        image: Image.Image,
        question: str,
        answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取完整的 embedding 和 vision token、answer token 的位置信息。

        LLaVA-NeXT 结构说明:
        - 图像会被分割成多个 patch（动态高分辨率）
        - 每个 patch 通过 vision_tower 编码
        - 通过 multi_modal_projector 投影为 hidden_dim
        - 与文本 embedding 在 <image> 占位符位置拼接

        参数:
            image: PIL Image 对象
            question: 问题文本
            answer: 答案文本（可选，如果提供则返回 answer 位置）

        返回:
            包含以下键的字典:
            - embeddings: torch.Tensor, shape (1, seq_len, hidden_dim)，完整的 embedding 序列
            - vision_token_positions: Tuple[int, int]，(start, end) vision token 的起始和结束位置
            - answer_token_positions: Tuple[int, int]，(start, end) answer token 的起始和结束位置（如果提供 answer）
            - input_ids: torch.Tensor, input token ids
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
            image_sizes = inputs.get('image_sizes', None)
            
            # 1) 获取文本 token 的嵌入
            text_token_embeds = self.model.get_input_embeddings()(input_ids)  # (1, T_text, D)
            
            # 2) 通过 model.get_image_features 获取图像特征
            # LLaVA-NeXT 需要传入 image_sizes 以支持动态分辨率
            image_features_list = self.model.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes
            )
            # 返回的是 list，每个元素对应一张图像
            # 对于单张图片，取第一个元素并增加 batch 维度
            image_token_embeds = image_features_list[0].unsqueeze(0)
            image_token_embeds = image_token_embeds.to(text_token_embeds.dtype)
            
            # 4) 找到图像占位 token 的位置
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
                "embeddings": full_embeddings,
                "attention_mask": full_attention_mask,
                "vision_token_positions": (vision_start, vision_end),
                "input_ids": input_ids,
                "num_vision_tokens": num_vision_tokens,
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
                
                answer_start = -total_added_len
                answer_end = answer_start + answer_len - 1
                
                result["answer_token_positions"] = (answer_start, answer_end)

        # 将所有 tensor 输出移到统一的设备上
        result["embeddings"] = result["embeddings"].to(self.output_device)
        result["attention_mask"] = result["attention_mask"].to(self.output_device)
        result["input_ids"] = result["input_ids"].to(self.output_device)

        return result

    def get_first_token_hidden_state(
        self,
        image: Image.Image,
        question: str
    ) -> Dict[str, Any]:
        """获取模型生成第一个 token 时的隐层状态。

        参数:
            image: PIL Image 对象
            question: 问题文本

        返回:
            包含以下键的字典:
            - hidden_state: torch.Tensor, shape (1, hidden_dim)
            - last_hidden_states: torch.Tensor, shape (num_layers, 1, seq_len, hidden_dim)
            - logits: torch.Tensor, shape (1, seq_len, vocab_size)
            - input_length: int
        """
        # 根据配置调整图像大小
        image = self._resize_image(image)
        
        prompt = self._build_prompt(question)
        
        inputs = self.processor(
            text=prompt,
            images=image.convert("RGB"),
            return_tensors="pt"
        )
        
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

            last_hidden_state = outputs.hidden_states[-1]
            input_length = last_hidden_state.shape[1]
            first_token_hidden = last_hidden_state[:, -1, :]
            logits = outputs.logits
            all_hidden_states = torch.stack(outputs.hidden_states, dim=0)

        return {
            "hidden_state": first_token_hidden.to(self.output_device),
            "last_hidden_states": all_hidden_states.to(self.output_device),
            "logits": logits.to(self.output_device),
            "input_length": input_length,
        }

    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = False
    ) -> Dict[str, Any]:
        """从 embeddings 直接 forward（支持 batch）。
        
        参数:
            embeddings: (batch, seq_len, hidden_dim)
            attention_mask: (batch, seq_len)
            output_hidden_states: 是否输出所有层的隐层状态
        
        返回:
            包含以下键的字典:
            - logits: torch.Tensor, 输出 logits
            - all_hidden_states: 如果 output_hidden_states=True
        """
        # LlavaNextForConditionalGeneration 结构与 LLaVA 1.5 类似
        outputs = self.model.language_model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # 使用 lm_head
        logits = self.model.lm_head(outputs.last_hidden_state)
        
        # 将输出移到统一的设备上
        result = {
            'logits': logits.to(self.output_device)
        }
        
        if output_hidden_states:
            result['all_hidden_states'] = tuple(h.to(self.output_device) for h in outputs.hidden_states)
        
        return result

    def get_vision_features(
        self,
        image: Image.Image
    ) -> torch.Tensor:
        """单独获取图像的 vision features（经过 projector 后）。
        
        参数:
            image: PIL Image 对象
            
        返回:
            vision_features: torch.Tensor, shape (1, num_patches, hidden_dim)
        """
        image = self._resize_image(image)
        
        inputs = self.processor(
            images=image.convert("RGB"),
            return_tensors="pt"
        )
        
        target_device = next(self.model.parameters()).device
        pixel_values = inputs['pixel_values'].to(target_device)
        image_sizes = inputs.get('image_sizes', None)
        if image_sizes is not None:
            image_sizes = image_sizes.to(target_device)
        
        with torch.no_grad():
            # LLaVA-NeXT 需要 image_sizes 参数
            image_features_list = self.model.get_image_features(
                pixel_values=pixel_values,
                image_sizes=image_sizes
            )
            # 返回的是 list，每个元素 shape 为 (num_patches, hidden_dim)
            image_features = image_features_list[0].unsqueeze(0)
        
        # 将输出移到统一的设备上
        return image_features.to(self.output_device)
