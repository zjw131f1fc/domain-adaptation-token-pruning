"""Qwen2.5-VL Backbone 实现。

实现 Qwen2.5-VL 模型的功能：
- 一般生成方法
- 基于 embedding 的生成方法
- 获取完整 embedding 和 vision token 位置
- 支持软 mask 的 forward 方法
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from ...base.mllm import BaseMLLMBackbone


class QwenMLLMBackbone(BaseMLLMBackbone):
    """Qwen2.5-VL MLLM Backbone 实现。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 Qwen Backbone。

        参数:
            config: Config 实例，包含 backbone_settings 等配置
        """
        super().__init__(config)
        self.device = self.config.global_settings["device"] # type: ignore
        self.mllm_cfg = self.backbone_cfg["mllm_settings"]
        self.image_max_size = self.mllm_cfg["image_max_size"]
        # 确保在加载模型前设置环境变量（通过 get_config 已经设置）
        self._load_model()
        # 获取模型实际所在的设备（当使用 device_map 时）
        if self.model is not None:
            # 获取第一个参数的设备作为模型的主要设备
            model_device = next(self.model.parameters()).device
            self.device = str(model_device)
        


    def _load_model(self):
        """加载 Qwen2.5-VL 模型和处理器。"""
        logger = logging.getLogger(__name__)
        
        # 从配置获取模型 ID
        model_id = self.backbone_cfg.get("model_id", None)
        if model_id is None:
            # 如果未指定，根据 name 映射
            model_name = self.model_name
            model_map = {
                "qwen-2.5-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
            }
            model_id = model_map[model_name]
        
        logger.info(f"Loading Qwen2.5-VL model: {model_id} (this may take a while)...")
        device_map = self.mllm_cfg["device_map"]
        cache_dir = self.config["global_settings"]["hf_cache_dir"] # type: ignore
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, 
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # 移除generation_config中不兼容的参数，避免警告
        if hasattr(self.model, 'generation_config'):
            if hasattr(self.model.generation_config, 'temperature'):
                self.model.generation_config.temperature = None
        
        logger.info("Qwen2.5-VL model loaded successfully.")

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
        
        message = [{
            "role": "user", 
            "content": [
                {"type": "image", "image": image}, 
                {"type": "text", "text": question}
            ]
        }]

        inputs = self.processor(
            text=[self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)],
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt"
        )
        
        # 始终将inputs移动到模型第一个参数所在的设备
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 确保创建 attention_mask（如果 processor 没有返回）
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            # 手动创建 attention_mask：所有非 padding 的位置为 1
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                attention_mask = (inputs['input_ids'] != pad_token_id).long()
            else:
                # 如果 pad_token_id 为 None，创建全 1 的 mask（对于单样本通常不会有 padding）
                attention_mask = torch.ones_like(inputs['input_ids'], dtype=torch.long)
            inputs['attention_mask'] = attention_mask

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                image_grid_thw=inputs['image_grid_thw'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=max_new_tokens,
                do_sample=False  # 确保确定性生成
            )

        # 只解码新生成的 token，不包括输入部分
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
        
        注意: 此方法不涉及图像处理，图像缩放应在调用get_embeddings时完成。
        """
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False  # 确保确定性生成
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

        参数:
            image: PIL Image 对象
            question: 问题文本
            answer: 答案文本（可选，如果提供则返回 answer 位置）

        返回:
            包含以下键的字典:
            - embeddings: torch.Tensor, shape (1, seq_len, hidden_dim)，完整的 embedding 序列
            - vision_token_positions: Tuple[int, int]，(start, end) vision token 的起始和结束位置
            - answer_token_positions: Tuple[int, int]，(start, end) answer token 的起始和结束位置（如果提供 answer）
            - input_ids: torch.Tensor, input token ids（用于定位 answer）
        """
        # 根据配置调整图像大小
        image = self._resize_image(image)
        
        message = [{
            "role": "user", 
            "content": [
                {"type": "image", "image": image}, 
                {"type": "text", "text": question}
            ]
        }]
        
        # 如果提供了 answer，添加到 message 中
        if answer:
            message.append({
                "role": "assistant",
                "content": answer
            })

        inputs = self.processor(
            text=[self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=(answer is None))],
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt"
        )
        
        # 始终将inputs移动到模型第一个参数所在的设备
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 确保创建 attention_mask（如果 processor 没有返回）
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            # 手动创建 attention_mask：所有非 padding 的位置为 1
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                attention_mask = (inputs['input_ids'] != pad_token_id).long()
            else:
                # 如果 pad_token_id 为 None，创建全 1 的 mask（对于单样本通常不会有 padding）
                attention_mask = torch.ones_like(inputs['input_ids'], dtype=torch.long)
            inputs['attention_mask'] = attention_mask

        with torch.no_grad():
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # 1) 文本 token 的嵌入
            text_token_embeds = self.model.get_input_embeddings()(input_ids)  # (1, T_text, D)

            # 2) 图像特征（展开成视觉 token 序列）
            image_features_tuple = self.model.get_image_features(
                pixel_values=inputs['pixel_values'], 
                image_grid_thw=inputs['image_grid_thw']
            )
            image_token_embeds = torch.cat(image_features_tuple, dim=0).unsqueeze(0)  # (1, T_img, D)

            # 3) 找到图像占位 token 的区间，并构建完整序列
            image_token_id = self.model.config.image_token_id
            image_token_indices = torch.where(input_ids[0] == image_token_id)[0]

            if len(image_token_indices) == 0:
                raise ValueError("未找到图像占位 token，请检查输入格式")

            img_token_start_idx = int(image_token_indices[0])
            img_token_end_idx = int(image_token_indices[-1])

            # 构建完整序列: 文本前段 + 视觉token + 文本后段
            text_embeds_part1 = text_token_embeds[:, :img_token_start_idx, :]
            text_embeds_part2 = text_token_embeds[:, img_token_end_idx + 1:, :]
            
            full_embeddings = torch.cat([
                text_embeds_part1,
                image_token_embeds,
                text_embeds_part2
            ], dim=1)  # (1, T_full, D)

            # vision token 在完整序列中的位置（start 和 end 分别标记起始和结束的 token，包含）
            vision_start = text_embeds_part1.shape[1]  # 第一个 vision token 的位置
            vision_end = vision_start + image_token_embeds.shape[1] - 1  # 最后一个 vision token 的位置（包含）
            
            result = {
                "embeddings": full_embeddings,
                "vision_token_positions": (vision_start, vision_end),
                "input_ids": input_ids,
            }
            
            # 如果提供了 answer，计算 answer token 位置
            if answer:
                # 构建不带 answer 但开启 add_generation_prompt 的 message
                message_no_answer = [{
                    "role": "user", 
                    "content": [
                        {"type": "image", "image": image}, 
                        {"type": "text", "text": question}
                    ]
                }]
                
                inputs_no_answer = self.processor(
                    text=[self.processor.apply_chat_template(message_no_answer, tokenize=False, add_generation_prompt=True)],
                    images=[image.convert("RGB")],
                    padding=True,
                    return_tensors="pt"
                )
                
                # 比较两个 input_ids，计算 answer + 结尾 token 的总长度
                total_added_len = input_ids.shape[1] - inputs_no_answer['input_ids'].shape[1]
                
                # 单独 tokenize answer，获取 answer 的实际长度（不含特殊 token）
                answer_tokens = self.processor.tokenizer(
                    answer,
                    add_special_tokens=False,
                    return_tensors='pt'
                )['input_ids']
                answer_len = answer_tokens.shape[1]
                
                # 使用负索引定位 answer 位置
                # answer 从 -total_added_len 开始，长度为 answer_len
                answer_start = -total_added_len
                answer_end = answer_start + answer_len - 1
                
                result["answer_token_positions"] = (answer_start, answer_end)

        return result

    def get_first_token_hidden_state(
        self,
        image: Image.Image,
        question: str
    ) -> Dict[str, Any]:
        """获取模型生成第一个 token 时的隐层状态。

        该方法模仿 generate 方法中的自回归生成过程，但只执行第一步前向传播，
        返回生成第一个 token 时的隐层状态（最后一个输入位置的隐层）。

        参数:
            image: PIL Image 对象
            question: 问题文本

        返回:
            包含以下键的字典:
            - hidden_state: torch.Tensor, shape (1, hidden_dim)，最后一个输入位置的隐层状态
            - last_hidden_states: torch.Tensor, shape (num_layers, 1, seq_len, hidden_dim)，
              所有层的隐层状态（可选，如果模型支持）
            - logits: torch.Tensor, shape (1, seq_len, vocab_size)，输出 logits
            - input_length: int，输入序列长度
        """
        # 根据配置调整图像大小
        image = self._resize_image(image)
        
        message = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }]

        inputs = self.processor(
            text=[self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)],
            images=[image.convert("RGB")],
            padding=True,
            return_tensors="pt"
        )
        
        # 始终将inputs移动到模型第一个参数所在的设备
        target_device = next(self.model.parameters()).device
        inputs = {k: v.to(target_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # 确保创建 attention_mask（如果 processor 没有返回）
        if 'attention_mask' not in inputs or inputs['attention_mask'] is None:
            # 手动创建 attention_mask：所有非 padding 的位置为 1
            pad_token_id = self.processor.tokenizer.pad_token_id
            if pad_token_id is not None:
                attention_mask = (inputs['input_ids'] != pad_token_id).long()
            else:
                # 如果 pad_token_id 为 None，创建全 1 的 mask（对于单样本通常不会有 padding）
                attention_mask = torch.ones_like(inputs['input_ids'], dtype=torch.long)
            inputs['attention_mask'] = attention_mask

        with torch.no_grad():
            # 执行前向传播，获取隐层状态
            outputs = self.model(
                input_ids=inputs['input_ids'],
                pixel_values=inputs['pixel_values'],
                image_grid_thw=inputs['image_grid_thw'],
                attention_mask=inputs['attention_mask'],
                output_hidden_states=True,
                return_dict=True
            )

            # 获取最后一个输入位置的隐层状态（用于生成第一个 token）
            # hidden_states 是一个 tuple，每个元素是 (batch_size, seq_len, hidden_dim)
            # 最后一个元素是最后一层的隐层状态
            last_hidden_state = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            input_length = last_hidden_state.shape[1]
            
            # 最后一个位置的隐层状态（这是生成第一个 token 时需要的上下文）
            first_token_hidden = last_hidden_state[:, -1, :]  # (1, hidden_dim)

            # 获取 logits
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # 可选：收集所有层的隐层状态
            all_hidden_states = torch.stack(outputs.hidden_states, dim=0)  # (num_layers, 1, seq_len, hidden_dim)

        return {
            "hidden_state": first_token_hidden,
            "last_hidden_states": all_hidden_states,
            "logits": logits,
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
            - all_hidden_states: 如果 output_hidden_states=True，返回所有层的隐层状态
        """
        outputs = self.model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # 获取 language model head 的 logits
        logits = self.model.lm_head(outputs.last_hidden_state)
        
        result = {
            'logits': logits
        }
        
        if output_hidden_states:
            result['all_hidden_states'] = outputs.hidden_states
        
        return result

