"""Attention Consistency Pruning - 训练逻辑

实现基于 attention 聚合一致性的剪枝训练。

核心特点：
1. 不需要 hook 机制 - 直接在继承的模型中实现
2. 不需要完整的 real forward - h_real 在剪枝层内部计算
3. 每个 answer token 独立判别
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional
from collections import defaultdict

from .models.prunable_llava import PrunableLlavaForConditionalGeneration, PrunableLlavaOutput


def compute_task_loss(
    logits: torch.Tensor,
    answer_start: int,
    answer_end: int,
    answer_text: str,
    tokenizer,
    attention_mask: Optional[torch.Tensor] = None,
    batch_idx: int = 0
) -> torch.Tensor:
    """计算单个样本的 task loss

    参数:
        logits: (seq_len, vocab_size) - 模型输出的 logits
        answer_start: answer token 开始位置
        answer_end: answer token 结束位置
        answer_text: ground truth answer 文本
        tokenizer: tokenizer
        attention_mask: (seq_len,) - 注意力掩码

    返回:
        loss: 标量 tensor
    """
    # 将 answer 转为首字母大写（解决 TRAINING_NOTES 中提到的大小写问题）
    answer_text = answer_text.capitalize()

    # Tokenize answer
    answer_tokens = tokenizer(answer_text, add_special_tokens=False)['input_ids']

    # 获取预测 logits（answer 位置之前的 token 预测 answer 的第一个 token）
    # 标准 causal LM loss：position i 的 logits 预测 position i+1 的 token
    pred_start = answer_start - 1
    pred_end = min(pred_start + len(answer_tokens), logits.shape[0] - 1)

    # 确保范围有效
    if pred_start < 0:
        pred_start = 0
    if pred_end <= pred_start:
        return torch.tensor(0.0, device=logits.device)

    # 获取对应的 logits 和 targets
    pred_logits = logits[pred_start:pred_end]  # (n_tokens, vocab_size)
    target_len = min(len(answer_tokens), pred_end - pred_start)
    targets = torch.tensor(answer_tokens[:target_len], device=logits.device)

    # Cross entropy loss
    loss = F.cross_entropy(pred_logits, targets)

    return loss


def train_step_acp(
    batch: List[Any],
    model: PrunableLlavaForConditionalGeneration,
    config: Dict[str, Any],
    current_step: int,
    total_steps: int,
    device: torch.device
) -> Dict[str, Dict[str, Any]]:
    """Attention Consistency Pruning 训练步骤

    参数:
        batch: 数据 batch，每个元素包含 image, question, answer
        model: PrunableLlavaForConditionalGeneration 模型
        config: 配置字典
        current_step: 当前步数
        total_steps: 总步数
        device: 设备

    返回:
        损失字典
    """
    method_config = config.get("method_settings", {})

    # === Temperature Annealing ===
    temperature = method_config.get('temperature', 1.0)
    temperature_min = method_config.get('temperature_min', 0.5)
    anneal_rate = method_config.get('temperature_anneal_rate', 0.4)

    progress = current_step / total_steps if total_steps > 0 else 0
    if progress < anneal_rate:
        current_temp = temperature - (progress / anneal_rate) * (temperature - temperature_min)
    else:
        current_temp = temperature_min

    model.set_temperature(current_temp)

    # === 初始化损失累加器 ===
    pruner_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    disc_losses = defaultdict(lambda: torch.tensor(0.0, device=device))
    stats = defaultdict(float)

    stats["temperature"] = current_temp

    batch_size = len(batch)

    # === 准备输入 ===
    # 使用 backbone 的 processor 进行预处理
    images = [sample["image"] for sample in batch]
    questions = [sample["question"] for sample in batch]
    answers = [sample["answer"] for sample in batch]

    # 获取 processor
    processor = model.base_model.processor if hasattr(model.base_model, 'processor') else None

    if processor is None:
        raise ValueError("Model does not have processor attached")

    # 预处理（这部分需要根据实际的 backbone 实现调整）
    # 假设 backbone 有 preprocess_batch 方法
    backbone = model.base_model

    with torch.no_grad():
        # 使用 processor 进行预处理
        prompts = []
        for q in questions:
            prompt = f"USER: <image>\n{q}\nASSISTANT:"
            prompts.append(prompt)

        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)

        # 获取 position 信息
        # 需要找到 vision tokens, question tokens, answer tokens 的位置
        # 对于 LLaVA，vision tokens 替换了 <image> placeholder

        # 找到 image token 位置
        image_token_id = backbone.config.image_token_id
        input_ids = inputs['input_ids']

        # 假设所有样本的 vision token 位置相同（因为图像大小统一）
        # 找第一个样本的位置
        image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
        if len(image_positions) > 0:
            # vision tokens 从第一个 image token 开始
            vision_start = image_positions[0].item()
            # LLaVA 1.5 使用 576 个 vision tokens
            n_vision_tokens = method_config.get('n_vision_tokens', 576)
            vision_end = vision_start + n_vision_tokens
        else:
            raise ValueError("No image token found in input")

        # Question 位置：从 vision_end 到 "ASSISTANT:" 之前
        # Answer 位置：从 "ASSISTANT:" 之后开始
        seq_len = input_ids.shape[1]

        # 简化：假设 ASSISTANT: 后面就是 answer 开始
        # 实际位置需要更精确的计算
        assistant_token_ids = processor.tokenizer.encode("ASSISTANT:", add_special_tokens=False)

        # 对每个样本找到 answer 开始位置
        answer_starts = []
        for i in range(batch_size):
            # 找到 ASSISTANT: 的位置
            ids = input_ids[i].tolist()
            found = False
            for j in range(len(ids) - len(assistant_token_ids) + 1):
                if ids[j:j+len(assistant_token_ids)] == assistant_token_ids:
                    answer_starts.append(j + len(assistant_token_ids))
                    found = True
                    break
            if not found:
                # 如果找不到，使用 vision_end + 一些 offset 作为默认值
                answer_starts.append(vision_end + 10)

        # Question 范围
        question_start = vision_end
        question_end = min(answer_starts)  # 使用最小的 answer_start 作为 question_end

        # Answer 范围（取最大范围）
        answer_start = min(answer_starts)
        answer_end = seq_len

    # === Forward Pass ===
    model.train()

    output = model(
        input_ids=inputs['input_ids'],
        pixel_values=inputs['pixel_values'],
        attention_mask=inputs['attention_mask'],
        vision_start=vision_start,
        vision_end=vision_end,
        question_start=question_start,
        question_end=question_end,
        answer_start=answer_start,
        answer_end=answer_end,
        return_pruning_info=True
    )

    # === 计算 Losses ===

    # 1. Task Loss
    task_loss = torch.tensor(0.0, device=device)
    for i in range(batch_size):
        sample_loss = compute_task_loss(
            logits=output.logits[i],
            answer_start=answer_starts[i],
            answer_end=seq_len,
            answer_text=answers[i],
            tokenizer=processor.tokenizer,
            batch_idx=i
        )
        task_loss = task_loss + sample_loss
    task_loss = task_loss / batch_size
    pruner_losses['task_loss'] = task_loss

    # 2. 如果有剪枝信息，计算 GAN 相关 losses
    if output.pruning_infos and len(output.pruning_infos) > 0:
        # 收集 h_real 和 h_fake
        h_real_dict = {idx: info['h_real'] for idx, info in output.pruning_infos.items()}
        h_fake_dict = {idx: info['h_fake'] for idx, info in output.pruning_infos.items()}

        # Adversarial Loss
        adv_loss = model.disc_manager.compute_adv_loss(h_fake_dict)
        pruner_losses['adv_loss'] = adv_loss

        # Discriminator Loss
        disc_loss = model.disc_manager.compute_disc_loss(h_real_dict, h_fake_dict)
        disc_losses['disc_loss'] = disc_loss

        # Discriminator Accuracy（用于监控）
        acc_info = model.disc_manager.compute_accuracy(h_real_dict, h_fake_dict)
        stats['disc_accuracy'] = acc_info['overall']
        stats['disc_real_acc'] = acc_info['real_acc']
        stats['disc_fake_acc'] = acc_info['fake_acc']

        # Sparsity Loss
        target_token_num = method_config.get('target_token_num', 144)
        n_vision = vision_end - vision_start
        target_ratio = target_token_num / n_vision

        sparsity_loss = torch.tensor(0.0, device=device)
        total_kept_ratio = 0

        for layer_idx, info in output.pruning_infos.items():
            hard_mask = info['hard_mask']
            kept_ratio = hard_mask.mean()
            total_kept_ratio += kept_ratio.item()
            sparsity_loss = sparsity_loss + torch.abs(kept_ratio - target_ratio)

            stats[f'L{layer_idx}_kept'] = kept_ratio.item()

        sparsity_loss = sparsity_loss / len(output.pruning_infos)
        pruner_losses['sparsity_loss'] = sparsity_loss

        avg_kept_ratio = total_kept_ratio / len(output.pruning_infos)
        stats['avg_kept_ratio'] = avg_kept_ratio
        stats['target_kept_ratio'] = target_ratio

    # === 应用权重 ===

    # 获取权重
    task_weight = method_config.get('task_loss_weight', 1.0)
    adv_weight = method_config.get('adv_loss_weight', 0.5)
    sparsity_weight = method_config.get('sparsity_weight', 0.2)
    disc_weight = method_config.get('disc_loss_weight', 1.0)

    # 动态权重调度
    warmup_ratio = method_config.get('loss_weight_warmup_ratio', 0.0)
    if warmup_ratio > 0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2

        task_weight_start = method_config.get('task_loss_weight_start', task_weight)
        adv_weight_start = method_config.get('adv_loss_weight_start', adv_weight)

        task_weight = task_weight_start + (task_weight - task_weight_start) * cosine_factor.item()
        adv_weight = adv_weight_start + (adv_weight - adv_weight_start) * cosine_factor.item()

    # Sparsity weight warmup
    sparsity_warmup = method_config.get('sparsity_warmup_enable', False)
    if sparsity_warmup:
        sparsity_warmup_ratio = method_config.get('sparsity_warmup_ratio', 0.2)
        sparsity_weight_max = method_config.get('sparsity_weight_max', sparsity_weight)
        if progress < sparsity_warmup_ratio:
            sparsity_weight = sparsity_weight + (sparsity_weight_max - sparsity_weight) * (progress / sparsity_warmup_ratio)
        else:
            sparsity_weight = sparsity_weight_max

    # 保存原始 loss（未加权）
    stats['raw_task_loss'] = pruner_losses['task_loss'].item()
    if 'adv_loss' in pruner_losses:
        stats['raw_adv_loss'] = pruner_losses['adv_loss'].item()
    if 'sparsity_loss' in pruner_losses:
        stats['raw_sparsity_loss'] = pruner_losses['sparsity_loss'].item()

    # 应用权重
    pruner_losses['task_loss'] = pruner_losses['task_loss'] * task_weight
    if 'adv_loss' in pruner_losses:
        pruner_losses['adv_loss'] = pruner_losses['adv_loss'] * adv_weight
    if 'sparsity_loss' in pruner_losses:
        pruner_losses['sparsity_loss'] = pruner_losses['sparsity_loss'] * sparsity_weight

    disc_losses['disc_loss'] = disc_losses.get('disc_loss', torch.tensor(0.0, device=device)) * disc_weight

    stats['current_task_weight'] = task_weight
    stats['current_adv_weight'] = adv_weight
    stats['current_sparsity_weight'] = sparsity_weight

    return {
        'layer_pruners': dict(pruner_losses),
        'discriminator': dict(disc_losses),
        'metrics': dict(stats)
    }


class ACPTrainer:
    """Attention Consistency Pruning Trainer

    封装训练逻辑，提供更方便的接口。
    """

    def __init__(
        self,
        model: PrunableLlavaForConditionalGeneration,
        config: Dict[str, Any],
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device

        self.current_step = 0
        self.total_steps = 0

        # 创建优化器
        self._create_optimizers()

    def _create_optimizers(self):
        """创建优化器"""
        opt_config = self.config.get('trainer_settings', {}).get('dl_settings', {}).get('optimizers', {})

        pruner_lr = opt_config.get('layer_pruners', {}).get('lr', 1e-4)
        disc_lr = opt_config.get('discriminator', {}).get('lr', 1.5e-4)

        self.pruner_optimizer = torch.optim.Adam(
            self.model.get_pruner_parameters(),
            lr=pruner_lr
        )

        self.disc_optimizer = torch.optim.Adam(
            self.model.get_discriminator_parameters(),
            lr=disc_lr
        )

    def train_step(self, batch: List[Any]) -> Dict[str, Any]:
        """执行一个训练步骤"""
        # 前向传播和 loss 计算
        losses = train_step_acp(
            batch=batch,
            model=self.model,
            config=self.config,
            current_step=self.current_step,
            total_steps=self.total_steps,
            device=self.device
        )

        # === 更新 Discriminator ===
        self.disc_optimizer.zero_grad()
        disc_total = sum(losses['discriminator'].values())
        if disc_total.requires_grad:
            disc_total.backward(retain_graph=True)
            self.disc_optimizer.step()

        # === 更新 Pruners ===
        self.pruner_optimizer.zero_grad()
        pruner_total = sum(losses['layer_pruners'].values())
        if pruner_total.requires_grad:
            pruner_total.backward()
            self.pruner_optimizer.step()

        self.current_step += 1

        return losses

    def set_total_steps(self, total_steps: int):
        """设置总步数"""
        self.total_steps = total_steps
