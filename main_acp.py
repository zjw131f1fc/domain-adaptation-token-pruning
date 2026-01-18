#!/usr/bin/env python
"""Attention Consistency Pruning - 完整训练脚本

基于新的 Attention Consistency Pruning 架构的完整训练流程。

特点：
1. 直接继承 LlavaForConditionalGeneration，不使用 hook
2. 在剪枝层计算 h_real 和 h_fake
3. 每个 answer token 独立判别
4. 使用你的数据加载器和 judge 功能
"""

import os
import sys
import yaml
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# ============================================================
# 配置加载
# ============================================================

@dataclass
class Config:
    """配置容器"""
    global_settings: Dict[str, Any]
    trainer_settings: Dict[str, Any]
    dataset_settings: Dict[str, Any]
    backbone_settings: Dict[str, Any]
    method_settings: Dict[str, Any]
    evaluation_settings: Dict[str, Any]
    logger: Any = None

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(
            global_settings=data.get('global_settings', {}),
            trainer_settings=data.get('trainer_settings', {}),
            dataset_settings=data.get('dataset_settings', {}),
            backbone_settings=data.get('backbone_settings', {}),
            method_settings=data.get('method_settings', {}),
            evaluation_settings=data.get('evaluation_settings', {}),
        )


# ============================================================
# 模型加载与预处理
# ============================================================

def load_model(config: Config, device: torch.device):
    """加载可剪枝的 LLaVA 模型"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    method_cfg = config.method_settings
    backbone_cfg = config.backbone_settings

    # 获取剪枝层配置
    pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])
    pruner_d_internal = method_cfg.get('pruner_d_internal', 128)
    disc_d_hidden = method_cfg.get('disc_d_d', 256)
    temperature = method_cfg.get('temperature', 1.0)
    dropout = method_cfg.get('pruner_dropout', 0.1)
    disc_spectral_norm = method_cfg.get('disc_use_spectral_norm', False)

    # 模型路径
    model_name = backbone_cfg.get('name', 'llava-1.5-7b')
    model_mapping = {
        'llava-1.5-7b': 'llava-hf/llava-1.5-7b-hf',
        'llava-1.5-13b': 'llava-hf/llava-1.5-13b-hf',
    }
    model_path = model_mapping.get(model_name, model_name)

    # 缓存目录
    cache_dir = config.global_settings.get('hf_cache_dir', None)

    print(f"Loading base model from {model_path}...")

    # 加载基础模型和处理器
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        cache_dir=cache_dir,
    )
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)

    # 设置 padding side 为 right（参考 TRAINING_NOTES）
    processor.tokenizer.padding_side = "right"

    # 将 processor 附加到模型
    base_model.processor = processor

    # 创建可剪枝模型
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration

    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=pruning_layers,
        pruner_d_internal=pruner_d_internal,
        disc_d_hidden=disc_d_hidden,
        temperature=temperature,
        dropout=dropout,
        disc_use_spectral_norm=disc_spectral_norm,
    )

    # 冻结基础模型
    model.freeze_base_model()

    print(f"Model loaded. Pruning layers: {pruning_layers}")
    print(f"Trainable parameters:")
    print(f"  - Pruners: {sum(p.numel() for p in model.get_pruner_parameters()):,}")
    print(f"  - Discriminators: {sum(p.numel() for p in model.get_discriminator_parameters()):,}")

    return model, processor


def preprocess_batch(
    batch: List[Dict[str, Any]],
    processor,
    device: torch.device,
    max_length: int = 512
) -> Dict[str, Any]:
    """预处理一个 batch 的数据

    参数:
        batch: 数据列表，每个元素包含 image, question, answer
        processor: LLaVA processor
        device: 设备
        max_length: 最大序列长度

    返回:
        预处理后的输入字典
    """
    images = [sample['image'] for sample in batch]
    questions = [sample['question'] for sample in batch]
    answers = [sample['answer'] for sample in batch]

    # 构建 prompt
    prompts = []
    for q in questions:
        # LLaVA 格式的 prompt
        prompt = f"USER: <image>\n{q}\nASSISTANT:"
        prompts.append(prompt)

    # 使用 processor 处理
    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    # 找到各个区域的位置
    input_ids = inputs['input_ids']
    batch_size, seq_len = input_ids.shape

    # 找 image token 位置
    # LLaVA 使用特殊的 image token id
    # 注意：实际的 vision tokens 会替换掉 <image> placeholder
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')

    # 对于 LLaVA 1.5，vision tokens 数量固定为 576 (24x24 patches)
    n_vision_tokens = 576

    # 找第一个样本的位置（假设 batch 内位置一致）
    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]

    if len(image_positions) > 0:
        vision_start = image_positions[0].item()
        vision_end = vision_start + n_vision_tokens
    else:
        # 如果找不到 image token，可能是因为已经被替换
        # 尝试通过序列结构推断
        vision_start = 1  # 跳过 BOS
        vision_end = vision_start + n_vision_tokens

    # Question 位置：从 vision_end 到 ASSISTANT: 之前
    # Answer 位置：从 ASSISTANT: 之后开始
    assistant_token = "ASSISTANT:"
    assistant_ids = processor.tokenizer.encode(assistant_token, add_special_tokens=False)

    answer_starts = []
    for i in range(batch_size):
        ids = input_ids[i].tolist()
        found = False
        for j in range(len(ids) - len(assistant_ids) + 1):
            if ids[j:j+len(assistant_ids)] == assistant_ids:
                answer_starts.append(j + len(assistant_ids))
                found = True
                break
        if not found:
            answer_starts.append(vision_end + 20)  # 默认值

    question_start = vision_end
    question_end = min(answer_starts)
    answer_start = min(answer_starts)
    answer_end = seq_len

    return {
        'inputs': inputs,
        'images': images,
        'questions': questions,
        'answers': answers,
        'vision_start': vision_start,
        'vision_end': vision_end,
        'question_start': question_start,
        'question_end': question_end,
        'answer_start': answer_start,
        'answer_end': answer_end,
        'answer_starts': answer_starts,  # 每个样本的 answer 开始位置
        'n_vision': n_vision_tokens,
    }


# ============================================================
# 损失计算
# ============================================================

def compute_task_loss(
    logits: torch.Tensor,
    answer_starts: List[int],
    answers: List[str],
    tokenizer,
    device: torch.device
) -> torch.Tensor:
    """计算 task loss (cross entropy)

    参数:
        logits: (batch, seq_len, vocab_size)
        answer_starts: 每个样本的 answer 开始位置
        answers: ground truth 答案列表
        tokenizer: tokenizer

    返回:
        task_loss: 标量
    """
    batch_size = logits.shape[0]
    total_loss = torch.tensor(0.0, device=device)

    for i in range(batch_size):
        # 将 answer 转为首字母大写（参考 TRAINING_NOTES）
        answer = answers[i].capitalize()

        # Tokenize answer
        answer_ids = tokenizer(answer, add_special_tokens=False)['input_ids']
        if len(answer_ids) == 0:
            continue

        # 预测位置：answer_start - 1 开始（causal LM）
        pred_start = answer_starts[i] - 1
        pred_end = min(pred_start + len(answer_ids), logits.shape[1] - 1)

        if pred_start < 0 or pred_end <= pred_start:
            continue

        # 获取 logits 和 targets
        pred_logits = logits[i, pred_start:pred_end]
        target_len = min(len(answer_ids), pred_end - pred_start)
        targets = torch.tensor(answer_ids[:target_len], device=device)

        # Cross entropy
        loss = F.cross_entropy(pred_logits, targets)
        total_loss = total_loss + loss

    return total_loss / batch_size if batch_size > 0 else total_loss


# ============================================================
# 训练步骤
# ============================================================

def train_step(
    batch: List[Dict[str, Any]],
    model,
    processor,
    config: Config,
    current_step: int,
    total_steps: int,
    device: torch.device
) -> Dict[str, Any]:
    """执行一个训练步骤

    返回:
        losses_dict: 包含各项损失和统计信息
    """
    method_cfg = config.method_settings

    # === Temperature Annealing ===
    temperature = method_cfg.get('temperature', 1.0)
    temperature_min = method_cfg.get('temperature_min', 0.5)
    anneal_rate = method_cfg.get('temperature_anneal_rate', 0.4)

    progress = current_step / total_steps if total_steps > 0 else 0
    if progress < anneal_rate:
        current_temp = temperature - (progress / anneal_rate) * (temperature - temperature_min)
    else:
        current_temp = temperature_min

    model.set_temperature(current_temp)

    # === 预处理 ===
    prep = preprocess_batch(batch, processor, device)
    inputs = prep['inputs']

    # === Forward ===
    model.train()

    output = model(
        input_ids=inputs['input_ids'],
        pixel_values=inputs['pixel_values'],
        attention_mask=inputs['attention_mask'],
        vision_start=prep['vision_start'],
        vision_end=prep['vision_end'],
        question_start=prep['question_start'],
        question_end=prep['question_end'],
        answer_start=prep['answer_start'],
        answer_end=prep['answer_end'],
        return_pruning_info=True,
    )

    # === 计算 Losses ===
    losses = {}
    stats = {'temperature': current_temp}

    # 1. Task Loss
    task_loss = compute_task_loss(
        output.logits,
        prep['answer_starts'],
        prep['answers'],
        processor.tokenizer,
        device
    )
    losses['task_loss'] = task_loss
    stats['raw_task_loss'] = task_loss.item()

    # 2. 如果有剪枝信息，计算 GAN 相关 losses
    if output.pruning_infos and len(output.pruning_infos) > 0:
        h_real_dict = {idx: info['h_real'] for idx, info in output.pruning_infos.items()}
        h_fake_dict = {idx: info['h_fake'] for idx, info in output.pruning_infos.items()}

        # Adversarial Loss
        adv_loss = model.disc_manager.compute_adv_loss(h_fake_dict)
        losses['adv_loss'] = adv_loss
        stats['raw_adv_loss'] = adv_loss.item()

        # Discriminator Loss
        disc_loss = model.disc_manager.compute_disc_loss(h_real_dict, h_fake_dict)
        losses['disc_loss'] = disc_loss
        stats['raw_disc_loss'] = disc_loss.item()

        # Discriminator Accuracy
        acc_info = model.disc_manager.compute_accuracy(h_real_dict, h_fake_dict)
        stats['disc_accuracy'] = acc_info['overall']
        stats['disc_real_acc'] = acc_info['real_acc']
        stats['disc_fake_acc'] = acc_info['fake_acc']

        # Sparsity Loss
        target_token_num = method_cfg.get('target_token_num', 144)
        n_vision = prep['n_vision']
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
        losses['sparsity_loss'] = sparsity_loss
        stats['raw_sparsity_loss'] = sparsity_loss.item()
        stats['avg_kept_ratio'] = total_kept_ratio / len(output.pruning_infos)
        stats['target_kept_ratio'] = target_ratio

    # === 应用权重 ===
    task_weight = method_cfg.get('task_loss_weight', 1.0)
    adv_weight = method_cfg.get('adv_loss_weight', 0.5)
    sparsity_weight = method_cfg.get('sparsity_weight', 0.2)

    # 动态权重 warmup
    warmup_ratio = method_cfg.get('loss_weight_warmup_ratio', 0.0)
    if warmup_ratio > 0 and progress < warmup_ratio:
        warmup_progress = progress / warmup_ratio
        cosine_factor = (1 - torch.cos(torch.tensor(warmup_progress * 3.14159))) / 2

        task_weight_start = method_cfg.get('task_loss_weight_start', task_weight)
        adv_weight_start = method_cfg.get('adv_loss_weight_start', adv_weight)

        task_weight = task_weight_start + (task_weight - task_weight_start) * cosine_factor.item()
        adv_weight = adv_weight_start + (adv_weight - adv_weight_start) * cosine_factor.item()

    # Sparsity warmup
    if method_cfg.get('sparsity_warmup_enable', False):
        sparsity_warmup_ratio = method_cfg.get('sparsity_warmup_ratio', 0.2)
        sparsity_weight_max = method_cfg.get('sparsity_weight_max', sparsity_weight)
        if progress < sparsity_warmup_ratio:
            sparsity_weight = sparsity_weight + (sparsity_weight_max - sparsity_weight) * (progress / sparsity_warmup_ratio)
        else:
            sparsity_weight = sparsity_weight_max

    stats['task_weight'] = task_weight
    stats['adv_weight'] = adv_weight
    stats['sparsity_weight'] = sparsity_weight

    # 加权损失
    weighted_losses = {
        'task_loss': losses['task_loss'] * task_weight,
    }
    if 'adv_loss' in losses:
        weighted_losses['adv_loss'] = losses['adv_loss'] * adv_weight
    if 'sparsity_loss' in losses:
        weighted_losses['sparsity_loss'] = losses['sparsity_loss'] * sparsity_weight
    if 'disc_loss' in losses:
        weighted_losses['disc_loss'] = losses['disc_loss']

    return {
        'losses': weighted_losses,
        'stats': stats,
    }


# ============================================================
# 评估
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    processor,
    dataset,
    judge,
    config: Config,
    device: torch.device,
    max_samples: int = 500,
    batch_size: int = 1,
) -> Dict[str, float]:
    """评估模型

    参数:
        model: 模型
        processor: processor
        dataset: 数据集
        judge: 评估函数
        config: 配置
        device: 设备
        max_samples: 最大评估样本数
        batch_size: batch 大小（生成时通常为 1）

    返回:
        评估结果字典
    """
    model.eval()

    n_samples = min(len(dataset), max_samples)
    predictions = []
    references = []

    for i in tqdm(range(n_samples), desc="Evaluating"):
        sample = dataset[i]

        # 构建输入
        prompt = f"USER: <image>\n{sample['question']}\nASSISTANT:"
        inputs = processor(
            text=prompt,
            images=sample['image'],
            return_tensors="pt",
        ).to(device)

        # 生成
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
            )

        # 解码
        generated = processor.decode(output_ids[0], skip_special_tokens=True)

        # 提取答案（ASSISTANT: 之后的部分）
        if "ASSISTANT:" in generated:
            pred = generated.split("ASSISTANT:")[-1].strip()
        else:
            pred = generated.strip()

        predictions.append(pred)

        # 参考答案
        if 'answers' in sample:
            references.append(sample['answers'])
        else:
            references.append(sample['answer'])

    # 使用 judge 评估
    result = judge(predictions, references)

    return {
        'accuracy': result['accuracy'],
        'correct': result['correct'],
        'total': result['total'],
    }


# ============================================================
# 主训练循环
# ============================================================

def train(config: Config):
    """主训练函数"""
    # 设置
    device = torch.device(config.global_settings.get('device', 'cuda'))
    seed = config.global_settings.get('seed', 42)
    torch.manual_seed(seed)

    # 加载模型
    model, processor = load_model(config, device)

    # 加载数据
    print("Loading dataset...")
    from engine.datas.impl.vqa_v2 import VQAV2Preparer
    data_preparer = VQAV2Preparer(config)
    data_bundle = data_preparer.get()

    train_dataset = data_bundle['splits']['train']
    test_dataset = data_bundle['splits'].get('test', None)
    judge = data_bundle['judge']

    print(f"Train samples: {len(train_dataset)}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset)}")

    # 创建优化器
    trainer_cfg = config.trainer_settings.get('dl_settings', {})
    opt_cfg = trainer_cfg.get('optimizers', {})

    pruner_lr = opt_cfg.get('layer_pruners', {}).get('lr', 1e-4)
    disc_lr = opt_cfg.get('discriminator', {}).get('lr', 1.5e-4)

    pruner_optimizer = torch.optim.Adam(model.get_pruner_parameters(), lr=pruner_lr)
    disc_optimizer = torch.optim.Adam(model.get_discriminator_parameters(), lr=disc_lr)

    # 训练参数
    epochs = trainer_cfg.get('epochs', 1)
    batch_size = trainer_cfg.get('batch_size', 4)
    print_every = trainer_cfg.get('print_loss_every_batches', 50)
    eval_every = trainer_cfg.get('eval_every_batches', 1000)
    eval_max_samples = trainer_cfg.get('eval_max_samples', 500)
    save_every = trainer_cfg.get('save_every_batches', 3000)
    grad_clip = trainer_cfg.get('grad_clip_max_norm', None)

    # 计算总步数
    total_batches = (len(train_dataset) + batch_size - 1) // batch_size
    total_steps = epochs * total_batches

    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total batches: {total_batches}")
    print(f"  Total steps: {total_steps}")
    print(f"  Pruner LR: {pruner_lr}")
    print(f"  Discriminator LR: {disc_lr}")

    # 保存目录
    save_dir = Path(config.global_settings.get('save_dir', './outputs/checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    global_step = 0

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*60}")

        # 打乱数据
        indices = torch.randperm(len(train_dataset)).tolist()

        epoch_losses = defaultdict(float)
        epoch_stats = defaultdict(float)
        n_batches = 0

        for batch_start in tqdm(range(0, len(train_dataset), batch_size), desc=f"Epoch {epoch+1}"):
            batch_indices = indices[batch_start:batch_start + batch_size]
            batch = [train_dataset[i] for i in batch_indices]

            # 训练步骤
            result = train_step(
                batch=batch,
                model=model,
                processor=processor,
                config=config,
                current_step=global_step,
                total_steps=total_steps,
                device=device,
            )

            losses = result['losses']
            stats = result['stats']

            # === 更新 Discriminator ===
            disc_optimizer.zero_grad()
            if 'disc_loss' in losses and losses['disc_loss'].requires_grad:
                losses['disc_loss'].backward(retain_graph=True)
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.get_discriminator_parameters(), grad_clip)
                disc_optimizer.step()

            # === 更新 Pruners ===
            pruner_optimizer.zero_grad()
            pruner_total = sum(v for k, v in losses.items() if k != 'disc_loss')
            if pruner_total.requires_grad:
                pruner_total.backward()
                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.get_pruner_parameters(), grad_clip)
                pruner_optimizer.step()

            # 累计统计
            for k, v in losses.items():
                epoch_losses[k] += v.item()
            for k, v in stats.items():
                epoch_stats[k] += v
            n_batches += 1
            global_step += 1

            # 打印
            if global_step % print_every == 0:
                avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
                avg_stats = {k: v / n_batches for k, v in epoch_stats.items()}

                print(f"\nStep {global_step}:")
                print(f"  Losses: " + ", ".join(f"{k}={v:.4f}" for k, v in avg_losses.items()))
                if 'avg_kept_ratio' in avg_stats:
                    print(f"  Kept ratio: {avg_stats['avg_kept_ratio']:.2%} (target: {avg_stats['target_kept_ratio']:.2%})")
                if 'disc_accuracy' in avg_stats:
                    print(f"  Disc acc: {avg_stats['disc_accuracy']:.2%}")

            # 评估
            if test_dataset and global_step % eval_every == 0:
                print(f"\nEvaluating at step {global_step}...")
                eval_result = evaluate(
                    model, processor, test_dataset, judge, config, device,
                    max_samples=eval_max_samples
                )
                print(f"  Accuracy: {eval_result['accuracy']:.2%}")
                model.train()

            # 保存
            if global_step % save_every == 0:
                ckpt_path = save_dir / f"checkpoint_step{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'pruner_state_dict': model.pruner_manager.state_dict(),
                    'disc_state_dict': model.disc_manager.state_dict(),
                    'pruner_optimizer': pruner_optimizer.state_dict(),
                    'disc_optimizer': disc_optimizer.state_dict(),
                }, ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")

        # Epoch 结束
        print(f"\nEpoch {epoch + 1} completed.")

    # 最终保存
    final_path = save_dir / "checkpoint_final.pt"
    torch.save({
        'step': global_step,
        'pruner_state_dict': model.pruner_manager.state_dict(),
        'disc_state_dict': model.disc_manager.state_dict(),
    }, final_path)
    print(f"\nTraining completed. Final checkpoint saved to {final_path}")

    # 最终评估
    if test_dataset:
        print("\nFinal evaluation...")
        eval_result = evaluate(
            model, processor, test_dataset, judge, config, device,
            max_samples=eval_max_samples
        )
        print(f"Final accuracy: {eval_result['accuracy']:.2%}")


# ============================================================
# 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Consistency Pruning Training")
    parser.add_argument('--config', type=str, default='configs/vision_token_pruning.yaml',
                        help='Path to config file')
    args = parser.parse_args()

    print("=" * 60)
    print("Attention Consistency Pruning - Training")
    print("=" * 60)

    # 加载配置
    config = Config.from_yaml(args.config)

    # 设置环境
    if config.global_settings.get('pytorch_cuda_alloc_conf'):
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = config.global_settings['pytorch_cuda_alloc_conf']
    if config.global_settings.get('hf_cache_dir'):
        os.environ['HF_HOME'] = config.global_settings['hf_cache_dir']
        os.environ['TRANSFORMERS_CACHE'] = config.global_settings['hf_cache_dir']

    # 训练
    train(config)


if __name__ == "__main__":
    main()
