#!/usr/bin/env python
"""Attention Consistency Pruning - DDP 分布式评估脚本

加载已训练的模型 checkpoint 并进行分布式评估。

启动方式：
    # 使用默认配置和 checkpoint
    ./scripts/run_eval_ddp.sh

    # 指定 GPU
    ./scripts/run_eval_ddp.sh 0,1,2,3

    # 命令行指定 checkpoint（覆盖配置）
    ./scripts/run_eval_ddp.sh 4,5,6,7 --checkpoint outputs/checkpoints/checkpoint_final.pt

配置说明：
    - eval_mode: 在配置文件 evaluation_settings.eval_mode 中设置
        - "origin": 不剪枝，直接 generate
        - "hard": 物理剪枝推理（用于测速度/保留率；可能存在已知 bug）
        - "hard_forward": forward()+greedy decode（不物理删除，更稳，且适配 delayed repair adapter）
    - max_samples: 在配置文件 trainer_settings.dl_settings.eval_max_samples 中设置
"""

import os
import sys

# 环境变量设置
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, List, Optional
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置加载器
from engine.configs.loader import load_config

# 从 main_acp_ddp 导入必要的函数
from main_acp_ddp import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    load_model,
    evaluate,
)


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: torch.device,
    logger=None
) -> Dict[str, Any]:
    """加载 checkpoint 到模型"""
    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")

    # 评估时优先在 CPU 读取 checkpoint，避免瞬时占用大量 GPU 显存。
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if 'pruner_state_dict' in checkpoint:
        model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
        if logger:
            logger.info("  Loaded pruner_manager state")

    # 支持分离式 Adapter（仅当模型启用 Adapter 时加载）
    if hasattr(model, 'use_adapter') and model.use_adapter:
        if 'separated_adapter_state_dict' in checkpoint and model.use_separated_adapters:
            model.separated_adapter_manager.load_state_dict(checkpoint['separated_adapter_state_dict'])
            if logger:
                logger.info("  Loaded separated_adapter_manager state")
        elif 'adapter_state_dict' in checkpoint and not model.use_separated_adapters:
            model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
            if logger:
                logger.info("  Loaded adapter_manager state")
    else:
        if logger and ('adapter_state_dict' in checkpoint or 'separated_adapter_state_dict' in checkpoint):
            logger.info("  Skipped adapter state (use_adapter=False)")

    if 'disc_state_dict' in checkpoint:
        model.disc_manager.load_state_dict(checkpoint['disc_state_dict'])
        if logger:
            logger.info("  Loaded disc_manager state")

    # 新版 delayed repair adapter（语言侧）
    if getattr(model, "use_repair_adapter", False):
        if 'repair_context_encoder_state_dict' in checkpoint and getattr(model, "repair_context_encoder", None) is not None:
            model.repair_context_encoder.load_state_dict(checkpoint['repair_context_encoder_state_dict'])
            if logger:
                logger.info("  Loaded repair_context_encoder state")
        if 'repair_adapter_state_dict' in checkpoint and getattr(model, "repair_adapter_manager", None) is not None:
            model.repair_adapter_manager.load_state_dict(checkpoint['repair_adapter_state_dict'])
            if logger:
                logger.info("  Loaded repair_adapter_manager state")

    if logger:
        if 'step' in checkpoint:
            logger.info(f"  Checkpoint from step {checkpoint['step']}")
        logger.info("Checkpoint loaded successfully.")

    return checkpoint


def infer_model_flags_from_checkpoint(
    checkpoint: Dict[str, Any],
    config,
    logger=None,
) -> None:
    """根据 checkpoint 内容自动推断是否启用 adapter（尤其是 delayed repair adapter）。

    目的：支持直接评估 pruner-only checkpoint（没有 repair adapter state），避免在 config 打开 use_repair_adapter
    时构建随机初始化的 adapter 影响 hard_forward 指标。
    """
    method_cfg = config.method_settings

    # delayed repair adapter：必须至少存在 repair_adapter_state_dict 才认为 checkpoint 支持 repair
    has_repair_adapter = 'repair_adapter_state_dict' in checkpoint
    has_repair_ctx = 'repair_context_encoder_state_dict' in checkpoint
    inferred_use_repair = bool(has_repair_adapter)

    # 旧版 attention-output adapter（如果未来要评估，也可按 state_dict 推断；当前默认不自动开启）
    # inferred_use_adapter = bool(('adapter_state_dict' in checkpoint) or ('separated_adapter_state_dict' in checkpoint))

    prev = bool(method_cfg.get('use_repair_adapter', False))
    method_cfg['use_repair_adapter'] = inferred_use_repair

    if logger:
        logger.info(
            f"Auto-infer repair adapter from checkpoint: use_repair_adapter={inferred_use_repair} "
            f"(was {prev}) [has_adapter={has_repair_adapter}, has_ctx={has_repair_ctx}]"
        )
        if inferred_use_repair and (not has_repair_ctx):
            logger.warning(
                "Checkpoint has repair_adapter_state_dict but no repair_context_encoder_state_dict; "
                "context encoder will stay randomly initialized and may hurt metrics."
            )


def evaluate_no_image_samples(
    model,
    processor,
    no_image_samples: List[Dict[str, Any]],
    judge,
    device: torch.device,
    max_samples: int,
    max_length: int,
    distributed: bool,
) -> Dict[str, Any]:
    """评估无图样本（纯文本问答）

    Args:
        model: 模型
        processor: processor
        no_image_samples: 无图样本列表
        judge: 评判函数
        device: 设备
        max_samples: 最大样本数
        max_length: 序列最大长度
        distributed: 是否分布式

    Returns:
        评估结果字典
    """
    model.eval()
    n_samples = min(len(no_image_samples), max_samples)

    # 分布式评估：每个 rank 处理一部分数据
    if distributed and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        indices = list(range(n_samples))
        local_indices = indices[rank::world_size]
    else:
        local_indices = list(range(n_samples))

    predictions = []
    references = []

    show_progress = is_main_process()

    for i in tqdm(local_indices, desc="Evaluating (no-image)", disable=not show_progress):
        sample = no_image_samples[i]

        # 纯文本模式：不传入图像
        prompt = f"USER: {sample['question']}\nASSISTANT:"
        inputs = processor.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)

        with torch.no_grad():
            output_ids = model.base_model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs.get('attention_mask'),
                max_new_tokens=32,
                do_sample=False,
            )

        generated = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if "ASSISTANT:" in generated:
            pred = generated.split("ASSISTANT:")[-1].strip()
        else:
            pred = generated.strip()

        predictions.append(pred)

        if 'answers' in sample:
            references.append(sample['answers'])
        else:
            references.append(sample['answer'])

    # 分布式评估：收集所有 rank 的结果
    if distributed and dist.is_initialized():
        all_predictions = [None] * dist.get_world_size()
        all_references = [None] * dist.get_world_size()
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_references, references)

        predictions = []
        references = []
        for preds, refs in zip(all_predictions, all_references):
            predictions.extend(preds)
            references.extend(refs)

    result = judge(predictions, references)
    return {
        'accuracy': result['accuracy'],
        'correct': result['correct'],
        'total': result['total'],
    }


def merge_eval_results(
    image_result: Dict[str, Any],
    no_image_result: Dict[str, Any],
) -> Dict[str, Any]:
    """合并有图和无图样本的评估结果

    Args:
        image_result: 有图样本评估结果
        no_image_result: 无图样本评估结果

    Returns:
        合并后的评估结果
    """
    total_correct = image_result['correct'] + no_image_result['correct']
    total_samples = image_result['total'] + no_image_result['total']
    merged = {
        'accuracy': total_correct / total_samples if total_samples > 0 else 0.0,
        'correct': total_correct,
        'total': total_samples,
        'image_accuracy': image_result['accuracy'],
        'image_correct': image_result['correct'],
        'image_total': image_result['total'],
        'no_image_accuracy': no_image_result['accuracy'],
        'no_image_correct': no_image_result['correct'],
        'no_image_total': no_image_result['total'],
    }
    # 保留有图样本的 kept_ratio 等信息
    for key in image_result:
        if key not in merged:
            merged[key] = image_result[key]
    return merged


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Consistency Pruning Evaluation (DDP)")
    parser.add_argument('--config', type=str, default='configs/vision_token_pruning.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (overrides config)')

    args = parser.parse_args()

    # 检测分布式环境
    if 'LOCAL_RANK' in os.environ:
        _, world_size, local_rank, device = setup_distributed()
        distributed = True
    else:
        world_size = 1
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        distributed = False

    try:
        # 加载配置
        config = load_config(override_file=args.config)
        if not is_main_process():
            config.logger = None
        logger = config.logger

        eval_cfg = config.evaluation_settings

        if is_main_process():
            print("=" * 60)
            print("Attention Consistency Pruning - Evaluation")
            if distributed:
                print(f"Distributed mode: world_size={world_size}")
            print("=" * 60)

        # 确定 checkpoint
        checkpoint_path = args.checkpoint or config.global_settings.get('checkpoint')
        if checkpoint_path is None:
            if is_main_process():
                print("Error: No checkpoint specified.")
                print("Use --checkpoint or set global_settings.checkpoint in config.")
            if distributed:
                cleanup_distributed()
            sys.exit(1)

        if not Path(checkpoint_path).exists():
            if is_main_process():
                print(f"Error: Checkpoint not found: {checkpoint_path}")
            if distributed:
                cleanup_distributed()
            sys.exit(1)

        if logger:
            logger.info(f"Checkpoint: {checkpoint_path}")

        # === 自动推断 adapter 开关（在构建模型前）===
        # hard_forward 模式会走 forward()，如果 use_repair_adapter=True 但 ckpt 里没有修复模块权重，
        # 会导致随机初始化 adapter 参与 forward，指标失真；因此这里自动推断是否启用 repair adapter。
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        infer_model_flags_from_checkpoint(checkpoint, config, logger)

        # 获取配置
        method_cfg = config.method_settings
        pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])

        # 加载模型
        if logger:
            logger.info("Loading model...")
        model, processor = load_model(config, device, local_rank)
        # 复用已加载的 checkpoint，避免重复 IO
        # （兼容：load_checkpoint 仍支持从文件加载，这里直接走 state_dict 加载逻辑）
        # 将 checkpoint 临时写回变量名以复用原逻辑结构
        # 直接调用 load_checkpoint 的实现：为了最小改动，传路径再读一次也行；但这里复用更快
        if logger:
            logger.info("Loading checkpoint weights into model...")
        # 手动复用 load_checkpoint 逻辑：保持行为一致
        # 这里不再重复 torch.load
        if 'pruner_state_dict' in checkpoint:
            model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
            if logger:
                logger.info("  Loaded pruner_manager state")
        if hasattr(model, 'use_adapter') and model.use_adapter:
            if 'separated_adapter_state_dict' in checkpoint and model.use_separated_adapters:
                model.separated_adapter_manager.load_state_dict(checkpoint['separated_adapter_state_dict'])
                if logger:
                    logger.info("  Loaded separated_adapter_manager state")
            elif 'adapter_state_dict' in checkpoint and not model.use_separated_adapters:
                model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
                if logger:
                    logger.info("  Loaded adapter_manager state")
        else:
            if logger and ('adapter_state_dict' in checkpoint or 'separated_adapter_state_dict' in checkpoint):
                logger.info("  Skipped adapter state (use_adapter=False)")
        if getattr(model, "use_repair_adapter", False):
            if 'repair_context_encoder_state_dict' in checkpoint and getattr(model, "repair_context_encoder", None) is not None:
                model.repair_context_encoder.load_state_dict(checkpoint['repair_context_encoder_state_dict'])
                if logger:
                    logger.info("  Loaded repair_context_encoder state")
            if 'repair_adapter_state_dict' in checkpoint and getattr(model, "repair_adapter_manager", None) is not None:
                model.repair_adapter_manager.load_state_dict(checkpoint['repair_adapter_state_dict'])
                if logger:
                    logger.info("  Loaded repair_adapter_manager state")
        if 'disc_state_dict' in checkpoint:
            model.disc_manager.load_state_dict(checkpoint['disc_state_dict'])
            if logger:
                logger.info("  Loaded disc_manager state")
        if logger:
            if 'step' in checkpoint:
                logger.info(f"  Checkpoint from step {checkpoint['step']}")
            logger.info("Checkpoint loaded successfully.")

        if logger:
            logger.info(f"Pruning layers: {pruning_layers}")

        # 加载数据集（评估时不过滤超长样本）
        if logger:
            logger.info("Loading dataset...")
        original_logger = config.logger
        if not is_main_process():
            config.logger = None
        # 评估时保留所有样本，不过滤超长样本
        config.dataset_settings['filter_long_samples'] = False
        from engine.datas.loader import load_dataset
        data_bundle = load_dataset(config)
        config.logger = original_logger

        # 默认使用 test 分割
        eval_split = 'test'
        if eval_split not in data_bundle['splits']:
            if logger:
                logger.error(f"Split '{eval_split}' not found.")
            return

        eval_dataset = data_bundle['splits'][eval_split]
        judge = data_bundle['judge']

        # 提取聚合评估相关信息
        meta = data_bundle.get('meta', {})
        aggregate_judge = data_bundle.get('aggregate_judge')
        requires_aggregate_eval = meta.get('requires_aggregate_eval', False)

        # 检查是否有无图样本
        no_image_samples_dict = meta.get('no_image_samples', {})
        no_image_samples = no_image_samples_dict.get(eval_split, [])
        has_no_image_samples = len(no_image_samples) > 0

        if logger:
            logger.info(f"Dataset: {config.dataset_settings.get('name', 'unknown')}")
            logger.info(f"Split: {eval_split}, samples with image: {len(eval_dataset)}")
            if has_no_image_samples:
                logger.info(f"Split: {eval_split}, samples without image: {len(no_image_samples)}")
            if requires_aggregate_eval:
                logger.info("Aggregate evaluation mode enabled (MME/GQA style)")

        # 确定样本数（从配置文件读取）
        max_samples = config.trainer_settings.get('dl_settings', {}).get('eval_max_samples', 500)
        max_length = config.trainer_settings.get('dl_settings', {}).get('max_length', 2048)
        if logger:
            logger.info(f"Max samples: {max_samples}")
            logger.info(f"Max length: {max_length}")

        # 同步
        if distributed:
            dist.barrier()

        # 普通评估
        # 从配置文件读取评估模式
        eval_modes = eval_cfg.get('eval_mode', ['hard'])
        if isinstance(eval_modes, str):
            eval_modes = [eval_modes]

        if is_main_process():
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)

        # 计算无图样本的 max_samples（按比例分配）
        if has_no_image_samples:
            total_samples = len(eval_dataset) + len(no_image_samples)
            no_image_ratio = len(no_image_samples) / total_samples
            no_image_max_samples = int(max_samples * no_image_ratio)
            image_max_samples = max_samples - no_image_max_samples
            if logger:
                logger.info(f"Samples allocation: {image_max_samples} with image, {no_image_max_samples} without image")
        else:
            image_max_samples = max_samples
            no_image_max_samples = 0

        # 如果 mode 包含 origin，先评估并缓存（用于后续 hard 模式的相对准确率）
        origin_result = None
        if 'origin' in eval_modes:
            if logger:
                logger.info("Evaluating 'origin' mode...")

            origin_result = evaluate(
                model=model,
                processor=processor,
                dataset=eval_dataset,
                judge=judge,
                config=config,
                device=device,
                max_samples=image_max_samples,
                mode='origin',
                distributed=distributed,
                aggregate_judge=aggregate_judge,
                requires_aggregate_eval=requires_aggregate_eval,
            )

            # 评估无图样本并合并（仅非聚合评估模式）
            if has_no_image_samples and no_image_max_samples > 0 and not requires_aggregate_eval:
                if logger:
                    logger.info("Evaluating no-image samples...")
                no_image_result = evaluate_no_image_samples(
                    model=model,
                    processor=processor,
                    no_image_samples=no_image_samples,
                    judge=judge,
                    device=device,
                    max_samples=no_image_max_samples,
                    max_length=max_length,
                    distributed=distributed,
                )
                origin_result = merge_eval_results(origin_result, no_image_result)

            if is_main_process():
                print(f"\n[ORIGIN]")
                # 根据数据集类型打印不同指标
                if 'total_score' in origin_result:
                    # MME
                    print(f"  MME Total Score: {origin_result['total_score']:.1f}")
                    print(f"  Simple Accuracy: {origin_result.get('simple_accuracy', 0):.2%}")
                    print(f"  Categories: {origin_result.get('num_categories', 0)}")
                elif 'balanced_accuracy' in origin_result:
                    # GQA
                    print(f"  Balanced Accuracy: {origin_result['balanced_accuracy']:.2%}")
                    print(f"  Simple Accuracy: {origin_result.get('simple_accuracy', 0):.2%}")
                    print(f"  Answer Categories: {origin_result.get('num_answer_categories', 0)}")
                else:
                    # 普通数据集
                    print(f"  Accuracy: {origin_result['accuracy']:.2%} ({origin_result.get('correct', 'N/A')}/{origin_result.get('total', 'N/A')})")
                    if has_no_image_samples and 'image_accuracy' in origin_result:
                        print(f"    - with image: {origin_result['image_accuracy']:.2%} ({origin_result['image_correct']}/{origin_result['image_total']})")
                        print(f"    - no image:   {origin_result['no_image_accuracy']:.2%} ({origin_result['no_image_correct']}/{origin_result['no_image_total']})")

        # 评估其他模式（跳过已评估的 origin）
        for eval_mode in eval_modes:
            if eval_mode == 'origin':
                # 已经评估过了，跳过
                continue

            if logger:
                logger.info(f"\nEvaluating '{eval_mode}' mode...")

            eval_result = evaluate(
                model=model,
                processor=processor,
                dataset=eval_dataset,
                judge=judge,
                config=config,
                device=device,
                max_samples=image_max_samples,
                mode=eval_mode,
                distributed=distributed,
                aggregate_judge=aggregate_judge,
                requires_aggregate_eval=requires_aggregate_eval,
            )

            # 评估无图样本并合并（仅非聚合评估模式）
            if has_no_image_samples and no_image_max_samples > 0 and not requires_aggregate_eval:
                if logger:
                    logger.info("Evaluating no-image samples...")
                no_image_result = evaluate_no_image_samples(
                    model=model,
                    processor=processor,
                    no_image_samples=no_image_samples,
                    judge=judge,
                    device=device,
                    max_samples=no_image_max_samples,
                    max_length=max_length,
                    distributed=distributed,
                )
                eval_result = merge_eval_results(eval_result, no_image_result)

            if is_main_process():
                print(f"\n[{eval_mode.upper()}]")
                # 根据数据集类型打印不同指标
                if 'total_score' in eval_result:
                    # MME
                    print(f"  MME Total Score: {eval_result['total_score']:.1f}")
                    print(f"  Simple Accuracy: {eval_result.get('simple_accuracy', 0):.2%}")
                    print(f"  Categories: {eval_result.get('num_categories', 0)}")
                elif 'balanced_accuracy' in eval_result:
                    # GQA
                    print(f"  Balanced Accuracy: {eval_result['balanced_accuracy']:.2%}")
                    print(f"  Simple Accuracy: {eval_result.get('simple_accuracy', 0):.2%}")
                    print(f"  Answer Categories: {eval_result.get('num_answer_categories', 0)}")
                else:
                    # 普通数据集
                    print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result.get('correct', 'N/A')}/{eval_result.get('total', 'N/A')})")
                    if has_no_image_samples and 'image_accuracy' in eval_result:
                        print(f"    - with image: {eval_result['image_accuracy']:.2%} ({eval_result['image_correct']}/{eval_result['image_total']})")
                        print(f"    - no image:   {eval_result['no_image_accuracy']:.2%} ({eval_result['no_image_correct']}/{eval_result['no_image_total']})")

                # 显示相对准确率（仅当 origin 也被评估时）
                if origin_result is not None and origin_result['accuracy'] > 0:
                    rel_acc = eval_result['accuracy'] / origin_result['accuracy']
                    print(f"  Relative Accuracy: {rel_acc:.2%}")

                if 'avg_kept_ratio' in eval_result:
                    print(f"  Kept ratio: {eval_result['avg_kept_ratio']:.2%}")
                    layer_ratios = []
                    for layer_idx in pruning_layers:
                        kept_key = f'L{layer_idx}_kept'
                        if kept_key in eval_result:
                            layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}")
                    if layer_ratios:
                        print(f"  Per-layer: [{', '.join(layer_ratios)}]")

        if is_main_process():
            print("\n" + "=" * 60)
            print("Evaluation completed.")
            print("=" * 60)

    finally:
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
