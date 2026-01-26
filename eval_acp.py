#!/usr/bin/env python
"""Attention Consistency Pruning - 评估脚本

加载已训练的模型 checkpoint 并进行评估。

用法:
    python eval_acp.py --config configs/vision_token_pruning.yaml --checkpoint outputs/checkpoints/checkpoint_final.pt
    python eval_acp.py --checkpoint outputs/checkpoints/checkpoint_step3000.pt --mode hard
    python eval_acp.py --checkpoint outputs/checkpoints/checkpoint_final.pt --mode origin hard --max_samples 1000
"""

import os
import sys

# ============================================================
# 环境变量设置（在任何其他 import 之前）
# ============================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# 添加项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入配置加载器
from engine.configs.loader import load_config

# 从 main_acp 导入必要的函数
from main_acp import load_model, preprocess_batch, evaluate


def load_checkpoint(model, checkpoint_path: str, device: torch.device, logger):
    """加载 checkpoint 到模型

    参数:
        model: PrunableLlavaForConditionalGeneration 模型
        checkpoint_path: checkpoint 文件路径
        device: 设备
        logger: 日志器
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 加载 pruner 状态
    if 'pruner_state_dict' in checkpoint:
        model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
        logger.info("  Loaded pruner_manager state")
    else:
        logger.warning("  No pruner_state_dict found in checkpoint")

    # 加载 discriminator 状态（评估时可能不需要，但加载以保持一致）
    if 'disc_state_dict' in checkpoint:
        model.disc_manager.load_state_dict(checkpoint['disc_state_dict'])
        logger.info("  Loaded disc_manager state")
    else:
        logger.warning("  No disc_state_dict found in checkpoint")

    # 打印 checkpoint 信息
    if 'step' in checkpoint:
        logger.info(f"  Checkpoint from step {checkpoint['step']}")

    logger.info("Checkpoint loaded successfully.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Consistency Pruning Evaluation")
    parser.add_argument('--config', type=str, default='configs/vision_token_pruning.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--mode', type=str, nargs='+', default=None,
                        choices=['origin', 'hard'],
                        help='Evaluation mode(s): origin (no pruning), hard (with pruning). If not specified, uses config evaluation_settings.eval_mode')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (default: use config value)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'val'],
                        help='Dataset split to evaluate on')

    args = parser.parse_args()

    print("=" * 60)
    print("Attention Consistency Pruning - Evaluation")
    print("=" * 60)

    # 加载配置
    config = load_config(override_file=args.config)
    logger = config.logger

    # 确定评估模式：命令行 > 配置文件 > 默认值
    if args.mode is not None:
        eval_modes = args.mode
    else:
        eval_cfg = getattr(config, 'evaluation_settings', {}) or {}
        eval_modes = eval_cfg.get('eval_mode', ['hard'])
        if isinstance(eval_modes, str):
            eval_modes = [eval_modes]
    logger.info(f"Evaluation modes: {eval_modes}")

    # 获取剪枝层配置
    method_cfg = config.method_settings
    pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])

    # 设置设备
    device = torch.device(config.global_settings.get('device', 'cuda'))
    logger.info(f"Using device: {device}")

    # 加载模型
    logger.info("Loading model...")
    model, processor = load_model(config, device)

    # 加载 checkpoint
    load_checkpoint(model, args.checkpoint, device, logger)

    # 加载数据集
    logger.info("Loading dataset...")
    from engine.datas.loader import load_dataset
    data_bundle = load_dataset(config)

    # 选择数据集 split
    if args.split in data_bundle['splits']:
        eval_dataset = data_bundle['splits'][args.split]
    else:
        available_splits = list(data_bundle['splits'].keys())
        logger.error(f"Split '{args.split}' not found. Available splits: {available_splits}")
        return

    judge = data_bundle['judge']

    dataset_name = config.dataset_settings.get('name', 'unknown')
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Evaluating on '{args.split}' split: {len(eval_dataset)} samples")

    # 确定评估样本数
    if args.max_samples is not None:
        max_samples = args.max_samples
    else:
        trainer_cfg = config.trainer_settings.get('dl_settings', {})
        max_samples = trainer_cfg.get('eval_max_samples', 500)

    logger.info(f"Max samples: {max_samples}")

    # 运行评估
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    for eval_mode in eval_modes:
        logger.info(f"\nEvaluating in '{eval_mode}' mode...")

        eval_result = evaluate(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            judge=judge,
            config=config,
            device=device,
            max_samples=max_samples,
            mode=eval_mode,
        )

        # 打印结果
        print(f"\n[{eval_mode.upper()}] Results:")
        print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")

        if 'avg_kept_ratio' in eval_result:
            print(f"  Avg kept ratio: {eval_result['avg_kept_ratio']:.2%}")

            # 打印每层保留率
            layer_ratios = []
            for layer_idx in pruning_layers:
                kept_key = f'L{layer_idx}_kept'
                n_kept_key = f'L{layer_idx}_n_kept'
                if kept_key in eval_result:
                    if n_kept_key in eval_result:
                        layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}({int(eval_result[n_kept_key])})")
                    else:
                        layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}")

            if layer_ratios:
                print(f"  Per-layer kept: [{', '.join(layer_ratios)}]")

    print("\n" + "=" * 60)
    print("Evaluation completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
