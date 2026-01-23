#!/usr/bin/env python
"""Attention Consistency Pruning - DDP 分布式评估脚本

加载已训练的模型 checkpoint 并进行分布式评估。

启动方式：
    # 单卡评估
    python eval_acp_ddp.py --checkpoint outputs/checkpoints/checkpoint_final.pt

    # 多卡分布式评估
    torchrun --nproc_per_node=4 eval_acp_ddp.py --checkpoint outputs/checkpoints/checkpoint_final.pt

    # 指定评估模式和样本数
    torchrun --nproc_per_node=4 eval_acp_ddp.py \
        --checkpoint outputs/checkpoints/checkpoint_final.pt \
        --mode origin hard \
        --max_samples 5000

    # 覆盖配置中的阈值
    torchrun --nproc_per_node=4 eval_acp_ddp.py \
        --checkpoint outputs/checkpoints/checkpoint_final.pt \
        --thresholds 4:0.5 14:0.5 24:0.5
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
    preprocess_batch,
    evaluate,
)


def load_checkpoint(
    model,
    checkpoint_path: str,
    device: torch.device,
    logger=None
) -> Dict[str, Any]:
    """加载 checkpoint 到模型

    参数:
        model: PrunableLlavaForConditionalGeneration 模型
        checkpoint_path: checkpoint 文件路径
        device: 设备
        logger: 日志器（仅主进程有效）

    返回:
        checkpoint 字典（包含 step 等元数据）
    """
    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 加载 pruner 状态
    if 'pruner_state_dict' in checkpoint:
        model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
        if logger:
            logger.info("  Loaded pruner_manager state")
    else:
        if logger:
            logger.warning("  No pruner_state_dict found in checkpoint")

    # 加载 adapter 状态
    if 'adapter_state_dict' in checkpoint:
        model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
        if logger:
            logger.info("  Loaded adapter_manager state")
    else:
        if logger:
            logger.warning("  No adapter_state_dict found in checkpoint")

    # 加载 discriminator 状态（评估时可选，但保持一致性）
    if 'disc_state_dict' in checkpoint:
        model.disc_manager.load_state_dict(checkpoint['disc_state_dict'])
        if logger:
            logger.info("  Loaded disc_manager state")
    else:
        if logger:
            logger.warning("  No disc_state_dict found in checkpoint")

    # 打印 checkpoint 信息
    if logger:
        if 'step' in checkpoint:
            logger.info(f"  Checkpoint from step {checkpoint['step']}")
        logger.info("Checkpoint loaded successfully.")

    return checkpoint


def parse_thresholds(threshold_strs: List[str]) -> Dict[int, float]:
    """解析命令行阈值参数

    格式: ["4:0.5", "14:0.5", "24:0.5"]
    返回: {4: 0.5, 14: 0.5, 24: 0.5}
    """
    thresholds = {}
    for s in threshold_strs:
        parts = s.split(':')
        if len(parts) == 2:
            layer_idx = int(parts[0])
            threshold = float(parts[1])
            thresholds[layer_idx] = threshold
    return thresholds


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Consistency Pruning Evaluation (DDP)")
    parser.add_argument('--config', type=str, default='configs/vision_token_pruning.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--mode', type=str, nargs='+', default=['hard'],
                        choices=['origin', 'hard'],
                        help='Evaluation mode(s): origin (no pruning), hard (with pruning)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to evaluate (default: use config value)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'val'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--thresholds', type=str, nargs='*', default=None,
                        help='Override pruning thresholds, format: 4:0.5 14:0.5 24:0.5')
    parser.add_argument('--inference_mode', type=str, default=None,
                        choices=['threshold', 'topk'],
                        help='Override inference mode (threshold or topk)')
    parser.add_argument('--topk_ks', type=str, nargs='*', default=None,
                        help='Override topk k values, format: 4:360 14:230 24:144')

    args = parser.parse_args()

    # 检测是否在分布式环境中
    if 'LOCAL_RANK' in os.environ:
        # 分布式模式
        rank, world_size, local_rank, device = setup_distributed()
        distributed = True
    else:
        # 单卡模式
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        distributed = False

    try:
        if is_main_process():
            print("=" * 60)
            print("Attention Consistency Pruning - Evaluation")
            if distributed:
                print(f"Distributed mode: world_size={world_size}")
            else:
                print("Single GPU mode")
            print("=" * 60)

        # 加载配置
        config = load_config(override_file=args.config)

        # 非主进程禁用 logger
        if not is_main_process():
            config.logger = None

        logger = config.logger

        # 获取剪枝层配置
        method_cfg = config.method_settings
        pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])

        # 处理命令行阈值覆盖
        if args.thresholds:
            override_thresholds = parse_thresholds(args.thresholds)
            # 更新配置
            if 'pruner_thresholds' not in method_cfg:
                method_cfg['pruner_thresholds'] = {}
            method_cfg['pruner_thresholds'].update(override_thresholds)
            if logger:
                logger.info(f"Override thresholds: {override_thresholds}")

        # 处理推理模式覆盖
        if args.inference_mode:
            method_cfg['pruner_inference_mode'] = args.inference_mode
            if logger:
                logger.info(f"Override inference mode: {args.inference_mode}")

        # 处理 topk_ks 覆盖
        if args.topk_ks:
            override_topk = parse_thresholds(args.topk_ks)  # 复用解析函数
            override_topk = {k: int(v) for k, v in override_topk.items()}
            method_cfg['pruner_topk_ks'] = override_topk
            if logger:
                logger.info(f"Override topk_ks: {override_topk}")

        # 加载模型
        if logger:
            logger.info("Loading model...")
        model, processor = load_model(config, device, local_rank)

        # 加载 checkpoint
        checkpoint = load_checkpoint(model, args.checkpoint, device, logger)

        # 打印当前配置
        if logger:
            logger.info(f"Model loaded. Pruning layers: {pruning_layers}")

            # 打印推理配置
            inference_mode = method_cfg.get('pruner_inference_mode', 'threshold')
            logger.info(f"Inference mode: {inference_mode}")

            if inference_mode == 'threshold':
                thresholds = method_cfg.get('pruner_thresholds', {})
                if thresholds:
                    thresh_str = ', '.join(f"L{k}={v}" for k, v in sorted(thresholds.items()))
                    logger.info(f"Thresholds: {thresh_str}")
            elif inference_mode == 'topk':
                topk_ks = method_cfg.get('pruner_topk_ks', {})
                if topk_ks:
                    topk_str = ', '.join(f"L{k}={v}" for k, v in sorted(topk_ks.items()))
                    logger.info(f"TopK values: {topk_str}")

        # 加载数据集
        if logger:
            logger.info("Loading dataset...")

        # 临时保存原始 logger
        original_logger = config.logger
        if not is_main_process():
            config.logger = None

        from engine.datas.loader import load_dataset
        data_bundle = load_dataset(config)

        # 恢复 logger
        config.logger = original_logger

        # 选择数据集 split
        if args.split in data_bundle['splits']:
            eval_dataset = data_bundle['splits'][args.split]
        else:
            available_splits = list(data_bundle['splits'].keys())
            if logger:
                logger.error(f"Split '{args.split}' not found. Available splits: {available_splits}")
            return

        judge = data_bundle['judge']

        dataset_name = config.dataset_settings.get('name', 'unknown')
        if logger:
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"Evaluating on '{args.split}' split: {len(eval_dataset)} samples")

        # 确定评估样本数
        if args.max_samples is not None:
            max_samples = args.max_samples
        else:
            trainer_cfg = config.trainer_settings.get('dl_settings', {})
            max_samples = trainer_cfg.get('eval_max_samples', 500)

        if logger:
            logger.info(f"Max samples: {max_samples}")

        # 同步所有进程
        if distributed:
            dist.barrier()

        # 运行评估
        if is_main_process():
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)

        results = {}
        for eval_mode in args.mode:
            if logger:
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
                distributed=distributed,
            )

            results[eval_mode] = eval_result

            # 打印结果（只在主进程）
            if is_main_process():
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
                                layer_ratios.append(
                                    f"L{layer_idx}={eval_result[kept_key]:.2%}({int(eval_result[n_kept_key])})"
                                )
                            else:
                                layer_ratios.append(f"L{layer_idx}={eval_result[kept_key]:.2%}")

                    if layer_ratios:
                        print(f"  Per-layer kept: [{', '.join(layer_ratios)}]")

        if is_main_process():
            print("\n" + "=" * 60)
            print("Evaluation completed.")
            print("=" * 60)

            # 输出结果摘要（方便复制）
            print("\nSummary (for copy):")
            for eval_mode, result in results.items():
                summary_parts = [f"{eval_mode}: acc={result['accuracy']:.4f}"]
                if 'avg_kept_ratio' in result:
                    summary_parts.append(f"kept={result['avg_kept_ratio']:.4f}")
                print("  " + ", ".join(summary_parts))

    finally:
        # 清理分布式环境
        if distributed:
            cleanup_distributed()


if __name__ == "__main__":
    main()
