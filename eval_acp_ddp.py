#!/usr/bin/env python
"""Attention Consistency Pruning - DDP 分布式评估脚本

加载已训练的模型 checkpoint 并进行分布式评估。

启动方式：
    # 单卡评估（使用配置文件中的 checkpoint）
    python eval_acp_ddp.py

    # 单卡评估（命令行指定 checkpoint）
    python eval_acp_ddp.py --checkpoint outputs/checkpoints/checkpoint_final.pt

    # 多卡分布式评估
    torchrun --nproc_per_node=4 eval_acp_ddp.py --checkpoint outputs/checkpoints/checkpoint_final.pt

    # 指定评估模式和样本数
    torchrun --nproc_per_node=4 eval_acp_ddp.py \\
        --checkpoint outputs/checkpoints/checkpoint_final.pt \\
        --mode origin hard \\
        --max_samples 5000

    # 覆盖配置中的阈值
    python eval_acp_ddp.py --thresholds 4:0.5 14:0.5 24:0.5

网格搜索：在配置文件中设置 evaluation_settings.grid_search.enable: true
"""

import os
import sys

# 环境变量设置
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import random
import torch
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Any, List, Optional
from itertools import product
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

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if 'pruner_state_dict' in checkpoint:
        model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
        if logger:
            logger.info("  Loaded pruner_manager state")

    if 'adapter_state_dict' in checkpoint:
        model.adapter_manager.load_state_dict(checkpoint['adapter_state_dict'])
        if logger:
            logger.info("  Loaded adapter_manager state")

    if 'disc_state_dict' in checkpoint:
        model.disc_manager.load_state_dict(checkpoint['disc_state_dict'])
        if logger:
            logger.info("  Loaded disc_manager state")

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


def print_grid_search_progress(
    results: List[Dict[str, Any]],
    origin_acc: float,
    pruning_layers: List[int],
):
    """打印当前所有网格搜索结果（按准确率排序）"""
    if not results:
        return

    # 按准确率排序
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    print("\n" + "-" * 90)
    print(f"Progress: {len(results)} combinations evaluated | Origin Acc: {origin_acc:.2%}")
    print("-" * 90)
    print(f"{'#':<3} {'Acc':<8} {'Rel':<8} {'Kept':<8} {'Thresholds'}")
    print("-" * 90)

    for i, r in enumerate(sorted_results):
        rel_acc = r['accuracy'] / origin_acc if origin_acc > 0 else 0
        thresh_str = '/'.join(f"{r['thresholds'][l]:.2f}" for l in pruning_layers)
        print(f"{i+1:<3} {r['accuracy']:.4f}   {rel_acc:.4f}   {r['avg_kept_ratio']:.4f}   {thresh_str}")

    print("-" * 90)


def run_grid_search(
    model,
    processor,
    eval_dataset,
    judge,
    config,
    device: torch.device,
    max_samples: int,
    pruning_layers: List[int],
    threshold_values: List[float],
    distributed: bool,
    origin_result: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """执行网格搜索（笛卡尔积，每层阈值可不同，随机顺序）"""
    results = []

    # 生成所有组合并随机打乱
    combinations = list(product(threshold_values, repeat=len(pruning_layers)))
    random.shuffle(combinations)

    origin_acc = origin_result['accuracy'] if origin_result else 0.0

    if is_main_process():
        print(f"\nGrid search: {len(combinations)} combinations (randomized)")
        print(f"Layers: {pruning_layers}")
        print(f"Values: {threshold_values}")
        if origin_result:
            print(f"Origin baseline: {origin_acc:.2%}")

    pbar = tqdm(combinations, desc="Grid search", disable=not is_main_process())

    for combo in pbar:
        # 构建阈值字典
        thresholds = {layer: t for layer, t in zip(pruning_layers, combo)}

        # 设置阈值
        model.pruner_manager.set_thresholds(thresholds)

        # 评估
        eval_result = evaluate(
            model=model,
            processor=processor,
            dataset=eval_dataset,
            judge=judge,
            config=config,
            device=device,
            max_samples=max_samples,
            mode='hard',
            distributed=distributed,
        )

        result_entry = {
            'thresholds': thresholds.copy(),
            'accuracy': eval_result['accuracy'],
            'avg_kept_ratio': eval_result.get('avg_kept_ratio', 0),
            'correct': eval_result['correct'],
            'total': eval_result['total'],
        }
        results.append(result_entry)

        # 更新进度条
        if is_main_process():
            thresh_str = '/'.join(f"{t:.2f}" for t in combo)
            rel_acc = eval_result['accuracy'] / origin_acc if origin_acc > 0 else 0
            pbar.set_postfix({
                'thresh': thresh_str,
                'acc': f"{eval_result['accuracy']:.2%}",
                'rel': f"{rel_acc:.2%}",
                'kept': f"{eval_result.get('avg_kept_ratio', 0):.2%}"
            })

            # 每步打印所有结果
            print_grid_search_progress(results, origin_acc, pruning_layers)

    return results


def print_grid_search_results(
    results: List[Dict[str, Any]],
    pruning_layers: List[int],
    origin_result: Optional[Dict[str, Any]] = None,
):
    """打印网格搜索结果"""
    if not results:
        print("No results.")
        return

    origin_acc = origin_result['accuracy'] if origin_result else 0.0

    # 按准确率排序
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)

    print("\n" + "=" * 90)
    print("Grid Search Results (sorted by accuracy)")
    if origin_result:
        print(f"Origin Baseline: {origin_acc:.2%} ({origin_result['correct']}/{origin_result['total']})")
    print("=" * 90)
    print(f"{'Rank':<5} {'Accuracy':<10} {'Rel Acc':<10} {'Kept':<10} {'Thresholds'}")
    print("-" * 90)

    for i, r in enumerate(sorted_results[:20]):
        rel_acc = r['accuracy'] / origin_acc if origin_acc > 0 else 0
        thresh_str = ', '.join(f"L{l}={r['thresholds'][l]:.2f}" for l in pruning_layers)
        print(f"{i+1:<5} {r['accuracy']:.4f}     {rel_acc:.4f}     {r['avg_kept_ratio']:.4f}     {thresh_str}")

    # 最佳配置
    best = sorted_results[0]
    print("\n" + "=" * 90)
    print("Best Configuration")
    print("=" * 90)
    print(f"Accuracy: {best['accuracy']:.4f} ({best['correct']}/{best['total']})")
    if origin_acc > 0:
        print(f"Relative Accuracy: {best['accuracy'] / origin_acc:.4f}")
    print(f"Kept Ratio: {best['avg_kept_ratio']:.4f}")
    print("\nConfig (copy to yaml):")
    print("pruner_thresholds:")
    for layer in pruning_layers:
        print(f"  {layer}: {best['thresholds'][layer]}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Attention Consistency Pruning Evaluation (DDP)")
    parser.add_argument('--config', type=str, default='configs/vision_token_pruning.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file (overrides config)')
    parser.add_argument('--mode', type=str, nargs='+', default=['hard'],
                        choices=['origin', 'hard'],
                        help='Evaluation mode(s)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to evaluate')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'test', 'val'],
                        help='Dataset split')
    parser.add_argument('--thresholds', type=str, nargs='*', default=None,
                        help='Override thresholds: 4:0.5 14:0.5 24:0.5')
    parser.add_argument('--inference_mode', type=str, default=None,
                        choices=['threshold', 'topk'],
                        help='Override inference mode')
    parser.add_argument('--topk_ks', type=str, nargs='*', default=None,
                        help='Override topk k values: 4:360 14:230 24:144')

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

        # 检查是否启用网格搜索
        eval_cfg = config.evaluation_settings
        grid_cfg = eval_cfg.get('grid_search', {})
        grid_search_enabled = grid_cfg.get('enable', False)

        if is_main_process():
            print("=" * 60)
            print("Attention Consistency Pruning - Evaluation")
            if distributed:
                print(f"Distributed mode: world_size={world_size}")
            if grid_search_enabled:
                print("Mode: Grid Search")
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

        # 获取配置
        method_cfg = config.method_settings
        pruning_layers = method_cfg.get('pruning_layers', [4, 14, 24])

        # 处理阈值覆盖
        if args.thresholds:
            override_thresholds = parse_thresholds(args.thresholds)
            if 'pruner_thresholds' not in method_cfg:
                method_cfg['pruner_thresholds'] = {}
            method_cfg['pruner_thresholds'].update(override_thresholds)
            if logger:
                logger.info(f"Override thresholds: {override_thresholds}")

        if args.inference_mode:
            method_cfg['pruner_inference_mode'] = args.inference_mode

        if args.topk_ks:
            override_topk = {int(k): int(v) for k, v in parse_thresholds(args.topk_ks).items()}
            method_cfg['pruner_topk_ks'] = override_topk

        # 加载模型
        if logger:
            logger.info("Loading model...")
        model, processor = load_model(config, device, local_rank)
        load_checkpoint(model, checkpoint_path, device, logger)

        if logger:
            logger.info(f"Pruning layers: {pruning_layers}")

        # 加载数据集
        if logger:
            logger.info("Loading dataset...")
        original_logger = config.logger
        if not is_main_process():
            config.logger = None
        from engine.datas.loader import load_dataset
        data_bundle = load_dataset(config)
        config.logger = original_logger

        if args.split not in data_bundle['splits']:
            if logger:
                logger.error(f"Split '{args.split}' not found.")
            return

        eval_dataset = data_bundle['splits'][args.split]
        judge = data_bundle['judge']

        if logger:
            logger.info(f"Dataset: {config.dataset_settings.get('name', 'unknown')}")
            logger.info(f"Split: {args.split}, samples: {len(eval_dataset)}")

        # 确定样本数
        max_samples = args.max_samples
        if max_samples is None:
            max_samples = config.trainer_settings.get('dl_settings', {}).get('eval_max_samples', 500)
        if logger:
            logger.info(f"Max samples: {max_samples}")

        # 同步
        if distributed:
            dist.barrier()

        # 网格搜索或普通评估
        if grid_search_enabled:
            # 从配置读取阈值列表
            threshold_values = grid_cfg.get('threshold_values', [0.2, 0.3, 0.4, 0.5, 0.6])

            if logger:
                logger.info(f"Grid search values: {threshold_values}")

            # 先评估 origin 作为 baseline（缓存，只需计算一次）
            if logger:
                logger.info("Evaluating origin mode as baseline...")
            if is_main_process():
                print("\n" + "=" * 60)
                print("Evaluating Origin Baseline (cached for grid search)")
                print("=" * 60)

            origin_result = evaluate(
                model=model,
                processor=processor,
                dataset=eval_dataset,
                judge=judge,
                config=config,
                device=device,
                max_samples=max_samples,
                mode='origin',
                distributed=distributed,
            )

            if is_main_process():
                print(f"Origin Accuracy: {origin_result['accuracy']:.2%} ({origin_result['correct']}/{origin_result['total']})")
                print("=" * 60)

            # 执行网格搜索
            grid_results = run_grid_search(
                model=model,
                processor=processor,
                eval_dataset=eval_dataset,
                judge=judge,
                config=config,
                device=device,
                max_samples=max_samples,
                pruning_layers=pruning_layers,
                threshold_values=threshold_values,
                distributed=distributed,
                origin_result=origin_result,
            )

            if is_main_process():
                print_grid_search_results(grid_results, pruning_layers, origin_result)
        else:
            # 普通评估
            if is_main_process():
                print("\n" + "=" * 60)
                print("Evaluation Results")
                print("=" * 60)

            # 如果 mode 包含 origin，先评估并缓存（用于后续 hard 模式的相对准确率）
            origin_result = None
            if 'origin' in args.mode:
                if logger:
                    logger.info("Evaluating 'origin' mode...")

                origin_result = evaluate(
                    model=model,
                    processor=processor,
                    dataset=eval_dataset,
                    judge=judge,
                    config=config,
                    device=device,
                    max_samples=max_samples,
                    mode='origin',
                    distributed=distributed,
                )

                if is_main_process():
                    print(f"\n[ORIGIN]")
                    print(f"  Accuracy: {origin_result['accuracy']:.2%} ({origin_result['correct']}/{origin_result['total']})")

            # 评估其他模式（跳过已评估的 origin）
            for eval_mode in args.mode:
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
                    max_samples=max_samples,
                    mode=eval_mode,
                    distributed=distributed,
                )

                if is_main_process():
                    print(f"\n[{eval_mode.upper()}]")
                    print(f"  Accuracy: {eval_result['accuracy']:.2%} ({eval_result['correct']}/{eval_result['total']})")

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
