"""Vision Token Pruning - HF Trainer + FSDP 多卡训练

使用 VisionTokenPruningModel (统一模型) + HuggingFace Trainer + FSDP 进行多卡并行训练。

运行方式:
    单卡测试:
        python main_hf_trainer.py

    多卡训练 (推荐):
        torchrun --nproc_per_node=3 main_hf_trainer.py

        或使用 accelerate:
        accelerate launch --num_processes=3 main_hf_trainer.py

特点:
1. 使用 VisionTokenPruningModel 封装所有组件
2. 通过 HF Trainer 的 FSDP 支持多卡训练
3. 自动处理分布式训练的细节
4. 支持 Parameter Groups (不同学习率)
"""

import torch
import os
from typing import Dict, Any

from engine.configs.loader import load_config


def main():
    """主函数"""
    # ==================== 1. 加载配置 ====================
    config = load_config(override_file="configs/vision_token_pruning_fsdp.yaml")
    logger = config["logger"]

    logger.info("=" * 60)
    logger.info("Vision Token Pruning - HF Trainer + FSDP")
    logger.info("=" * 60)

    # ==================== 2. 加载Backbone ====================
    from engine.backbones.loader import load_backbone

    logger.info("加载Backbone...")
    backbone = load_backbone(config)

    # 冻结Backbone参数
    logger.info("冻结Backbone参数...")
    if hasattr(backbone, "model"):
        for param in backbone.model.parameters():
            param.requires_grad = False

    # ==================== 3. 加载数据集 ====================
    from engine.datas.loader import load_dataset

    logger.info("加载Dataset...")
    dataset_bundle = load_dataset(config)

    # ==================== 4. 创建独立组件 ====================
    from method import (
        LearnableTokenMerger,
        LearnableTokenMergerV2,
        LearnableTokenMergerV3,
        LayerSpecificPruner,
        Discriminator,
    )

    device = config.get("global_settings", {}).get("device")
    method_config = config["method_settings"]

    # 4.1 Token Merger (可选)
    token_merger = None
    enable_token_merger = method_config.get("enable_token_merger", False)

    if enable_token_merger:
        logger.info("创建Token Merger...")
        merger_type = method_config.get("merger_type", "simple")

        if merger_type == "fixed_pooling":
            token_merger = LearnableTokenMergerV3(
                d_vision=config["backbone_settings"]["mllm_settings"]["vision_dim"],
                d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
                d_internal=method_config["pruner_d_internal"],
                num_heads=method_config["pruner_num_heads"],
                merge_ratio=method_config["merge_ratio"],
                use_question=True
            ).to(device=device)
        elif merger_type == "question_aware":
            token_merger = LearnableTokenMergerV2(
                d_vision=config["backbone_settings"]["mllm_settings"]["vision_dim"],
                d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
                d_internal=method_config["pruner_d_internal"],
                num_heads=method_config["pruner_num_heads"],
                merge_ratio=method_config["merge_ratio"]
            ).to(device=device)
        else:
            token_merger = LearnableTokenMerger(
                d_model=config["backbone_settings"]["mllm_settings"]["vision_dim"],
                num_heads=method_config["pruner_num_heads"],
                merge_ratio=method_config["merge_ratio"]
            ).to(device=device)

    # 4.2 Layer-Specific Pruners
    logger.info("创建Layer-Specific Pruners...")
    layer_pruners = LayerSpecificPruner(
        d_model=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        layer_indices=method_config["pruning_layers"],
        d_internal=method_config["pruner_d_internal"],
        num_heads=method_config["pruner_num_heads"],
        use_attn_residual=method_config.get("use_attn_residual", False),
        attn_residual_weight=method_config.get("attn_residual_weight", 0.5),
        learnable_attn_weight=method_config.get("learnable_attn_weight", False)
    ).to(device=device)

    # 4.3 Discriminator
    logger.info("创建Discriminator...")
    discriminator = Discriminator(
        d_model=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        num_layers=method_config["disc_num_layers"],
        d_d=method_config["disc_d_d"],
        dropout=method_config["disc_dropout"],
        use_layer_norm=True,
        use_spectral_norm=method_config["disc_use_spectral_norm"]
    ).to(device=device)

    # ==================== 5. 创建统一模型 ====================
    from method.models.unified_model import VisionTokenPruningModel

    logger.info("创建VisionTokenPruningModel...")
    model = VisionTokenPruningModel(
        config=config,
        backbone=backbone,
        layer_pruners=layer_pruners,
        discriminator=discriminator,
        token_merger=token_merger
    )

    # ==================== 6. 创建HF Trainer ====================
    from engine.trainers.loader import load_trainer

    logger.info("创建HF Trainer...")
    trainer = load_trainer(config, dataset_bundle)

    # 构建 HF Trainer（传入统一模型）
    logger.info("构建HF Trainer...")
    trainer.build_trainer(model=model)

    # ==================== 7. 执行训练 ====================
    logger.info("=" * 60)
    logger.info("开始训练...")
    logger.info(f"Token Merger: {'Enabled' if enable_token_merger else 'Disabled'}")
    if enable_token_merger:
        logger.info(f"  - Type: {method_config.get('merger_type', 'simple')}")
        logger.info(f"  - Merge Ratio: {method_config['merge_ratio']}")
    logger.info(f"Pruning Layers: {method_config['pruning_layers']}")
    logger.info(f"Temperature: {method_config['temperature']} → {method_config['temperature_min']}")

    # 检查是否是分布式训练
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        logger.info(f"分布式训练: Rank {rank}/{world_size}")
    else:
        logger.info("单GPU训练")

    logger.info("=" * 60)

    # 训练
    result = trainer.run()

    # ==================== 8. 输出结果 ====================
    logger.info("=" * 60)
    logger.info("训练完成!")
    if result:
        logger.info(f"最终结果: {result}")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
