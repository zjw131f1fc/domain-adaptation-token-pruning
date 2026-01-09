"""Vision Token Pruning - 简化版训练脚本

直接顺序执行训练流程，不使用manager管理。
"""

import torch
from typing import Dict, Any

from engine.configs.loader import load_config


def main():
    """主函数 - 直接顺序执行"""
    # ==================== 1. 加载配置 ====================
    config = load_config(override_file="configs/vision_token_pruning_fsdp.yaml")
    logger = config["logger"]
    device = config.get("global_settings", {}).get("device")

    logger.info("=" * 60)
    logger.info("Vision Token Pruning - 简化版")
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

    # 将dataset_bundle放入config供eval_step使用
    config["_dataset_bundle"] = dataset_bundle

    # ==================== 4. 创建Trainer ====================
    from engine.trainers.loader import load_trainer

    logger.info("创建Trainer...")
    trainer = load_trainer(config, dataset_bundle)

    # ==================== 5. 创建方法模块 ====================
    from method import (
        LearnableTokenMerger,
        LearnableTokenMergerV2,
        LearnableTokenMergerV3,
        LayerSpecificPruner,
        Discriminator,
        train_step,
        eval_step
    )

    # 获取backbone输出设备和数据类型
    backbone_output_device = getattr(backbone, 'output_device', backbone.device)
    model_dtype = torch.float16 if str(backbone_output_device).startswith('cuda') else torch.float32

    # 5.1 Token Merger
    logger.info("创建Token Merger...")
    merger_type = config["method_settings"].get("merger_type", "simple")

    if merger_type == "fixed_pooling":
        # V3: 固定输出M个tokens的可学习池化（推荐）
        token_merger = LearnableTokenMergerV3(
            d_vision=config["backbone_settings"]["mllm_settings"]["vision_dim"],
            d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
            d_internal=config["method_settings"]["pruner_d_internal"],
            num_heads=config["method_settings"]["pruner_num_heads"],
            merge_ratio=config["method_settings"]["merge_ratio"],
            use_question=True
        ).to(device=device)
    elif merger_type == "question_aware":
        # V2: Question-aware with top-k
        token_merger = LearnableTokenMergerV2(
            d_vision=config["backbone_settings"]["mllm_settings"]["vision_dim"],
            d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
            d_internal=config["method_settings"]["pruner_d_internal"],
            num_heads=config["method_settings"]["pruner_num_heads"],
            merge_ratio=config["method_settings"]["merge_ratio"]
        ).to(device=device)
    else:
        # V1: Simple with top-k
        token_merger = LearnableTokenMerger(
            d_model=config["backbone_settings"]["mllm_settings"]["vision_dim"],
            num_heads=config["method_settings"]["pruner_num_heads"],
            merge_ratio=config["method_settings"]["merge_ratio"]
        ).to(device=device)

    # 5.2 Layer-Specific Pruners
    logger.info("创建Layer-Specific Pruners...")
    layer_pruners = LayerSpecificPruner(
        d_model=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        layer_indices=config["method_settings"]["pruning_layers"],
        d_internal=config["method_settings"]["pruner_d_internal"],
        num_heads=config["method_settings"]["pruner_num_heads"],
        use_attn_residual=config["method_settings"].get("use_attn_residual", False),
        attn_residual_weight=config["method_settings"].get("attn_residual_weight", 0.5),
        learnable_attn_weight=config["method_settings"].get("learnable_attn_weight", False)
    ).to(device=device)

    # 5.3 Discriminator
    logger.info("创建Discriminator...")
    discriminator = Discriminator(
        d_model=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        num_layers=config["method_settings"]["disc_num_layers"],
        d_d=config["method_settings"]["disc_d_d"],
        dropout=config["method_settings"]["disc_dropout"],
        use_layer_norm=True,
        use_spectral_norm=config["method_settings"]["disc_use_spectral_norm"]
    ).to(device=device)

    # ==================== 6. 注册模型和参数 ====================
    logger.info("注册模型到Trainer...")

    # 注册模型
    trainer.register_model("layer_pruners", layer_pruners)
    trainer.register_model("discriminator", discriminator)
    trainer.register_model("backbone", backbone)

    # 添加参数组（支持不同学习率）
    trainer.add_param_group("layer_pruners", list(layer_pruners.parameters()))
    trainer.add_param_group("discriminator", list(discriminator.parameters()))

    # ==================== 7. 创建优化器 ====================
    logger.info("创建优化器...")
    trainer.setup_optimizers()

    # ==================== 8. 注册训练/评估函数 ====================
    logger.info("注册训练和评估函数...")
    trainer.register_train_step(train_step)
    trainer.register_eval_step(eval_step)

    # ==================== 9. 执行训练 ====================
    logger.info("=" * 60)
    logger.info("开始训练...")
    logger.info(f"Token Merger类型: {merger_type}")
    logger.info(f"Merge Ratio: {config['method_settings']['merge_ratio']}")
    logger.info(f"Pruning Layers: {config['method_settings']['pruning_layers']}")
    logger.info(f"Temperature: {config['method_settings']['temperature']} → {config['method_settings']['temperature_min']}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    result = trainer.run()

    # ==================== 10. 输出结果 ====================
    logger.info("=" * 60)
    logger.info("训练完成!")
    if result:
        logger.info(f"最终结果: {result}")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    main()
