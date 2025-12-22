"""Vision Token Pruning with GAN - 主训练脚本

使用新的Engine架构实现对抗训练的视觉token剪枝。
"""

import torch
from typing import Dict, Any

from engine.configs.loader import load_config

# ===================== Manager函数 =====================

def preload_fn(config: Dict) -> Dict[str, Any]:
    """预加载重量级资源"""
    from engine.datas.loader import load_dataset
    from engine.backbones.loader import load_backbone

    logger = config["logger"]
    logger.info("预加载Backbone和Dataset...")
    
    backbone = load_backbone(config)
    
    # 冻结Backbone参数，避免梯度累积和显存浪费
    logger.info("冻结Backbone参数...")
    if hasattr(backbone, "model"):
        for param in backbone.model.parameters():
            param.requires_grad = False
    
    dataset_bundle = load_dataset(config)
    
    return {
        "backbone": backbone,
        "dataset_bundle": dataset_bundle
    }


def run_fn(config: Dict, cache: Dict[str, Any]) -> Dict[str, Any]:
    """执行训练"""
    from engine.trainers.loader import load_trainer
    from method import (
        LearnableTokenMerger,
        LearnableTokenMergerV2,
        LearnableTokenMergerV3,
        LayerSpecificPruner,
        Discriminator,
        train_step,
        train_step_batch,
        eval_step
    )

    logger = config["logger"]
    backbone = cache["backbone"]
    dataset_bundle = cache["dataset_bundle"]
    device = config.get("global_settings", {}).get("device")

    # 根据配置选择训练函数
    enable_true_batch = config["backbone_settings"]["mllm_settings"].get("enable_true_batch", False)
    if enable_true_batch:
        logger.info("⚡ Batch化模式已启用 - 使用 train_step_batch")
        train_step_fn = train_step_batch
    else:
        logger.info("📝 标准模式 - 使用 train_step")
        train_step_fn = train_step

    # 将dataset_bundle放入config供eval_step使用
    config["_dataset_bundle"] = dataset_bundle

    logger.info("创建Trainer...")
    trainer = load_trainer(config, dataset_bundle)

    # 获取 backbone 的输出设备
    backbone_output_device = getattr(backbone, 'output_device', backbone.device)
    model_dtype = torch.float16 if str(backbone_output_device).startswith('cuda') else torch.float32

    # ==================== 创建两阶段剪枝模块 ====================

    # 1. Token Merger
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
            use_question=True  # 默认启用question-aware
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

    # 2. Layer-Specific Pruners
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

    # 3. Discriminator
    logger.info("创建Discriminator...")
    discriminator = Discriminator(
        d_model=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        num_layers=config["method_settings"]["disc_num_layers"],
        d_d=config["method_settings"]["disc_d_d"],
        dropout=config["method_settings"]["disc_dropout"],
        use_layer_norm=True,
        use_spectral_norm=config["method_settings"]["disc_use_spectral_norm"]
    ).to(device=device)

    # ==================== 注册模型 ====================

    # trainer.register_model("token_merger", token_merger)
    trainer.register_model("layer_pruners", layer_pruners)
    trainer.register_model("discriminator", discriminator)
    trainer.register_model("backbone", backbone)

    # ==================== 添加参数组 ====================
    # 拆分为3个独立参数组，支持不同学习率：
    # 1. token_merger: Token合并器（输入阶段剪枝）
    # 2. layer_pruners: 逐层剪枝器（LLM内部剪枝）
    # 3. discriminator: 判别器

    # trainer.add_param_group("token_merger", list(token_merger.parameters()))
    trainer.add_param_group("layer_pruners", list(layer_pruners.parameters()))
    trainer.add_param_group("discriminator", list(discriminator.parameters()))

    # ==================== 创建优化器 ====================

    trainer.setup_optimizers()

    # ==================== 注册训练和评估函数 ====================

    trainer.register_train_step(train_step_fn)  # 使用动态选择的训练函数
    trainer.register_eval_step(eval_step)

    # ==================== 执行训练 ====================

    logger.info("开始训练...")
    logger.info(f"训练函数: {train_step_fn.__name__}")  # 显示使用的训练函数
    logger.info(f"Token Merger类型: {merger_type}")
    logger.info(f"Merge Ratio: {config['method_settings']['merge_ratio']}")
    logger.info(f"Pruning Layers: {config['method_settings']['pruning_layers']}")
    logger.info(f"Temperature: {config['method_settings']['temperature']} → {config['method_settings']['temperature_min']}")

    result = trainer.run()

    return result


# ===================== 主函数 =====================

def main():
    """主函数"""
    # 加载配置
    config = load_config(override_file="configs/vision_token_pruning.yaml")
    logger = config["logger"]

    from engine.managers.loader import load_manager

    logger.info("=" * 60)
    logger.info("Vision Token Pruning with GAN")
    logger.info("=" * 60)

    # 创建Manager
    manager = load_manager(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=None,  # 单任务模式
        result_handler_fn=None
    )
    
    # 启动训练
    manager.start()
    manager.wait()
    
    # 获取结果摘要
    summary = manager.get_summary()
    logger.info("=" * 60)
    logger.info("训练完成!")
    logger.info(f"总任务数: {summary['total_tasks']}")
    logger.info(f"已完成: {summary['completed']}")
    logger.info(f"失败: {summary['failed']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
