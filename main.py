"""Vision Token Pruning with GAN - 主训练脚本

使用新的Engine架构实现对抗训练的视觉token剪枝。
"""

import torch
from typing import Dict, Any

from engine.configs.loader import load_config


# ===================== GPU显存保护 =====================

def reserve_gpu_memory(reserve_ratio=0.90):
    """
    预分配GPU显存，防止被其他程序抢占

    注意：如果使用了CUDA_VISIBLE_DEVICES，这里会自动处理可见的GPU

    参数:
        reserve_ratio: 预留比例（0-1），默认0.90表示预留90%显存
    """
    print(f"🛡️  正在预分配GPU显存以防止被抢占...")

    reserved_tensors = []
    num_gpus = torch.cuda.device_count()  # 获取当前可见的GPU数量

    if num_gpus == 0:
        print("   ⚠️  未检测到可用GPU")
        return reserved_tensors

    for device_id in range(num_gpus):
        try:
            # 获取GPU总显存
            total_memory = torch.cuda.get_device_properties(device_id).total_memory
            reserve_size = int(total_memory * reserve_ratio)

            # 分配一个大tensor占住显存
            # 使用int8节省空间（1 byte per element）
            num_elements = reserve_size // 1  # int8 = 1 byte
            dummy_tensor = torch.empty(num_elements, dtype=torch.int8, device=f'cuda:{device_id}')
            reserved_tensors.append(dummy_tensor)

            gpu_name = torch.cuda.get_device_name(device_id)
            print(f"   GPU {device_id} ({gpu_name}): 已预留 {reserve_size / 1024**3:.2f} GB / {total_memory / 1024**3:.2f} GB")
        except Exception as e:
            print(f"   ⚠️  GPU {device_id} 预分配失败: {e}")

    return reserved_tensors

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
        LayerSpecificPruner,
        Discriminator,
        train_step,
        eval_step
    )

    logger = config["logger"]
    backbone = cache["backbone"]
    dataset_bundle = cache["dataset_bundle"]
    device = config.get("global_settings", {}).get("device")

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

    if merger_type == "question_aware":
        token_merger = LearnableTokenMergerV2(
            d_vision=config["backbone_settings"]["mllm_settings"]["vision_dim"],
            d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
            d_internal=config["method_settings"]["pruner_d_internal"],
            num_heads=config["method_settings"]["pruner_num_heads"],
            merge_ratio=config["method_settings"]["merge_ratio"]
        ).to(device=device)
    else:
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
        num_heads=config["method_settings"]["pruner_num_heads"]
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

    trainer.register_model("token_merger", token_merger)
    trainer.register_model("layer_pruners", layer_pruners)
    trainer.register_model("discriminator", discriminator)
    trainer.register_model("backbone", backbone)

    # ==================== 添加参数组 ====================
    # 拆分为3个独立参数组，支持不同学习率：
    # 1. token_merger: Token合并器（输入阶段剪枝）
    # 2. layer_pruners: 逐层剪枝器（LLM内部剪枝）
    # 3. discriminator: 判别器

    trainer.add_param_group("token_merger", list(token_merger.parameters()))
    trainer.add_param_group("layer_pruners", list(layer_pruners.parameters()))
    trainer.add_param_group("discriminator", list(discriminator.parameters()))

    # ==================== 创建优化器 ====================

    trainer.setup_optimizers()

    # ==================== 注册训练和评估函数 ====================

    trainer.register_train_step(train_step)
    trainer.register_eval_step(eval_step)

    # ==================== 执行训练 ====================

    logger.info("开始训练...")
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

    # ========== GPU显存保护（防止被其他程序抢占） ==========
    # 自动检测所有可见的GPU（考虑CUDA_VISIBLE_DEVICES）
    if torch.cuda.is_available():
        logger.info("🛡️  启用GPU显存保护...")
        reserved_tensors = reserve_gpu_memory(reserve_ratio=0.90)
        # 注意: reserved_tensors不能被删除，否则显存会被释放
        # 训练过程中PyTorch会自动管理实际使用的显存
    else:
        logger.info("⚠️  未检测到GPU，跳过显存保护")
        reserved_tensors = []

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
