"""Vision Token Pruning with GAN - 主训练脚本 (HuggingFace Trainer 版本)

使用 HuggingFace Trainer 实现并行训练：
- 支持 DDP 分布式训练
- 支持混合精度训练 (fp16/bf16)
- 支持梯度累积
- 自动 checkpoint 保存

启动方式：
- 单 GPU: python ./main.py
- 多 GPU DDP: torchrun --nproc_per_node=2 ./main.py
- 或使用 accelerate: accelerate launch ./main.py
"""

import os
import torch
from transformers import TrainingArguments

from configs.loader import load_config
from datas.loader import load_dataset
from backbones.loader import load_backbone
from models import LayerSpecificPruner, Discriminator, PruningModelWrapper
from training import PruningTrainer


def main():
    """主函数"""
    # 加载配置
    config = load_config(override_file="configs/vision_token_pruning.yaml")
    logger = config.get("logger")

    # 获取 local_rank（DDP 模式下由 torchrun 设置）
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # 只在主进程打印
    if local_rank <= 0:
        logger.info("=" * 60)
        logger.info("Vision Token Pruning with GAN (HuggingFace Trainer)")
        logger.info("=" * 60)

    # 设置设备（FSDP 会自动管理）
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    # ==================== 加载 Backbone ====================
    if local_rank <= 0:
        logger.info("加载 Backbone...")
    backbone = load_backbone(config)

    # ==================== 加载数据集 ====================
    if local_rank <= 0:
        logger.info("加载数据集...")
    dataset_bundle = load_dataset(config)

    # 获取训练集
    train_dataset = dataset_bundle["splits"]["train"]

    # 获取验证集（如果有）- 优先用 val，否则用 test
    eval_dataset = dataset_bundle["splits"].get("val") or dataset_bundle["splits"].get("test")

    # 获取 judge_fn
    judge_fn = dataset_bundle.get("judge")

    # ==================== 创建模型组件 ====================
    method_cfg = config.get("method_settings", {})
    backbone_cfg = config.get("backbone_settings", {})
    mllm_cfg = backbone_cfg.get("mllm_settings", {})
    trainer_cfg = config.get("trainer_settings", {}).get("dl_settings", {})

    # Layer-Specific Pruners（不指定 device，让 FSDP 管理）
    if local_rank <= 0:
        logger.info("创建 Layer-Specific Pruners...")
    layer_pruners = LayerSpecificPruner(
        d_model=mllm_cfg.get("hidden_dim", 4096),
        d_text=mllm_cfg.get("hidden_dim", 4096),
        layer_indices=method_cfg.get("pruning_layers", [5, 15, 25]),
        d_internal=method_cfg.get("pruner_d_internal", 256),
        num_heads=method_cfg.get("pruner_num_heads", 4),
        use_attn_residual=method_cfg.get("use_attn_residual", False),
        attn_residual_weight=method_cfg.get("attn_residual_weight", 0.5),
        learnable_attn_weight=method_cfg.get("learnable_attn_weight", False)
    )

    # Discriminator
    if local_rank <= 0:
        logger.info("创建 Discriminator...")
    discriminator = Discriminator(
        d_model=mllm_cfg.get("hidden_dim", 4096),
        num_layers=method_cfg.get("disc_num_layers", 3),
        d_d=method_cfg.get("disc_d_d", 512),
        dropout=method_cfg.get("disc_dropout", 0.1),
        use_layer_norm=True,
        use_spectral_norm=method_cfg.get("disc_use_spectral_norm", True)
    )

    # 创建统一的 Wrapper 模型（包含 backbone、pruner、discriminator）
    if local_rank <= 0:
        logger.info("创建 PruningModelWrapper...")
    model = PruningModelWrapper(
        backbone=backbone,
        layer_pruners=layer_pruners,
        discriminator=discriminator,
        freeze_backbone=True
    )

    # ==================== 配置 TrainingArguments ====================
    output_dir = config.get("global_settings", {}).get("save_dir", "outputs/.")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=trainer_cfg.get("epochs", 3),
        per_device_train_batch_size=trainer_cfg.get("batch_size", 4),
        per_device_eval_batch_size=trainer_cfg.get("batch_size", 4),
        learning_rate=trainer_cfg.get("optimizers", {}).get("layer_pruners", {}).get("lr", 1e-4),
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=trainer_cfg.get("print_loss_every_batches", 10),
        save_steps=trainer_cfg.get("save_every_batches", 500),
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=trainer_cfg.get("eval_every_batches", 500) if eval_dataset else None,
        save_total_limit=3,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available() and trainer_cfg.get("fp16", True),
        bf16=trainer_cfg.get("bf16", False),
        gradient_accumulation_steps=trainer_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=trainer_cfg.get("grad_clip_max_norm", 1.0),
        dataloader_pin_memory=True,
        # FSDP 设置 - 将模型参数分片到多个 GPU
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "transformer_layer_cls_to_wrap": ["LlamaDecoderLayer"],
            "use_orig_params": True,  # 保持原始参数形状，让 get_input_embeddings() 正常工作
        },
    )

    # ==================== 创建 Trainer ====================
    if local_rank <= 0:
        logger.info("创建 PruningTrainer...")
    trainer = PruningTrainer(
        model=model,  # 传入 wrapper，包含所有组件
        args=training_args,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        judge_fn=judge_fn,
    )

    # ==================== 开始训练 ====================
    if local_rank <= 0:
        logger.info("开始训练...")
        logger.info(f"Pruning Layers: {method_cfg.get('pruning_layers', [5, 15, 25])}")
        logger.info(f"Temperature: {method_cfg.get('temperature', 1.0)} → {method_cfg.get('temperature_min', 0.2)}")
        logger.info(f"Target Token Num: {method_cfg.get('target_token_num', 144)}")
        logger.info(f"Batch Size: {training_args.per_device_train_batch_size}")
        logger.info(f"Epochs: {training_args.num_train_epochs}")
        logger.info(f"FP16: {training_args.fp16}")

    trainer.train()

    # ==================== 保存模型 ====================
    if local_rank <= 0:
        logger.info("保存模型...")
    trainer.save_model()

    if local_rank <= 0:
        logger.info("=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
