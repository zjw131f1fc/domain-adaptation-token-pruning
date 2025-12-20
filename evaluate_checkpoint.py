"""直接评估checkpoint脚本

加载指定的.pt文件，直接调用eval_step进行评估，完全绕开trainer。
复用main.py中的preload_fn和模型创建逻辑。

用法:
    1. 修改CHECKPOINT_PATH指定要评估的checkpoint文件
    2. (可选) 修改CONFIG_OVERRIDES覆盖配置参数
    3. 运行: python evaluate_checkpoint.py
"""

import torch
from typing import Dict

from engine.configs.loader import load_config
from main import preload_fn  # 复用main.py的preload函数
from method import (
    LearnableTokenMerger,
    LearnableTokenMergerV2,
    LearnableTokenMergerV3,
    LayerSpecificPruner,
    eval_step  # 直接使用eval_step
)
from tqdm import tqdm

# ==================== 配置区域 ====================

# 要评估的checkpoint路径
CHECKPOINT_PATH = "outputs/checkpoints/epoch1_batch2000.pt"

# 配置文件路径（默认使用vision_token_pruning.yaml）
CONFIG_FILE = "configs/vision_token_pruning.yaml"

# 配置覆盖（可选，用于修改评估参数）
# load_config的override_dict参数会自动递归合并到默认配置中
CONFIG_OVERRIDES = {
    "global_settings": {
        "study_name": "eval_checkpoint"  # 任务名
    },
    "evaluation_settings": {
        "eval_mode": ["origin", "hard"]  # 评估模式
    },
    "dataset_settings": {
        "split": {
            "train": 1,  # train样本数（评估时不需要train，设为最小值）
            "test": 10000  # 测试集大小
        }
    },
    "trainer_settings": {
        "dl_settings": {
            "batch_size": 10  # batch size
        }
    }
}

# ==================== 实现区域 ====================

def create_models_from_config(config: Dict, backbone, device: torch.device):
    """创建模型（完全复用main.py中run_fn的逻辑）"""
    logger = config["logger"]

    # 1. Token Merger
    logger.info("创建Token Merger...")
    merger_type = config["method_settings"].get("merger_type", "simple")

    if merger_type == "fixed_pooling":
        token_merger = LearnableTokenMergerV3(
            d_vision=config["backbone_settings"]["mllm_settings"]["vision_dim"],
            d_text=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
            d_internal=config["method_settings"]["pruner_d_internal"],
            num_heads=config["method_settings"]["pruner_num_heads"],
            merge_ratio=config["method_settings"]["merge_ratio"],
            use_question=True
        ).to(device=device)
    elif merger_type == "question_aware":
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
        num_heads=config["method_settings"]["pruner_num_heads"],
        use_attn_residual=config["method_settings"].get("use_attn_residual", False),
        attn_residual_weight=config["method_settings"].get("attn_residual_weight", 0.5),
        learnable_attn_weight=config["method_settings"].get("learnable_attn_weight", False)
    ).to(device=device)

    return {
        "token_merger": token_merger,
        "layer_pruners": layer_pruners
    }


def load_checkpoint_into_models(checkpoint_path: str, models: Dict, config: Dict):
    """从checkpoint加载模型参数"""
    logger = config["logger"]
    logger.info(f"加载checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = torch.device(config["global_settings"]["device"])

    # 加载token_merger参数
    token_merger = models["token_merger"]
    if token_merger is not None:
        if "param_groups" in checkpoint and "token_merger" in checkpoint["param_groups"]:
            token_merger_params = checkpoint["param_groups"]["token_merger"]
            state_dict = {}
            param_names = [name for name, _ in token_merger.named_parameters()]
            for name, param_tensor in zip(param_names, token_merger_params):
                state_dict[name] = param_tensor.to(device)
            token_merger.load_state_dict(state_dict, strict=False)
            logger.info(f"✓ Token Merger参数已加载 ({len(token_merger_params)}个参数)")
        else:
            logger.warning("⚠ Checkpoint中未找到token_merger参数")

    # 加载layer_pruners参数
    layer_pruners = models["layer_pruners"]
    if "param_groups" in checkpoint and "layer_pruners" in checkpoint["param_groups"]:
        layer_pruners_params = checkpoint["param_groups"]["layer_pruners"]
        state_dict = {}
        param_names = [name for name, _ in layer_pruners.named_parameters()]
        for name, param_tensor in zip(param_names, layer_pruners_params):
            state_dict[name] = param_tensor.to(device)
        layer_pruners.load_state_dict(state_dict, strict=False)
        logger.info(f"✓ Layer Pruners参数已加载 ({len(layer_pruners_params)}个参数)")
    else:
        raise ValueError("Checkpoint中未找到layer_pruners参数!")

    # 设置为评估模式
    if token_merger is not None:
        token_merger.eval()
    layer_pruners.eval()

    checkpoint_info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "batch": checkpoint.get("batch", "unknown")
    }
    logger.info(f"Checkpoint信息: Epoch={checkpoint_info['epoch']}, Batch={checkpoint_info['batch']}")

    return checkpoint_info


def run_evaluation(config: Dict, models: Dict, backbone, dataset_bundle: Dict, device: torch.device):
    """运行评估，直接调用eval_step"""
    logger = config["logger"]
    test_dataset = dataset_bundle["splits"]["test"]

    if test_dataset is None or len(test_dataset) == 0:
        logger.error("测试集不存在或为空!")
        return {}

    logger.info(f"测试集大小: {len(test_dataset)}")

    # 准备info字典
    info = {
        "config": config,
        "models": {
            "backbone": backbone,
            "token_merger": models["token_merger"],
            "layer_pruners": models["layer_pruners"]
        },
        "persistent_state": {}
    }

    # 获取batch size
    batch_size = config.get("trainer_settings", {}).get("dl_settings", {}).get("batch_size", 1)

    # 累积评估结果
    all_metrics = {}
    num_batches = 0

    logger.info(f"开始评估 (batch_size={batch_size})...")

    # 分批评估，直接调用eval_step
    print_interval = 100  # 每100个样本打印一次
    with torch.no_grad():
        for i in tqdm(range(0, len(test_dataset), batch_size), desc="Evaluating"):
            batch = test_dataset[i:min(i+batch_size, len(test_dataset))]
            batch_metrics = eval_step(batch, device, info)

            for key, value in batch_metrics.items():
                if key in all_metrics:
                    all_metrics[key] += value
                else:
                    all_metrics[key] = value

            num_batches += 1

            # 每隔print_interval个样本打印一次当前统计
            current_samples = i + len(batch)
            if current_samples % print_interval == 0 or current_samples == len(test_dataset):
                # 计算当前平均值
                current_avg = {k: v / num_batches for k, v in all_metrics.items()}

                # 打印关键指标
                logger.info(f"[样本 {current_samples}/{len(test_dataset)}]")

                if "accuracy_baseline" in current_avg:
                    logger.info(f"  accuracy_baseline: {current_avg['accuracy_baseline']:.4f}")
                if "accuracy_hard" in current_avg:
                    logger.info(f"  accuracy_hard: {current_avg['accuracy_hard']:.4f}")
                if "hard_avg_keep_ratio" in current_avg:
                    logger.info(f"  hard_avg_keep_ratio: {current_avg['hard_avg_keep_ratio']:.4f}")
                if "hard_avg_tokens" in current_avg:
                    logger.info(f"  hard_avg_tokens: {current_avg['hard_avg_tokens']:.2f}")

    # 计算最终平均值
    if num_batches > 0:
        for key in all_metrics:
            all_metrics[key] /= num_batches

    return all_metrics


def main():
    # 加载配置（直接使用override_dict参数）
    config = load_config(
        override_file=CONFIG_FILE,
        override_dict=CONFIG_OVERRIDES if CONFIG_OVERRIDES else None
    )
    logger = config["logger"]
    device = torch.device(config["global_settings"]["device"])

    logger.info("=" * 60)
    logger.info("直接评估Checkpoint")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"Config: {CONFIG_FILE}")
    logger.info(f"Device: {device}")

    # 1. 使用preload_fn加载backbone和dataset
    logger.info("\n" + "=" * 60)
    logger.info("预加载资源（使用main.py的preload_fn）")
    logger.info("=" * 60)
    cache = preload_fn(config)
    backbone = cache["backbone"]
    dataset_bundle = cache["dataset_bundle"]

    # 将dataset_bundle放入config供eval_step使用
    config["_dataset_bundle"] = dataset_bundle

    # 2. 创建模型
    logger.info("\n" + "=" * 60)
    logger.info("创建模型")
    logger.info("=" * 60)
    models = create_models_from_config(config, backbone, device)

    # 3. 加载checkpoint参数
    logger.info("\n" + "=" * 60)
    logger.info("加载Checkpoint")
    logger.info("=" * 60)
    checkpoint_info = load_checkpoint_into_models(CHECKPOINT_PATH, models, config)

    # 4. 执行评估
    logger.info("\n" + "=" * 60)
    logger.info("开始评估")
    logger.info("=" * 60)
    results = run_evaluation(config, models, backbone, dataset_bundle, device)

    # 5. 打印结果
    logger.info("\n" + "=" * 60)
    logger.info("评估结果")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: Epoch={checkpoint_info['epoch']}, Batch={checkpoint_info['batch']}")
    logger.info("")

    # 分类打印结果
    accuracy_metrics = {k: v for k, v in results.items() if k.startswith("accuracy_")}
    token_metrics = {k: v for k, v in results.items() if "token" in k or "keep_ratio" in k or "hard_" in k}
    other_metrics = {k: v for k, v in results.items() if k not in accuracy_metrics and k not in token_metrics}

    if accuracy_metrics:
        logger.info("准确率指标:")
        for key, value in sorted(accuracy_metrics.items()):
            logger.info(f"  {key}: {value:.4f}")
        logger.info("")

    if token_metrics:
        logger.info("Token统计:")
        for key, value in sorted(token_metrics.items()):
            logger.info(f"  {key}: {value:.2f}")
        logger.info("")

    if other_metrics:
        logger.info("其他指标:")
        for key, value in sorted(other_metrics.items()):
            logger.info(f"  {key}: {value:.4f}")
        logger.info("")

    # 计算准确率下降
    if "accuracy_baseline" in results and "accuracy_hard" in results:
        acc_drop = results["accuracy_baseline"] - results["accuracy_hard"]
        acc_drop_pct = acc_drop / results["accuracy_baseline"] * 100 if results["accuracy_baseline"] > 0 else 0
        logger.info(f"准确率下降: {acc_drop:.4f} ({acc_drop_pct:.2f}%)")

    logger.info("=" * 60)
    logger.info("评估完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
