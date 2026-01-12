"""Loss 收敛性测试脚本

用法: python run_loss_test.py <test_name>

test_name 可选值:
  - task_loss
  - adv_loss
  - sparsity_loss
  - token_count_loss
  - binarization_loss
  - disc_loss
  - all_losses
"""

import sys
import os
import yaml
import torch
from copy import deepcopy
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def merge_dict(base: dict, override: dict) -> dict:
    """递归合并字典"""
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def analyze_and_print_results(test_name: str, loss_history: dict, num_steps: int):
    """分析并打印 loss 收敛结果"""

    print("\n" + "=" * 70)
    print(f"  {test_name} 收敛分析")
    print("=" * 70)

    results = {}

    for key, values in loss_history.items():
        if len(values) < 10:
            continue

        # 计算前 20% 和后 20% 的平均值
        n = len(values)
        early_n = max(1, n // 5)
        late_n = max(1, n // 5)

        early_avg = sum(values[:early_n]) / early_n
        late_avg = sum(values[-late_n:]) / late_n

        # 计算变化率
        if abs(early_avg) > 1e-8:
            change_rate = (late_avg - early_avg) / abs(early_avg) * 100
        else:
            change_rate = 0

        results[key] = {
            "early_avg": early_avg,
            "late_avg": late_avg,
            "change_rate": change_rate,
            "converged": change_rate < -5  # 下降超过 5% 认为收敛
        }

    # 打印结果表格
    print(f"\n{'Loss 名称':<25} {'初始值':<12} {'最终值':<12} {'变化率':<12} {'状态':<10}")
    print("-" * 70)

    for key, r in sorted(results.items()):
        if r["change_rate"] < -5:
            status = "✓ 下降"
        elif r["change_rate"] > 5:
            status = "✗ 上升"
        else:
            status = "- 持平"

        print(f"{key:<25} {r['early_avg']:<12.4f} {r['late_avg']:<12.4f} {r['change_rate']:>+8.1f}%    {status}")

    print("-" * 70)

    # 总结
    main_losses = ["task_loss", "adv_loss", "sparsity_loss", "token_count_loss",
                   "binarization_loss", "disc_real_loss", "disc_fake_loss", "layer_pruners_total"]

    tested_main = [k for k in main_losses if k in results]
    converged_main = [k for k in tested_main if results[k]["converged"]]

    print(f"\n主要 Loss 收敛情况: {len(converged_main)}/{len(tested_main)}")

    if test_name != "all_losses":
        # 单个 loss 测试，检查对应的 loss
        target_key = test_name.replace("_loss", "") + "_loss" if not test_name.endswith("_loss") else test_name

        # 特殊处理
        if test_name == "disc_loss":
            target_keys = ["disc_real_loss", "disc_fake_loss"]
        elif test_name == "sparsity_loss":
            target_keys = ["sparsity_loss"]
        else:
            target_keys = [target_key, "layer_pruners_total"]

        found = False
        for tk in target_keys:
            if tk in results:
                found = True
                r = results[tk]
                if r["converged"]:
                    print(f"\n>>> 结论: {test_name} ✓ 可以收敛 (下降 {-r['change_rate']:.1f}%)")
                elif r["change_rate"] > 5:
                    print(f"\n>>> 结论: {test_name} ✗ 不收敛，反而上升 ({r['change_rate']:+.1f}%)")
                else:
                    print(f"\n>>> 结论: {test_name} ~ 基本持平 ({r['change_rate']:+.1f}%)")
                break

        if not found:
            print(f"\n>>> 结论: 未找到 {test_name} 对应的 loss 记录")
    else:
        # 所有 loss 一起测试
        all_converged = all(results[k]["converged"] for k in tested_main if k in results)
        if all_converged:
            print(f"\n>>> 结论: 所有 Loss ✓ 均可收敛")
        else:
            not_converged = [k for k in tested_main if k in results and not results[k]["converged"]]
            print(f"\n>>> 结论: 部分 Loss 未收敛: {', '.join(not_converged)}")

    print("=" * 70 + "\n")

    return results


def run_test(test_name: str, config: dict, num_steps: int = 200):
    """运行单个 loss 测试"""

    from engine.datas.loader import load_dataset
    from engine.backbones.loader import load_backbone
    from method import LayerSpecificPruner, Discriminator, train_step

    device = config.get("global_settings", {}).get("device", "cuda")

    print(f"\n{'#' * 70}")
    print(f"#  测试: {test_name}")
    print(f"#  步数: {num_steps}")
    print(f"{'#' * 70}\n")

    # 加载 backbone
    print("[1/4] 加载 Backbone...")
    backbone = load_backbone(config)
    if hasattr(backbone, "model"):
        for param in backbone.model.parameters():
            param.requires_grad = False

    # 加载数据集
    print("[2/4] 加载数据集...")
    dataset_bundle = load_dataset(config)
    train_loader = dataset_bundle["train_loader"]

    # 创建模块
    print("[3/4] 创建模块...")
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

    discriminator = Discriminator(
        d_model=config["backbone_settings"]["mllm_settings"]["hidden_dim"],
        num_layers=config["method_settings"]["disc_num_layers"],
        d_d=config["method_settings"]["disc_d_d"],
        dropout=config["method_settings"]["disc_dropout"],
        use_layer_norm=True,
        use_spectral_norm=config["method_settings"]["disc_use_spectral_norm"]
    ).to(device=device)

    # 创建优化器
    pruner_lr = config['trainer_settings']['dl_settings']['optimizers']['layer_pruners']['lr']
    disc_lr = config['trainer_settings']['dl_settings']['optimizers']['discriminator']['lr']

    pruner_optimizer = torch.optim.Adam(layer_pruners.parameters(), lr=pruner_lr)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=disc_lr)

    # 训练循环
    print(f"[4/4] 开始训练 ({num_steps} steps)...")
    print("-" * 50)

    loss_history = defaultdict(list)
    data_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        try:
            losses, stats = train_step(
                batch=batch,
                backbone=backbone,
                token_merger=None,
                layer_pruners=layer_pruners,
                discriminator=discriminator,
                pruner_optimizer=pruner_optimizer,
                disc_optimizer=disc_optimizer,
                config=config,
                current_step=step,
                total_steps=num_steps * 2,
                device=device
            )

            # 记录 losses
            for k, v in losses.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                loss_history[k].append(val)

            # 记录关键 stats
            for k in ["avg_kept_ratio", "disc_accuracy"]:
                if k in stats:
                    loss_history[k].append(float(stats[k]))

            # 打印进度
            if (step + 1) % 50 == 0:
                total_loss = losses.get("layer_pruners_total", torch.tensor(0))
                if isinstance(total_loss, torch.Tensor):
                    total_loss = total_loss.item()
                disc_loss = losses.get("disc_total", torch.tensor(0))
                if isinstance(disc_loss, torch.Tensor):
                    disc_loss = disc_loss.item()
                print(f"  Step {step+1:4d}/{num_steps}: pruner_loss={total_loss:.4f}, disc_loss={disc_loss:.4f}")

        except Exception as e:
            print(f"  Step {step+1} 出错: {e}")
            import traceback
            traceback.print_exc()
            break

    print("-" * 50)

    # 分析结果
    results = analyze_and_print_results(test_name, loss_history, num_steps)

    # 清理
    del backbone, layer_pruners, discriminator
    torch.cuda.empty_cache()

    return results


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    test_name = sys.argv[1]
    num_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200

    # 配置文件路径
    base_config_path = "configs/loss_tests/base_test.yaml"
    test_config_path = f"configs/loss_tests/test_{test_name}.yaml"

    if not os.path.exists(base_config_path):
        print(f"错误: 基础配置文件不存在: {base_config_path}")
        sys.exit(1)

    if not os.path.exists(test_config_path):
        print(f"错误: 测试配置文件不存在: {test_config_path}")
        print(f"可用: task_loss, adv_loss, sparsity_loss, token_count_loss, binarization_loss, disc_loss, all_losses")
        sys.exit(1)

    # 加载并合并配置
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    with open(test_config_path, 'r') as f:
        test_config = yaml.safe_load(f)

    config = merge_dict(base_config, test_config)

    # 添加必要的默认值
    config.setdefault("logger", type('Logger', (), {
        'info': lambda self, msg: print(f"[INFO] {msg}"),
        'warning': lambda self, msg: print(f"[WARN] {msg}"),
        'error': lambda self, msg: print(f"[ERROR] {msg}"),
        'debug': lambda self, msg: None
    })())

    # 设置 hidden_dim 和 vision_dim（如果没有）
    config["backbone_settings"]["mllm_settings"].setdefault("hidden_dim", 4096)
    config["backbone_settings"]["mllm_settings"].setdefault("vision_dim", 1024)

    # 运行测试
    run_test(test_name, config, num_steps)


if __name__ == "__main__":
    main()
