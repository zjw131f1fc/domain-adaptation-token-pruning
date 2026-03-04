#!/usr/bin/env python
"""简化版 repair objective 分析脚本 - 只需要 checkpoint 参数"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="训练好的检查点路径")
    p.add_argument("--skip_first_adapter", action="store_true", help="跳过第一个 repair adapter 的加载（覆盖 deployed layer 设置）")
    return p.parse_args()


def load_model_and_config(checkpoint_path: str, device: torch.device, skip_first_adapter: bool = False):
    """加载模型和配置"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
    from engine.configs.loader import load_config

    # 固定配置路径和模型路径
    config_path = "configs/vision_token_pruning.yaml"
    model_path = "llava-hf/llava-1.5-7b-hf"

    config = load_config(override_file=config_path)
    method_cfg = config["method_settings"]

    # 检查 checkpoint 是否有 repair 权重
    meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_has_repair = ("repair_context_encoder_state_dict" in meta) and ("repair_adapter_state_dict" in meta)
    use_repair_adapter = bool(method_cfg.get("use_repair_adapter", False) and ckpt_has_repair)

    if bool(method_cfg.get("use_repair_adapter", False)) and not ckpt_has_repair:
        print("注意: 配置要求 repair adapter，但 checkpoint 没有 repair 权重；禁用 repair")

    # 获取 repair layers 并可能跳过第一个
    repair_layers_cfg = method_cfg.get("repair_layers", None)
    repair_source_layers_cfg = method_cfg.get("repair_source_layers", None)

    if skip_first_adapter and repair_layers_cfg and len(repair_layers_cfg) > 0:
        repair_layers_cfg = list(repair_layers_cfg)[1:]  # 跳过第一个
        # 同步跳过 repair_source_layers 的第一个元素
        if repair_source_layers_cfg and len(repair_source_layers_cfg) > 0:
            repair_source_layers_cfg = list(repair_source_layers_cfg)[1:]
            print(f"跳过第一个 adapter，使用 repair_layers: {repair_layers_cfg}, repair_source_layers: {repair_source_layers_cfg}")
        else:
            print(f"跳过第一个 adapter，使用 repair_layers: {repair_layers_cfg}")

    # 加载基础模型
    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)

    # 创建可剪枝模型
    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=method_cfg.get("pruning_layers", [4, 14, 24]),
        pruner_d_internal=method_cfg.get("pruner_d_internal", 512),
        pruner_n_heads=method_cfg.get("pruner_n_heads", 4),
        pruner_n_queries=method_cfg.get("pruner_n_queries", 32),
        pruner_query_dropout=0.0,
        use_adapter=method_cfg.get("use_adapter", False),
        temperature=method_cfg.get("eval_temperature", 0.1),
        dropout=0.0,
        use_gumbel_noise=False,
        pruning_threshold=method_cfg.get("eval_pruning_threshold", 0.5),
        use_question_condition=method_cfg.get("use_question_condition", False),
        use_repair_adapter=use_repair_adapter,
        repair_layers=repair_layers_cfg,  # 使用修改后的 repair_layers
        repair_source_layers=repair_source_layers_cfg,  # 使用修改后的 repair_source_layers
        repair_bottleneck_dim=method_cfg.get("repair_bottleneck_dim", 512),
        repair_dropout=0.0,
        repair_mask_encoder_type=method_cfg.get("repair_mask_encoder_type", "attention"),
        repair_use_pruned_info=method_cfg.get("repair_use_pruned_info", True),
        repair_alpha_init=method_cfg.get("repair_alpha_init", 0.1),
    )
    model.freeze_base_model()

    # 加载权重
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "pruner_state_dict" in ckpt:
        model.pruner_manager.load_state_dict(ckpt["pruner_state_dict"])
        print("已加载 pruner_state_dict")

    if model.use_repair_adapter:
        model.repair_context_encoder.load_state_dict(ckpt["repair_context_encoder_state_dict"])

        # 如果跳过第一个 adapter，需要过滤 state_dict
        if skip_first_adapter:
            original_repair_layers = list(method_cfg.get("repair_layers", []))
            if original_repair_layers:
                first_layer = original_repair_layers[0]
                adapter_state_dict = ckpt["repair_adapter_state_dict"]

                # 过滤掉第一个 adapter 的权重
                filtered_state_dict = {}
                for key, value in adapter_state_dict.items():
                    # 检查 key 是否属于第一个 adapter
                    # 格式通常是 "adapters.{layer_idx}.xxx"
                    if key.startswith(f"adapters.{first_layer}."):
                        print(f"跳过加载权重: {key}")
                        continue
                    filtered_state_dict[key] = value

                model.repair_adapter_manager.load_state_dict(filtered_state_dict, strict=False)
                print(f"已加载 repair adapter（跳过第一个 adapter layer {first_layer}）")
        else:
            model.repair_adapter_manager.load_state_dict(ckpt["repair_adapter_state_dict"])
            print("已加载 repair_context_encoder_state_dict + repair_adapter_state_dict")

    model.eval()

    # 加载 processor
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"

    return model, config, processor


def infer_num_decoder_layers(model) -> int:
    """推断解码器层数"""
    # 尝试从 config 读取
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "base_model", None), "config", None)):
        if cfg is None:
            continue
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is None:
            continue
        for attr in ("num_hidden_layers", "num_layers", "n_layer"):
            if hasattr(text_cfg, attr):
                try:
                    return int(getattr(text_cfg, attr))
                except Exception:
                    pass

    # 尝试直接访问 layers
    try:
        layers = model.base_model.language_model.model.layers
        return int(len(layers))
    except Exception:
        pass

    try:
        layers = model.base_model.model.layers
        return int(len(layers))
    except Exception:
        pass

    raise RuntimeError("无法推断解码器层数")


def load_samples(config_path: str, num_samples: int = 64):
    """加载数据样本"""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset
    from itertools import islice

    config = load_config(override_file=config_path)
    bundle = load_dataset(config)
    dataset = bundle["splits"]["test"]
    return list(islice(dataset, num_samples))


def flatten_capture_tokens(capture_entry: Dict[str, torch.Tensor]) -> torch.Tensor:
    """展平 (b,L,D) -> (N,D) 使用 mask"""
    h = capture_entry["h"]
    m = capture_entry["mask"]
    b, L, d = h.shape
    h2 = h.reshape(b * L, d)
    m2 = m.reshape(b * L).to(dtype=torch.bool)
    if int(m2.sum().item()) <= 0:
        return h2[:0]
    return h2[m2]


class StreamingMSE:
    """流式计算 token-wise MSE"""
    def __init__(self):
        self.sse = 0.0
        self.n_elem = 0

    def update(self, X: torch.Tensor, Y: torch.Tensor):
        if X is None or Y is None or X.shape != Y.shape or X.numel() <= 0:
            return
        diff = X.float() - Y.float()
        self.sse += float((diff * diff).sum().detach().cpu().item())
        self.n_elem += int(diff.numel())

    def value(self) -> float:
        return float(self.sse / self.n_elem) if self.n_elem > 0 else float("nan")


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"加载模型: {args.checkpoint}")
    if args.skip_first_adapter:
        print("模式: 跳过第一个 repair adapter")
    model, config, processor = load_model_and_config(args.checkpoint, device, args.skip_first_adapter)

    # 固定参数
    config_path = "configs/vision_token_pruning.yaml"
    num_samples = 64
    batch_size = 1
    max_length = 1024

    # 获取配置
    method_cfg = config["method_settings"]
    pruning_layers = [int(x) for x in (method_cfg.get("pruning_layers", []) or [])]
    repair_layers = [int(x) for x in (getattr(model, "repair_layers", None) or [])] if model.use_repair_adapter else []
    target_token_num = method_cfg.get("target_token_num", None)
    teacher_pruning_mode = method_cfg.get("teacher_pruning_mode", "keep_all")

    # 推断层数
    num_layers = infer_num_decoder_layers(model)
    capture_layers = list(range(num_layers))

    print(f"模型层数: {num_layers}")
    print(f"剪枝层: {pruning_layers}")
    print(f"Repair 层: {repair_layers}")
    print(f"目标 token 数: {target_token_num}")

    # 加载数据
    print(f"\n加载数据: {num_samples} 个样本")
    samples = load_samples(config_path, num_samples)

    # 初始化指标累加器
    token_mse_acc = {
        l: {"off": StreamingMSE(), "on": StreamingMSE()}
        for l in capture_layers
    }

    from engine.data_utils import preprocess_batch

    # 主循环
    print("\n开始评估...")
    for i, sample in enumerate(samples):
        batch = [sample]
        batch_prep = preprocess_batch(batch, processor, device, max_length=max_length, mode="train")

        with torch.no_grad():
            # 三次前向传播
            out_teacher = model(
                input_ids=batch_prep["inputs"]["input_ids"],
                pixel_values=batch_prep["inputs"].get("pixel_values", None),
                attention_mask=batch_prep["inputs"].get("attention_mask", None),
                vision_start=batch_prep["vision_start"],
                vision_end=batch_prep["vision_end"],
                question_starts=batch_prep["question_starts"],
                question_ends=batch_prep["question_ends"],
                answer_starts=batch_prep["answer_starts"],
                answer_ends=batch_prep["answer_ends"],
                return_pruning_info=False,
                pruning_mode=teacher_pruning_mode,
                target_token_num=target_token_num,
                apply_repair=False,
                capture_layers=capture_layers,
            )

            out_off = model(
                input_ids=batch_prep["inputs"]["input_ids"],
                pixel_values=batch_prep["inputs"].get("pixel_values", None),
                attention_mask=batch_prep["inputs"].get("attention_mask", None),
                vision_start=batch_prep["vision_start"],
                vision_end=batch_prep["vision_end"],
                question_starts=batch_prep["question_starts"],
                question_ends=batch_prep["question_ends"],
                answer_starts=batch_prep["answer_starts"],
                answer_ends=batch_prep["answer_ends"],
                return_pruning_info=False,
                pruning_mode="normal",
                target_token_num=target_token_num,
                apply_repair=False,
                capture_layers=capture_layers,
            )

            out_on = model(
                input_ids=batch_prep["inputs"]["input_ids"],
                pixel_values=batch_prep["inputs"].get("pixel_values", None),
                attention_mask=batch_prep["inputs"].get("attention_mask", None),
                vision_start=batch_prep["vision_start"],
                vision_end=batch_prep["vision_end"],
                question_starts=batch_prep["question_starts"],
                question_ends=batch_prep["question_ends"],
                answer_starts=batch_prep["answer_starts"],
                answer_ends=batch_prep["answer_ends"],
                return_pruning_info=False,
                pruning_mode="normal",
                target_token_num=target_token_num,
                apply_repair=True,
                capture_layers=capture_layers,
            )

        # 提取捕获的隐藏状态
        cap_teacher = getattr(out_teacher, "captured", None) or {}
        cap_off = (getattr(out_off, "captured_for_repair", None) or getattr(out_off, "captured", None)) or {}
        cap_on = (getattr(out_on, "captured_for_repair", None) or getattr(out_on, "captured", None)) or {}

        # 更新指标
        for layer in capture_layers:
            if layer not in cap_teacher or layer not in cap_off or layer not in cap_on:
                continue

            t_entry = cap_teacher[layer]
            off_entry = cap_off[layer]
            on_entry = cap_on[layer]

            # 使用公共 mask
            m_common = (t_entry["mask"] > 0.5) & (off_entry["mask"] > 0.5) & (on_entry["mask"] > 0.5)

            t_tok = flatten_capture_tokens({"h": t_entry["h"], "mask": m_common})
            off_tok = flatten_capture_tokens({"h": off_entry["h"], "mask": m_common})
            on_tok = flatten_capture_tokens({"h": on_entry["h"], "mask": m_common})

            # 更新 MSE
            token_mse_acc[layer]["off"].update(off_tok, t_tok)
            token_mse_acc[layer]["on"].update(on_tok, t_tok)

        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{num_samples} 个样本...")

        # 清理内存
        del out_teacher, out_off, out_on
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # 计算结果
    print("\n计算结果...")
    results = []
    for layer in capture_layers:
        mse_off = token_mse_acc[layer]["off"].value()
        mse_on = token_mse_acc[layer]["on"].value()
        gain = mse_off - mse_on

        results.append({
            "layer": layer,
            "mse_off": mse_off,
            "mse_on": mse_on,
            "gain": gain,
            "is_pruning": layer in pruning_layers,
            "is_repair": layer in repair_layers,
        })

    # 输出目录
    checkpoint_name = Path(args.checkpoint).parent.parent.name
    suffix = "_skip1st" if args.skip_first_adapter else ""
    output_dir = f"outputs/visualizations/simple_repair_{checkpoint_name}{suffix}"
    os.makedirs(output_dir, exist_ok=True)

    # 保存 CSV
    import csv
    csv_path = os.path.join(output_dir, "repair_analysis.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["layer", "mse_off", "mse_on", "gain", "is_pruning", "is_repair"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n已保存 CSV: {csv_path}")

    # 生成图表
    layers = [r["layer"] for r in results]
    mse_off = [r["mse_off"] for r in results]
    mse_on = [r["mse_on"] for r in results]
    gains = [r["gain"] for r in results]

    # 图1: MSE 对比
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, mse_off, marker="o", color="tab:red", label="OFF (no repair)")
    ax.plot(layers, mse_on, marker="o", color="tab:blue", label="ON (repair)")

    # 标记剪枝层和 repair 层
    for l in pruning_layers:
        ax.axvline(l, color="k", linestyle="--", alpha=0.2, linewidth=1.0)
    for l in repair_layers:
        ax.axvline(l, color="tab:green", linestyle="--", alpha=0.3, linewidth=1.2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("MSE to Teacher")
    ax.set_title("Repair Objective: MSE to Teacher (gen_answer tokens)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, "mse_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图表: {fig_path}")

    # 图2: Gain
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, gains, marker="o", color="tab:purple", label="Gain = OFF - ON")
    ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)

    for l in pruning_layers:
        ax.axvline(l, color="k", linestyle="--", alpha=0.2, linewidth=1.0)
    for l in repair_layers:
        ax.axvline(l, color="tab:green", linestyle="--", alpha=0.3, linewidth=1.2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Gain (higher is better)")
    ax.set_title("Repair Gain Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, "gain.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图表: {fig_path}")

    # 打印摘要
    print("\n=== 摘要 ===")
    avg_mse_off = np.mean([r["mse_off"] for r in results])
    avg_mse_on = np.mean([r["mse_on"] for r in results])
    avg_gain = np.mean([r["gain"] for r in results])

    print(f"平均 MSE (OFF): {avg_mse_off:.6f}")
    print(f"平均 MSE (ON):  {avg_mse_on:.6f}")
    print(f"平均 Gain:      {avg_gain:.6f}")

    if repair_layers:
        repair_results = [r for r in results if r["is_repair"]]
        avg_gain_repair = np.mean([r["gain"] for r in repair_results])
        print(f"\nRepair 层平均 Gain: {avg_gain_repair:.6f}")

    print(f"\n所有输出已保存到: {output_dir}")


if __name__ == "__main__":
    main()

