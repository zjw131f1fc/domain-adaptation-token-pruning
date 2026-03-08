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
    import re

    # 固定配置路径和模型路径
    config_path = "configs/vision_token_pruning.yaml"
    model_path = "llava-hf/llava-1.5-7b-hf"

    config = load_config(override_file=config_path)
    method_cfg = config["method_settings"]

    # 检查 checkpoint 是否有 repair 权重
    meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # --- Infer pruner hyper-params from checkpoint to avoid config mismatch ---
    # Some checkpoints were trained with different `pruner_n_queries` / `use_question_condition`
    # than the current config; loading with mismatched shapes will fail.
    pruner_sd = meta.get("pruner_state_dict", {}) or {}
    inferred_pruning_layers = None
    inferred_n_queries = None
    inferred_d_internal = None
    inferred_use_question_condition = None

    if isinstance(pruner_sd, dict) and pruner_sd:
        pruner_indices = set()
        for k in pruner_sd.keys():
            m = re.match(r"pruners\.(\d+)\.", str(k))
            if m:
                pruner_indices.add(int(m.group(1)))
        if pruner_indices:
            inferred_pruning_layers = sorted(pruner_indices)

        for k, v in pruner_sd.items():
            if str(k).endswith("pruning_queries") and hasattr(v, "shape") and len(v.shape) == 3:
                inferred_n_queries = int(v.shape[1])
                inferred_d_internal = int(v.shape[2])
                break

        if inferred_d_internal is None:
            for k, v in pruner_sd.items():
                if str(k).endswith("vision_proj.weight") and hasattr(v, "shape") and len(v.shape) == 2:
                    inferred_d_internal = int(v.shape[0])
                    break

        inferred_use_question_condition = any("question_proj" in str(k) for k in pruner_sd.keys())

    cfg_pruning_layers = list(method_cfg.get("pruning_layers", [4, 14, 24]))
    cfg_pruner_d_internal = int(method_cfg.get("pruner_d_internal", 512))
    cfg_pruner_n_queries = int(method_cfg.get("pruner_n_queries", 32))
    cfg_use_question_condition = bool(method_cfg.get("use_question_condition", False))

    pruning_layers = inferred_pruning_layers or cfg_pruning_layers
    pruner_d_internal = int(inferred_d_internal or cfg_pruner_d_internal)
    pruner_n_queries = int(inferred_n_queries or cfg_pruner_n_queries)
    use_question_condition = bool(
        inferred_use_question_condition if inferred_use_question_condition is not None else cfg_use_question_condition
    )

    if pruning_layers != cfg_pruning_layers:
        print(f"[simple_repair_analysis] Override pruning_layers: {cfg_pruning_layers} -> {pruning_layers} (from ckpt)")
    if pruner_d_internal != cfg_pruner_d_internal:
        print(
            f"[simple_repair_analysis] Override pruner_d_internal: {cfg_pruner_d_internal} -> {pruner_d_internal} (from ckpt)"
        )
    if pruner_n_queries != cfg_pruner_n_queries:
        print(
            f"[simple_repair_analysis] Override pruner_n_queries: {cfg_pruner_n_queries} -> {pruner_n_queries} (from ckpt)"
        )
    if use_question_condition != cfg_use_question_condition:
        print(
            f"[simple_repair_analysis] Override use_question_condition: {cfg_use_question_condition} -> {use_question_condition} (from ckpt)"
        )
    ckpt_has_repair = ("repair_context_encoder_state_dict" in meta) and ("repair_adapter_state_dict" in meta)
    use_repair_adapter = bool(method_cfg.get("use_repair_adapter", False) and ckpt_has_repair)

    if bool(method_cfg.get("use_repair_adapter", False)) and not ckpt_has_repair:
        print("注意: 配置要求 repair adapter，但 checkpoint 没有 repair 权重；禁用 repair")

    # 获取 repair layers 并可能跳过第一个 / 手动 drop 某些层
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

    # Hard-drop adapters that are known to be harmful / ablation-only.
    # User request: always ignore L12 and L13 (do NOT introduce extra CLI flags).
    hard_drop_set = {12, 13}
    if repair_layers_cfg:
        original_layers = list(repair_layers_cfg)
        keep_mask = [int(l) not in hard_drop_set for l in original_layers]
        if not all(keep_mask):
            dropped = sorted({int(l) for l, keep in zip(original_layers, keep_mask) if not keep})
            repair_layers_cfg = [l for l, keep in zip(original_layers, keep_mask) if keep]
            if repair_source_layers_cfg is not None:
                repair_source_layers_cfg = [l for l, keep in zip(list(repair_source_layers_cfg), keep_mask) if keep]
            print(
                f"[simple_repair_analysis] Hard-drop repair adapters at layers {dropped}; "
                f"now repair_layers={repair_layers_cfg}"
            )

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
        pruning_layers=pruning_layers,
        pruner_d_internal=pruner_d_internal,
        pruner_n_heads=method_cfg.get("pruner_n_heads", 4),
        pruner_n_queries=pruner_n_queries,
        pruner_query_dropout=0.0,
        use_adapter=method_cfg.get("use_adapter", False),
        temperature=method_cfg.get("eval_temperature", 0.1),
        dropout=0.0,
        use_gumbel_noise=False,
        pruning_threshold=method_cfg.get("eval_pruning_threshold", 0.5),
        use_question_condition=use_question_condition,
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
        # Context encoder is shared; keep strict=False for robustness across ablations.
        model.repair_context_encoder.load_state_dict(ckpt["repair_context_encoder_state_dict"], strict=False)

        # Adapter manager is layer-indexed. Some checkpoints may contain extra adapters
        # (e.g., an ablation added adapter@L12). Filter to only the layers enabled in
        # the current model to avoid "Unexpected key(s)" errors.
        adapter_state_dict = ckpt["repair_adapter_state_dict"]
        allowed_layers = set(int(x) for x in (getattr(model, "repair_layers", None) or []))

        if not allowed_layers:
            print("[simple_repair_analysis] Warning: use_repair_adapter=True but no repair_layers configured; skip loading adapters.")
        else:
            filtered_state_dict = {}
            dropped_layers = set()
            for key, value in adapter_state_dict.items():
                m = re.match(r"adapters\.(\d+)\.", str(key))
                if m:
                    layer_idx = int(m.group(1))
                    if layer_idx not in allowed_layers:
                        dropped_layers.add(layer_idx)
                        continue
                filtered_state_dict[key] = value

            if dropped_layers:
                print(f"[simple_repair_analysis] Drop adapter weights for layers not in repair_layers={sorted(allowed_layers)}: {sorted(dropped_layers)}")

            incompatible = model.repair_adapter_manager.load_state_dict(filtered_state_dict, strict=False)
            if getattr(incompatible, "unexpected_keys", None):
                print(f"[simple_repair_analysis] Unexpected adapter keys (ignored): {len(incompatible.unexpected_keys)}")
            if getattr(incompatible, "missing_keys", None):
                # Missing keys can happen if some adapters are disabled by config.
                print(f"[simple_repair_analysis] Missing adapter keys (ignored): {len(incompatible.missing_keys)}")

            print("已加载 repair_context_encoder_state_dict + repair_adapter_state_dict (filtered)")

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


class StreamingDistributionAlignment:
    """流式计算分布对齐指标（与训练时相同）"""
    def __init__(self):
        self.sum_s = None  # student 的和
        self.sum_s2 = None  # student 的平方和
        self.sum_t = None  # teacher 的和
        self.sum_t2 = None  # teacher 的平方和
        self.count = 0  # token 数量

    def update(self, student: torch.Tensor, teacher: torch.Tensor):
        """更新统计量
        Args:
            student: (N, D) 学生表示
            teacher: (N, D) 教师表示
        """
        if student is None or teacher is None or student.shape != teacher.shape or student.numel() <= 0:
            return

        N, D = student.shape
        s = student.float()
        t = teacher.float()

        # 累积统计量
        s_sum = s.sum(dim=0)  # (D,)
        s2_sum = (s * s).sum(dim=0)  # (D,)
        t_sum = t.sum(dim=0)  # (D,)
        t2_sum = (t * t).sum(dim=0)  # (D,)

        if self.sum_s is None:
            self.sum_s = s_sum.cpu().numpy()
            self.sum_s2 = s2_sum.cpu().numpy()
            self.sum_t = t_sum.cpu().numpy()
            self.sum_t2 = t2_sum.cpu().numpy()
        else:
            self.sum_s += s_sum.cpu().numpy()
            self.sum_s2 += s2_sum.cpu().numpy()
            self.sum_t += t_sum.cpu().numpy()
            self.sum_t2 += t2_sum.cpu().numpy()

        self.count += N

    def compute_alignment(self, var_weight: float = 1.0) -> Dict[str, float]:
        """计算分布对齐指标
        Returns:
            dict with keys:
              - mean_mse: MSE between per-dim means
              - var_mse:  MSE between per-dim variances (2nd central moments)
              - std_mse:  MSE between per-dim stds (sqrt variances)
              - total:    mean_mse + var_weight * var_mse (training objective)
              - w2_sq:    diag-Gaussian W2^2 surrogate (mean_mse + std_mse)
        """
        if self.count <= 0 or self.sum_s is None:
            return {
                "mean_mse": float("nan"),
                "var_mse": float("nan"),
                "std_mse": float("nan"),
                "total": float("nan"),
                "w2_sq": float("nan"),
            }

        # 计算均值
        inv = 1.0 / float(self.count)
        mean_s = self.sum_s * inv
        mean_t = self.sum_t * inv

        # 计算方差 (unbiased=False, 与训练一致)
        var_s = self.sum_s2 * inv - mean_s * mean_s
        var_t = self.sum_t2 * inv - mean_t * mean_t

        # 数值稳定性
        var_s = np.maximum(var_s, 0.0)
        var_t = np.maximum(var_t, 0.0)

        # 计算 MSE
        mean_mse = float(np.mean((mean_s - mean_t) ** 2))
        var_mse = float(np.mean((var_s - var_t) ** 2))
        total = mean_mse + var_weight * var_mse

        # Diagonal-Gaussian W2^2 uses std (sqrt variance), not variance.
        std_s = np.sqrt(var_s)
        std_t = np.sqrt(var_t)
        std_mse = float(np.mean((std_s - std_t) ** 2))
        w2_sq = mean_mse + std_mse

        return {
            "mean_mse": mean_mse,
            "var_mse": var_mse,
            "std_mse": std_mse,
            "total": total,
            "w2_sq": w2_sq,
        }


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
    var_weight = float(method_cfg.get("repair_var_weight", 1.0))

    # 推断层数
    num_layers = infer_num_decoder_layers(model)
    capture_layers = list(range(num_layers))

    print(f"模型层数: {num_layers}")
    print(f"剪枝层: {pruning_layers}")
    print(f"Repair 层: {repair_layers}")
    print(f"目标 token 数: {target_token_num}")
    print(f"Var weight: {var_weight}")

    # 加载数据
    print(f"\n加载数据: {num_samples} 个样本")
    samples = load_samples(config_path, num_samples)

    # 初始化指标累加器（使用分布对齐指标）
    alignment_acc = {
        l: {"off": StreamingDistributionAlignment(), "on": StreamingDistributionAlignment()}
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

            # 更新分布对齐指标
            alignment_acc[layer]["off"].update(off_tok, t_tok)
            alignment_acc[layer]["on"].update(on_tok, t_tok)

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
        metrics_off = alignment_acc[layer]["off"].compute_alignment(var_weight)
        metrics_on = alignment_acc[layer]["on"].compute_alignment(var_weight)

        total_off = metrics_off["total"]
        total_on = metrics_on["total"]
        gain = total_off - total_on

        results.append({
            "layer": layer,
            "total_off": total_off,
            "mean_mse_off": metrics_off["mean_mse"],
            "var_mse_off": metrics_off["var_mse"],
            "std_mse_off": metrics_off["std_mse"],
            "w2_sq_off": metrics_off["w2_sq"],
            "total_on": total_on,
            "mean_mse_on": metrics_on["mean_mse"],
            "var_mse_on": metrics_on["var_mse"],
            "std_mse_on": metrics_on["std_mse"],
            "w2_sq_on": metrics_on["w2_sq"],
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
        writer = csv.DictWriter(f, fieldnames=[
            "layer",
            "total_off", "mean_mse_off", "var_mse_off", "std_mse_off", "w2_sq_off",
            "total_on", "mean_mse_on", "var_mse_on", "std_mse_on", "w2_sq_on",
            "gain", "is_pruning", "is_repair"
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n已保存 CSV: {csv_path}")

    # 生成图表（排除 Layer 31 - 解码层噪声太大）
    results_for_plot = [r for r in results if r["layer"] != 31]

    layers = [r["layer"] for r in results_for_plot]
    total_off = [r["total_off"] for r in results_for_plot]
    total_on = [r["total_on"] for r in results_for_plot]
    gains = [r["gain"] for r in results_for_plot]

    # 图1: MSE 对比
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(layers, total_off, marker="o", color="tab:red", label="OFF (no repair)")
    ax.plot(layers, total_on, marker="o", color="tab:blue", label="ON (repair)")

    # 标记剪枝层和 repair 层
    for l in pruning_layers:
        ax.axvline(l, color="k", linestyle="--", alpha=0.2, linewidth=1.0)
    for l in repair_layers:
        ax.axvline(l, color="tab:green", linestyle="--", alpha=0.3, linewidth=1.2)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Distribution Alignment Loss (mean_mse + var_mse)")
    ax.set_title("Repair Objective: Distribution Alignment (gen_answer tokens)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(output_dir, "mse_comparison.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存图表: {fig_path}")

    # 图2: Gain（改进版 - 多种可视化）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 2.1: 绝对 Gain
    ax = axes[0, 0]
    ax.plot(layers, gains, marker="o", color="tab:purple", label="Gain = OFF - ON")
    ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)
    for l in pruning_layers:
        ax.axvline(l, color="k", linestyle="--", alpha=0.2, linewidth=1.0)
    for l in repair_layers:
        ax.axvline(l, color="tab:green", linestyle="--", alpha=0.3, linewidth=1.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gain (MSE)")
    ax.set_title("Absolute Gain (OFF - ON)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2.2: 相对 Gain (百分比)
    ax = axes[0, 1]
    relative_gains = [(g / off * 100 if off > 1e-10 else 0) for g, off in zip(gains, total_off)]
    ax.plot(layers, relative_gains, marker="o", color="tab:orange", label="Relative Gain %")
    ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)
    for l in pruning_layers:
        ax.axvline(l, color="k", linestyle="--", alpha=0.2, linewidth=1.0)
    for l in repair_layers:
        ax.axvline(l, color="tab:green", linestyle="--", alpha=0.3, linewidth=1.2)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Relative Gain (%)")
    ax.set_title("Relative Gain: (OFF - ON) / OFF × 100%")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2.3: 只显示有显著差异的层（gain != 0）
    ax = axes[1, 0]
    nonzero_indices = [i for i, g in enumerate(gains) if abs(g) > 1e-10]
    if nonzero_indices:
        nonzero_layers = [layers[i] for i in nonzero_indices]
        nonzero_gains = [gains[i] for i in nonzero_indices]
        ax.bar(nonzero_layers, nonzero_gains, color="tab:purple", alpha=0.7)
        ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Gain (MSE)")
        ax.set_title(f"Non-zero Gains Only ({len(nonzero_indices)} layers)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No non-zero gains", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Non-zero Gains Only (none found)")

    # 2.4: 对数尺度的 MSE 对比（更容易看出差异）
    ax = axes[1, 1]
    # 过滤掉 0 值
    valid_indices = [i for i, (off, on) in enumerate(zip(total_off, total_on)) if off > 1e-10 and on > 1e-10]
    if valid_indices:
        valid_layers = [layers[i] for i in valid_indices]
        valid_off = [total_off[i] for i in valid_indices]
        valid_on = [total_on[i] for i in valid_indices]
        ax.plot(valid_layers, valid_off, marker="o", color="tab:red", label="OFF", alpha=0.7)
        ax.plot(valid_layers, valid_on, marker="o", color="tab:blue", label="ON", alpha=0.7)
        ax.set_yscale("log")
        for l in pruning_layers:
            ax.axvline(l, color="k", linestyle="--", alpha=0.2, linewidth=1.0)
        for l in repair_layers:
            ax.axvline(l, color="tab:green", linestyle="--", alpha=0.3, linewidth=1.2)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Distribution Alignment Loss (log scale)")
        ax.set_title("Distribution Alignment: OFF vs ON (Log Scale)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "gain_detailed.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存详细增益图表: {fig_path}")

    # 原来的简单 gain 图也保留
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

    # 打印摘要（排除 Layer 31 - 解码层噪声太大）
    print("\n=== 摘要 ===")

    # 排除 Layer 31
    results_filtered = [r for r in results if r["layer"] != 31]

    avg_total_off = np.mean([r["total_off"] for r in results_filtered])
    avg_total_on = np.mean([r["total_on"] for r in results_filtered])
    avg_gain = np.mean([r["gain"] for r in results_filtered])

    print(f"平均 Total (OFF): {avg_total_off:.6f} (排除 Layer 31)")
    print(f"平均 Total (ON):  {avg_total_on:.6f} (排除 Layer 31)")
    print(f"平均 Gain:        {avg_gain:.6f} (排除 Layer 31)")

    # 统计非零 gain 的层（排除 Layer 31）
    nonzero_gains = [r for r in results_filtered if abs(r["gain"]) > 1e-10]
    if nonzero_gains:
        print(f"\n非零 Gain 的层数: {len(nonzero_gains)} (排除 Layer 31)")
        print("非零 Gain 的层:")
        for r in nonzero_gains:
            print(f"  Layer {r['layer']}: gain={r['gain']:.6e} ({r['gain']/r['total_off']*100:.2f}%)")

    if repair_layers:
        repair_results = [r for r in results_filtered if r["is_repair"]]
        if repair_results:
            avg_gain_repair = np.mean([r["gain"] for r in repair_results])
            print(f"\nRepair 层平均 Gain: {avg_gain_repair:.6f} (排除 Layer 31)")
        else:
            print("\n注意: 没有 repair 层有数据（可能都被跳过了）")

    print(f"\n所有输出已保存到: {output_dir}")


if __name__ == "__main__":
    main()
