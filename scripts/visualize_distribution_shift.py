#!/usr/bin/env python
"""可视化 h_real, h_fake, h_corrected 的分布偏移

用法:
    python scripts/visualize_distribution_shift.py --checkpoint <path>

分析内容:
1. Global Mean Distance - 中心点距离
2. Per-Sample Distance - 样本级距离
3. Distribution Shape - 方差、标准差
4. MMD - Maximum Mean Discrepancy
5. PCA / t-SNE / 1D projection 可视化
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
from pathlib import Path
from collections import defaultdict
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="configs/vision_token_pruning.yaml", help="Config file path")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Output directory")
    parser.add_argument("--no_layernorm", action="store_true", help="Don't apply next layer's LayerNorm (show raw adapter output)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (subsampling, t-SNE, etc.)")
    parser.add_argument("--max_vis_tokens", type=int, default=1000, help="Max tokens used for PCA/t-SNE per group")
    parser.add_argument("--tsne_pca_dim", type=int, default=50, help="PCA reduce dim before t-SNE (0 disables)")
    parser.add_argument("--no_tsne_standardize", action="store_true", help="Disable StandardScaler before t-SNE")
    parser.add_argument("--tsne_metric", type=str, default="euclidean", choices=["euclidean", "cosine"], help="t-SNE metric")
    parser.add_argument("--tag", type=str, default="", help="Optional run tag (used as output subdir name)")
    parser.add_argument("--flat_output", action="store_true", help="Write directly into output_dir (may overwrite files)")
    return parser.parse_args()


def load_model_and_processor(checkpoint_path, config_path, device):
    """加载模型和 checkpoint"""
    from transformers import LlavaForConditionalGeneration, AutoProcessor
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
    from engine.configs.loader import load_config

    # 加载配置
    config = load_config(override_file=config_path)
    method_cfg = config['method_settings']

    # 先在 CPU 上读取 checkpoint，避免因为 config 打开 use_repair_adapter 但 ckpt 里没有权重，
    # 导致随机初始化的 adapter 参与 forward，污染 “h_corrected”。
    print(f"Inspecting checkpoint keys (CPU) from {checkpoint_path}...")
    checkpoint_cpu = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    has_repair_adapter = isinstance(checkpoint_cpu, dict) and ("repair_adapter_state_dict" in checkpoint_cpu)
    has_repair_ctx = isinstance(checkpoint_cpu, dict) and ("repair_context_encoder_state_dict" in checkpoint_cpu)
    if method_cfg.get("use_repair_adapter", False) and (not has_repair_adapter):
        print(
            "[Auto] Checkpoint has no repair_adapter_state_dict; disabling use_repair_adapter for visualization "
            "to avoid random adapter affecting h_corrected."
        )
        method_cfg["use_repair_adapter"] = False
    if method_cfg.get("use_repair_adapter", False) and has_repair_adapter and (not has_repair_ctx):
        print(
            "[Warning] Checkpoint has repair_adapter_state_dict but no repair_context_encoder_state_dict; "
            "context encoder will stay randomly initialized and may hurt h_corrected."
        )

    model_path = "llava-hf/llava-1.5-7b-hf"
    print(f"Loading base model from {model_path}...")

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "right"

    # 创建可剪枝模型（从配置读取参数）
    # 新版架构：pruning_layers 做剪枝，repair_layers 做修复（Delayed Repair）
    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=method_cfg.get('pruning_layers', [4, 14, 24]),
        pruner_d_internal=method_cfg.get('pruner_d_internal', 512),
        pruner_n_heads=method_cfg.get('pruner_n_heads', 4),
        pruner_n_queries=method_cfg.get('pruner_n_queries', 32),
        pruner_query_dropout=0.0,  # 分析时关闭 dropout
        use_adapter=method_cfg.get('use_adapter', False),  # 旧版 adapter 不再使用
        temperature=method_cfg.get('eval_temperature', 0.1),
        dropout=0.0,  # 分析时关闭 dropout
        use_gumbel_noise=False,  # 分析时关闭 Gumbel noise
        pruning_threshold=method_cfg.get('eval_pruning_threshold', 0.5),
        use_question_condition=method_cfg.get('use_question_condition', False),
        # Delayed Repair Adapter 参数
        use_repair_adapter=method_cfg.get('use_repair_adapter', False),
        repair_layers=method_cfg.get('repair_layers', None),
        repair_source_layers=method_cfg.get('repair_source_layers', None),
        repair_bottleneck_dim=method_cfg.get('repair_bottleneck_dim', 512),
        repair_dropout=0.0,  # 分析时关闭 dropout
        repair_mask_encoder_type=method_cfg.get('repair_mask_encoder_type', 'attention'),
        repair_use_pruned_info=method_cfg.get('repair_use_pruned_info', True),
        repair_alpha_init=method_cfg.get('repair_alpha_init', 0.1),
        # Strong delayed-repair options (backward compatible: defaults keep old behavior)
        repair_adapter_type=method_cfg.get('repair_adapter_type', 'lightweight'),
        repair_context_num_tokens=int(method_cfg.get('repair_context_num_tokens', 0)),
        repair_context_dropout=float(method_cfg.get('repair_context_dropout', 0.0)),
        repair_context_use_q2v_relevance=bool(method_cfg.get('repair_context_use_q2v_relevance', False)),
        repair_apply_only_gen_tokens=bool(method_cfg.get('repair_apply_only_gen_tokens', True)),
    )

    model.freeze_base_model()

    # 加载 checkpoint（直接复用 CPU 读取的内容；state_dict 会在 load_state_dict 时搬到对应 device）
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = checkpoint_cpu

    if 'pruner_state_dict' in checkpoint:
        model.pruner_manager.load_state_dict(checkpoint['pruner_state_dict'])
        print("  Loaded pruner_state_dict")

    # 新版：加载 repair adapter
    if 'repair_context_encoder_state_dict' in checkpoint and model.use_repair_adapter:
        model.repair_context_encoder.load_state_dict(checkpoint['repair_context_encoder_state_dict'])
        print("  Loaded repair_context_encoder_state_dict")
    if 'repair_adapter_state_dict' in checkpoint and model.use_repair_adapter:
        model.repair_adapter_manager.load_state_dict(checkpoint['repair_adapter_state_dict'])
        print("  Loaded repair_adapter_state_dict")

    model.eval()
    print("Model loaded.")

    return model, processor


def load_samples(num_samples, config_path):
    """加载样本"""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset
    from itertools import islice

    config = load_config(override_file=config_path)

    data_bundle = load_dataset(config)
    test_dataset = data_bundle['splits']['train']

    # 使用 islice 避免加载整个数据集
    return list(islice(test_dataset, num_samples))


def preprocess_sample(sample, processor, device):
    """预处理样本"""
    image = sample['image']
    question = sample['question']
    answer = sample['answer']

    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer.capitalize()}"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(device)

    input_ids = inputs['input_ids']
    batch_size, seq_len = input_ids.shape

    # 找 vision tokens 位置
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<image>')
    n_vision_tokens = 576

    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(image_positions) > 0:
        vision_start = image_positions[0].item()
        vision_end = vision_start + n_vision_tokens
    else:
        vision_start = 1
        vision_end = vision_start + n_vision_tokens

    # 找 ASSISTANT: 位置
    assistant_ids = processor.tokenizer.encode("\nASSISTANT:", add_special_tokens=False)
    if assistant_ids[0] == 29871:
        assistant_ids = assistant_ids[1:]

    ids = input_ids[0].tolist()
    assistant_pos = None
    for j in range(len(ids) - len(assistant_ids) + 1):
        if ids[j:j+len(assistant_ids)] == assistant_ids:
            assistant_pos = j + len(assistant_ids)
            break

    if assistant_pos is None:
        return None

    question_starts = [vision_end]
    question_ends = [assistant_pos]
    answer_starts = [assistant_pos]

    pad_token_id = processor.tokenizer.pad_token_id
    answer_end = seq_len
    for j in range(assistant_pos, seq_len):
        if ids[j] == pad_token_id:
            answer_end = j
            break
    answer_ends = [answer_end]

    return {
        'inputs': inputs,
        'vision_start': vision_start,
        'vision_end': vision_end,
        'question_starts': question_starts,
        'question_ends': question_ends,
        'answer_starts': answer_starts,
        'answer_ends': answer_ends,
        'seq_len': seq_len,
    }


def compute_mmd(X, Y, gamma=1.0):
    """计算 Maximum Mean Discrepancy"""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    XX = np.dot(X, X.T)
    YY = np.dot(Y, Y.T)
    XY = np.dot(X, Y.T)

    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    K_XX = np.exp(-gamma * (X_sqnorms[:, None] + X_sqnorms[None, :] - 2 * XX))
    K_YY = np.exp(-gamma * (Y_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * YY))
    K_XY = np.exp(-gamma * (X_sqnorms[:, None] + Y_sqnorms[None, :] - 2 * XY))

    m = X.shape[0]
    n = Y.shape[0]

    mmd = (K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n))
    return mmd


def _subsample_pair(X, Y, max_samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y), max_samples)
    if n <= 0:
        return None, None
    idx_x = rng.choice(len(X), n, replace=False)
    idx_y = rng.choice(len(Y), n, replace=False)
    return X[idx_x], Y[idx_y]


def compute_c2st(X, Y, test_size=0.3, seed=42):
    """Classifier two-sample test (C2ST). Accuracy/AUC ~ 0.5 => distributions close."""
    Xs, Ys = _subsample_pair(X, Y, max_samples=5000, seed=seed)
    if Xs is None:
        return None
    X_all = np.vstack([Xs, Ys])
    y_all = np.array([0] * len(Xs) + [1] * len(Ys))

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=test_size, random_state=seed, stratify=y_all
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=1000, solver='liblinear')
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = None

    return {'acc': acc, 'auc': auc}


def compute_swd(X, Y, n_projections=100, seed=42):
    """Sliced Wasserstein Distance with random 1D projections."""
    Xs, Ys = _subsample_pair(X, Y, max_samples=2000, seed=seed)
    if Xs is None:
        return None
    rng = np.random.default_rng(seed)
    d = Xs.shape[1]
    dirs = rng.normal(size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8

    dists = []
    for v in dirs:
        proj_x = Xs @ v
        proj_y = Ys @ v
        proj_x = np.sort(proj_x)
        proj_y = np.sort(proj_y)
        dists.append(np.mean(np.abs(proj_x - proj_y)))
    return float(np.mean(dists))


def _sqrtm_psd(mat):
    """Matrix square root for symmetric PSD matrices."""
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def compute_frechet_distance(X, Y, pca_dim=64, seed=42):
    """Fréchet distance on PCA-reduced features."""
    Xs, Ys = _subsample_pair(X, Y, max_samples=5000, seed=seed)
    if Xs is None:
        return None

    d = Xs.shape[1]
    pca_dim = min(pca_dim, d, len(Xs) - 1, len(Ys) - 1)
    if pca_dim < 2:
        return None

    pca = PCA(n_components=pca_dim, random_state=seed)
    Z = pca.fit_transform(np.vstack([Xs, Ys]))
    Zx = Z[:len(Xs)]
    Zy = Z[len(Xs):]

    mu_x = Zx.mean(axis=0)
    mu_y = Zy.mean(axis=0)
    cov_x = np.cov(Zx, rowvar=False)
    cov_y = np.cov(Zy, rowvar=False)

    cov_x_sqrt = _sqrtm_psd(cov_x)
    cov_prod = cov_x_sqrt @ cov_y @ cov_x_sqrt
    cov_mean = _sqrtm_psd(cov_prod)

    diff = mu_x - mu_y
    frechet = diff @ diff + np.trace(cov_x + cov_y - 2 * cov_mean)
    return float(frechet)


def cohens_d(x, y):
    """计算 Cohen's d 效应量"""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1)**2 + (ny - 1) * np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)


def _describe_vectors(name: str, X: np.ndarray) -> None:
    """Quick numeric sanity checks for pooled token vectors."""
    if X is None:
        print(f"      {name}: None")
        return
    if not isinstance(X, np.ndarray):
        print(f"      {name}: (non-numpy) {type(X)}")
        return
    if X.ndim != 2:
        print(f"      {name}: shape={X.shape} (expected 2D)")
        return
    n, d = X.shape
    if n <= 0:
        print(f"      {name}: empty, dim={d}")
        return

    finite_rows = np.isfinite(X).all(axis=1)
    frac_finite = float(finite_rows.mean())
    X0 = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(X0, axis=1)
    rms = np.sqrt((X0 ** 2).mean(axis=1))
    print(
        f"      {name}: n_tokens={n}, dim={d}, finite_rows={frac_finite*100:.1f}%, "
        f"||x|| mean={norms.mean():.3f} std={norms.std():.3f} min={norms.min():.3f} max={norms.max():.3f}, "
        f"rms mean={rms.mean():.3f}"
    )


def _silverman_bandwidth(x: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth for 1D KDE."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n <= 1:
        return 1.0
    std = float(np.std(x, ddof=1))
    if std <= 1e-12:
        return 1.0
    return 1.06 * std * (n ** (-1 / 5))


def _plot_1d_density(ax, series, *, bins=60, kde_points=400, kde_bw=None, title="", xlabel=""):
    """Overlay hist + KDE for multiple 1D series. series: list of (name, data, color)."""
    cleaned = []
    for name, data, color in series:
        if data is None:
            continue
        x = np.asarray(data, dtype=np.float64)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        cleaned.append((name, x, color))

    if not cleaned:
        ax.axis("off")
        ax.set_title(title)
        return

    all_x = np.concatenate([x for _, x, _ in cleaned], axis=0)
    lo = float(np.percentile(all_x, 0.5))
    hi = float(np.percentile(all_x, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(all_x))
        hi = float(np.max(all_x) + 1e-6)

    edges = np.linspace(lo, hi, bins + 1)
    grid = np.linspace(lo, hi, kde_points).reshape(-1, 1)

    for name, x, color in cleaned:
        ax.hist(
            x,
            bins=edges,
            density=True,
            alpha=0.20,
            color=color,
            edgecolor="none",
            label=name,
        )
        bw = float(kde_bw) if kde_bw is not None else _silverman_bandwidth(x)
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)
        kde.fit(x.reshape(-1, 1))
        logp = kde.score_samples(grid)
        p = np.exp(logp)
        ax.plot(grid[:, 0], p, color=color, linewidth=2.0)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=10)


def _mean_var_alignment_stats(Xs: np.ndarray, Xt: np.ndarray) -> dict:
    """Training-aligned summary: compare mean + diagonal variance between two token sets.

    Note: this matches the *statistics* logged in training for mean/var/cos/var_ratio,
    but does not require token-wise alignment.
    """
    if Xs is None or Xt is None:
        return {}
    Xs = np.asarray(Xs, dtype=np.float32)
    Xt = np.asarray(Xt, dtype=np.float32)
    if Xs.ndim != 2 or Xt.ndim != 2:
        return {}
    if Xs.shape[1] != Xt.shape[1]:
        return {}

    ms = Xs.mean(axis=0)
    mt = Xt.mean(axis=0)
    vs = Xs.var(axis=0)
    vt = Xt.var(axis=0)

    mean_loss = float(((ms - mt) ** 2).mean())
    var_loss = float(((vs - vt) ** 2).mean())
    var_ratio = float(vs.mean() / (vt.mean() + 1e-8))
    cos = float(np.dot(ms, mt) / (np.linalg.norm(ms) * np.linalg.norm(mt) + 1e-8))

    # Token-wise (training-style) MSE: assumes token order is aligned.
    n = int(min(Xs.shape[0], Xt.shape[0]))
    token_mse = float(((Xs[:n] - Xt[:n]) ** 2).mean()) if n > 0 else float("nan")
    token_rmse = float(np.sqrt(max(token_mse, 0.0))) if np.isfinite(token_mse) else float("nan")
    return {
        "mean_loss": mean_loss,
        "var_loss": var_loss,
        "var_ratio": var_ratio,
        "cosine": cos,
        "token_mse": token_mse,
        "token_rmse": token_rmse,
        "student_var_mean": float(vs.mean()),
        "teacher_var_mean": float(vt.mean()),
        "n_student": int(Xs.shape[0]),
        "n_teacher": int(Xt.shape[0]),
        "dim": int(Xs.shape[1]),
    }


def _save_layerwise_alignment_report(report_rows: list[dict], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "layerwise_repair_alignment.csv")
    if not report_rows:
        return out_path

    # stable header order
    fieldnames = [
        "layer",
        "kept_ratio_target",
        "mode",
        "cosine",
        "var_ratio",
        "mean_loss",
        "var_loss",
        "token_mse",
        "token_rmse",
        "student_var_mean",
        "teacher_var_mean",
        "n_student",
        "n_teacher",
        "dim",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in report_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    return out_path


def _plot_layerwise_alignment_report(report_rows: list[dict], output_dir: str) -> None:
    """Plot layer-wise cosine/var_ratio for fake/corrected vs real."""
    if not report_rows:
        return
    # group rows by mode
    rows_fake = [r for r in report_rows if r.get("mode") == "fake_vs_real"]
    rows_corr = [r for r in report_rows if r.get("mode") == "corrected_vs_real"]
    if not rows_fake and not rows_corr:
        return

    def _extract(rows, key):
        rows2 = sorted(rows, key=lambda x: x["layer"])
        layers = [int(r["layer"]) for r in rows2]
        vals = [float(r.get(key, np.nan)) for r in rows2]
        return layers, vals

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if rows_fake:
        x, y = _extract(rows_fake, "cosine")
        axes[0].plot(x, y, marker="o", linewidth=2, label="fake vs real", color="orange")
        x2, y2 = _extract(rows_fake, "var_ratio")
        axes[1].plot(x2, y2, marker="o", linewidth=2, label="fake vs real", color="orange")
    if rows_corr:
        x, y = _extract(rows_corr, "cosine")
        axes[0].plot(x, y, marker="o", linewidth=2, label="corrected vs real", color="green")
        x2, y2 = _extract(rows_corr, "var_ratio")
        axes[1].plot(x2, y2, marker="o", linewidth=2, label="corrected vs real", color="green")

    axes[0].set_title("Layer-wise mean direction (cosine)")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("cosine(ms, mt)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set_title("Layer-wise variance scale (var_ratio)")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("mean(var_s) / mean(var_t)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "layerwise_repair_alignment.png"), dpi=150)
    plt.close()


def collect_h_vectors(model, processor, samples, device, apply_next_layernorm: bool = True):
    """收集所有样本的 h_real, h_fake, h_corrected

    新版架构（Delayed Repair）：
    - h_real: teacher（pruning_mode="keep_all"，不剪枝）
    - h_fake: 剪枝后、repair 前（pruning_mode="normal", apply_repair=False）
    - h_corrected: 剪枝后、repair 后（pruning_mode="normal", apply_repair=True）

    在 repair_layers 捕获这些表征。

    Args:
        apply_next_layernorm: 是否应用下一层的 input_layernorm，
            这样可视化的是 LLM 下一层实际看到的分布
    """
    # 获取 repair_layers
    repair_layers = getattr(model, 'repair_layers', [])
    if not repair_layers:
        print("Warning: No repair_layers configured, using pruning_layers instead")
        repair_layers = model.pruning_layers

    # 获取各层的 input_layernorm（用于归一化）
    llm = model.base_model.model.language_model
    layer_norms = {}
    if apply_next_layernorm:
        for layer_idx in repair_layers:
            next_layer_idx = layer_idx + 1
            if next_layer_idx < len(llm.layers):
                # 获取下一层的 input_layernorm
                next_layer = llm.layers[next_layer_idx]
                if hasattr(next_layer, 'original_layer'):
                    # PrunableLlamaDecoderLayer
                    layer_norms[layer_idx] = next_layer.original_layer.input_layernorm
                else:
                    layer_norms[layer_idx] = next_layer.input_layernorm
            else:
                # 最后一层，用 final norm
                layer_norms[layer_idx] = llm.norm
        print(f"  Will apply next layer's input_layernorm for visualization")

    h_real_all = defaultdict(list)
    h_fake_all = defaultdict(list)
    h_corrected_all = defaultdict(list)

    print(f"Collecting h vectors at layers: {repair_layers}")
    for i, sample in enumerate(samples):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(samples)}...")

        prep = preprocess_sample(sample, processor, device)
        if prep is None:
            continue

        inputs = prep['inputs']
        common_kwargs = dict(
            input_ids=inputs['input_ids'],
            pixel_values=inputs['pixel_values'],
            attention_mask=inputs['attention_mask'],
            vision_start=prep['vision_start'],
            vision_end=prep['vision_end'],
            question_starts=prep['question_starts'],
            question_ends=prep['question_ends'],
            answer_starts=prep['answer_starts'],
            answer_ends=prep['answer_ends'],
            return_pruning_info=False,
            capture_layers=repair_layers,
        )

        with torch.no_grad():
            # 1. h_real: teacher（不剪枝）
            output_real = model(**common_kwargs, pruning_mode="keep_all", apply_repair=False)

            # 2. h_fake: 剪枝后、repair 前
            output_fake = model(**common_kwargs, pruning_mode="normal", apply_repair=False)

            # 3. h_corrected: 剪枝后、repair 后
            output_corrected = model(**common_kwargs, pruning_mode="normal", apply_repair=True)

        # 提取表征
        captured_real = getattr(output_real, 'captured', None) or {}
        captured_fake = getattr(output_fake, 'captured', None) or {}
        captured_corrected = getattr(output_corrected, 'captured', None) or {}

        for layer_idx in repair_layers:
            ln = layer_norms.get(layer_idx, None)

            # h_real
            if layer_idx in captured_real:
                h = captured_real[layer_idx]['h']
                if ln is not None:
                    h = ln(h)
                mask = captured_real[layer_idx]['mask']
                for b in range(h.shape[0]):
                    valid_len = int(mask[b].sum().item())
                    if valid_len > 0:
                        h_real_all[layer_idx].append(h[b, :valid_len].float().cpu().numpy())

            # h_fake
            if layer_idx in captured_fake:
                h = captured_fake[layer_idx]['h']
                if ln is not None:
                    h = ln(h)
                mask = captured_fake[layer_idx]['mask']
                for b in range(h.shape[0]):
                    valid_len = int(mask[b].sum().item())
                    if valid_len > 0:
                        h_fake_all[layer_idx].append(h[b, :valid_len].float().cpu().numpy())

            # h_corrected
            if layer_idx in captured_corrected:
                h = captured_corrected[layer_idx]['h']
                if ln is not None:
                    h = ln(h)
                mask = captured_corrected[layer_idx]['mask']
                for b in range(h.shape[0]):
                    valid_len = int(mask[b].sum().item())
                    if valid_len > 0:
                        h_corrected_all[layer_idx].append(h[b, :valid_len].float().cpu().numpy())

    # 合并所有样本
    result = {}
    for layer_idx in repair_layers:
        if layer_idx not in h_real_all or not h_real_all[layer_idx]:
            print(f"  Warning: No data collected for layer {layer_idx}")
            continue
        result[layer_idx] = {
            'h_real': np.concatenate(h_real_all[layer_idx], axis=0),
            'h_fake': np.concatenate(h_fake_all[layer_idx], axis=0) if h_fake_all[layer_idx] else None,
            'h_corrected': np.concatenate(h_corrected_all[layer_idx], axis=0) if h_corrected_all[layer_idx] else None,
        }
        print(f"  Layer {layer_idx}: h_real={result[layer_idx]['h_real'].shape}, "
              f"h_fake={result[layer_idx]['h_fake'].shape if result[layer_idx]['h_fake'] is not None else None}, "
              f"h_corrected={result[layer_idx]['h_corrected'].shape if result[layer_idx]['h_corrected'] is not None else None}")

    return result


def analyze_and_visualize(
    h_data,
    layer_idx,
    output_dir,
    *,
    seed: int = 42,
    max_vis_tokens: int = 1000,
    tsne_pca_dim: int = 50,
    tsne_standardize: bool = True,
    tsne_metric: str = "euclidean",
):
    """分析并可视化单层的分布偏移"""
    h_real = h_data['h_real']
    h_fake = h_data['h_fake']
    h_corrected = h_data['h_corrected']
    rng = np.random.default_rng(seed)

    # 检查数据有效性
    if h_fake is None and h_corrected is None:
        print(f"  Layer {layer_idx}: No h_fake or h_corrected data, skipping...")
        return

    n_real = int(h_real.shape[0])
    n_fake = int(h_fake.shape[0]) if h_fake is not None else 0
    n_corrected = int(h_corrected.shape[0]) if h_corrected is not None else 0
    print(f"  Layer {layer_idx}: n_tokens real={n_real}, fake={n_fake}, corrected={n_corrected}")
    print("    Vector sanity stats (t-SNE axis scale is arbitrary):")
    _describe_vectors("h_real", h_real)
    _describe_vectors("h_fake", h_fake)
    _describe_vectors("h_corrected", h_corrected)

    # 如果 h_fake 为 None，用 h_corrected 作为 h_fake（只分析 corrected vs real）
    if h_fake is None:
        print(f"    Warning: h_fake is None, using h_corrected as baseline")
        h_fake = h_corrected
        h_corrected = None

    # ==================== [1] Global Mean Distance ====================
    print(f"    [1] Global Mean Distance (原有度量 - 可能掩盖样本级差异):")

    center_real = h_real.mean(axis=0)
    center_fake = h_fake.mean(axis=0)
    center_corrected = h_corrected.mean(axis=0) if h_corrected is not None else None

    l2_fake = np.linalg.norm(center_fake - center_real)
    l2_corrected = np.linalg.norm(center_corrected - center_real) if center_corrected is not None else 0

    cos_fake = 1 - np.dot(center_fake, center_real) / (np.linalg.norm(center_fake) * np.linalg.norm(center_real) + 1e-8)
    cos_corrected = 1 - np.dot(center_corrected, center_real) / (np.linalg.norm(center_corrected) * np.linalg.norm(center_real) + 1e-8) if center_corrected is not None else 0

    l2_improvement = (l2_fake - l2_corrected) / (l2_fake + 1e-8) * 100
    cos_improvement = (cos_fake - cos_corrected) / (cos_fake + 1e-8) * 100

    print(f"      Center L2 (h_fake -> h_real): {l2_fake:.4f}")
    print(f"      Center L2 (h_corrected -> h_real): {l2_corrected:.4f}")
    print(f"      L2 improvement: {l2_improvement:.1f}%")
    print(f"      Center cosine dist (h_fake -> h_real): {cos_fake:.4f}")
    print(f"      Center cosine dist (h_corrected -> h_real): {cos_corrected:.4f}")
    print(f"      Cosine improvement: {cos_improvement:.1f}%")

    # ==================== [2] Token-wise Distance ====================
    print(f"\n    [2] Token-wise Distance (需要 h_* 数组 token 对齐):")

    per_sample_l2_fake = None
    per_sample_l2_corrected = None
    per_sample_cos_fake = None
    per_sample_cos_corrected = None

    # Token-wise distances only make sense when arrays are aligned (same tokens, same order).
    if h_fake.shape == h_real.shape:
        per_sample_l2_fake = np.linalg.norm(h_fake - h_real, axis=1)
        per_sample_cos_fake = 1 - np.sum(h_fake * h_real, axis=1) / (
            np.linalg.norm(h_fake, axis=1) * np.linalg.norm(h_real, axis=1) + 1e-8
        )
    else:
        print(
            f"      Warning: h_fake shape {h_fake.shape} != h_real shape {h_real.shape}; "
            f"skip token-wise distance (needs alignment)."
        )

    if (h_corrected is not None) and (h_corrected.shape == h_real.shape):
        per_sample_l2_corrected = np.linalg.norm(h_corrected - h_real, axis=1)
        per_sample_cos_corrected = 1 - np.sum(h_corrected * h_real, axis=1) / (
            np.linalg.norm(h_corrected, axis=1) * np.linalg.norm(h_real, axis=1) + 1e-8
        )
    elif h_corrected is not None:
        print(
            f"      Warning: h_corrected shape {h_corrected.shape} != h_real shape {h_real.shape}; "
            f"skip token-wise distance (needs alignment)."
        )

    if per_sample_l2_fake is not None and per_sample_l2_corrected is not None:
        l2_imp = (per_sample_l2_fake.mean() - per_sample_l2_corrected.mean()) / (per_sample_l2_fake.mean() + 1e-8) * 100
        print(f"      Token-wise L2 (h_fake -> h_real): mean={per_sample_l2_fake.mean():.4f}, std={per_sample_l2_fake.std():.4f}")
        print(f"      Token-wise L2 (h_corrected -> h_real): mean={per_sample_l2_corrected.mean():.4f}, std={per_sample_l2_corrected.std():.4f}")
        print(f"      Token-wise L2 improvement: {l2_imp:.1f}%")

    if per_sample_cos_fake is not None and per_sample_cos_corrected is not None:
        cos_imp = (per_sample_cos_fake.mean() - per_sample_cos_corrected.mean()) / (per_sample_cos_fake.mean() + 1e-8) * 100
        print(f"      Token-wise cosine (h_fake -> h_real): mean={per_sample_cos_fake.mean():.4f}, std={per_sample_cos_fake.std():.4f}")
        print(f"      Token-wise cosine (h_corrected -> h_real): mean={per_sample_cos_corrected.mean():.4f}, std={per_sample_cos_corrected.std():.4f}")
        print(f"      Token-wise cosine improvement: {cos_imp:.1f}%")

    # ==================== [3] Distribution Shape ====================
    print(f"\n    [3] Distribution Shape (方差、标准差 - 关键诊断；这里的“样本”指 token):")

    std_real = h_real.std(axis=1).mean()
    std_fake = h_fake.std(axis=1).mean()
    std_corrected = h_corrected.std(axis=1).mean() if h_corrected is not None else 0

    var_real = h_real.var(axis=0).mean()
    var_fake = h_fake.var(axis=0).mean()
    var_corrected = h_corrected.var(axis=0).mean() if h_corrected is not None else 0

    print(f"      Mean std across tokens (h_real): {std_real:.4f}")
    print(f"      Mean std across tokens (h_fake): {std_fake:.4f}")
    print(f"      Mean std across tokens (h_corrected): {std_corrected:.4f}")
    print(f"      Std ratio (fake/real): {std_fake / (std_real + 1e-8):.4f}")
    print(f"      Std ratio (corrected/real): {std_corrected / (std_real + 1e-8):.4f}")
    print(f"      Mean var across dimensions (h_real): {var_real:.4f}")
    print(f"      Mean var across dimensions (h_fake): {var_fake:.4f}")
    print(f"      Mean var across dimensions (h_corrected): {var_corrected:.4f}")

    # ==================== [3.5] Correction Diagnostics ====================
    # This directly answers: "does the adapter output different magnitude/direction residuals per token?"
    # It decomposes correction into components parallel/orthogonal to the ideal gap (real - fake).
    if (h_corrected is not None) and (h_real.shape == h_fake.shape == h_corrected.shape):
        print(f"\n    [3.5] Correction Diagnostics (delta = corrected - fake, gap = real - fake):")
        delta = h_corrected - h_fake
        gap = h_real - h_fake

        delta_norm = np.linalg.norm(delta, axis=1)
        gap_norm = np.linalg.norm(gap, axis=1)
        dot = np.sum(delta * gap, axis=1)
        cos = dot / (delta_norm * gap_norm + 1e-8)
        # scalar amount along gap direction: how much of the gap is closed (can be >1 overshoot)
        frac = dot / (gap_norm * gap_norm + 1e-8)

        # orthogonal component magnitude relative to gap
        # delta_parallel = frac[:,None] * gap
        # delta_orth = delta - delta_parallel
        # We compute its norm efficiently.
        delta_orth_sq = np.clip((delta_norm ** 2) - (frac ** 2) * (gap_norm ** 2), 0.0, None)
        delta_orth_norm = np.sqrt(delta_orth_sq)
        rel_orth = delta_orth_norm / (gap_norm + 1e-8)

        moved_toward = float((frac > 0).mean()) * 100.0
        overshoot = float((frac > 1).mean()) * 100.0

        def _pct(x, p):
            return float(np.percentile(x, p))

        print(
            f"      ||gap||: mean={gap_norm.mean():.4f} std={gap_norm.std():.4f} "
            f"(p50={_pct(gap_norm,50):.4f}, p90={_pct(gap_norm,90):.4f})"
        )
        print(
            f"      ||delta||: mean={delta_norm.mean():.4f} std={delta_norm.std():.4f} "
            f"(p50={_pct(delta_norm,50):.4f}, p90={_pct(delta_norm,90):.4f})"
        )
        print(
            f"      cos(delta, gap): mean={cos.mean():.4f} std={cos.std():.4f} "
            f"(p10={_pct(cos,10):.4f}, p50={_pct(cos,50):.4f}, p90={_pct(cos,90):.4f})"
        )
        print(
            f"      frac_along_gap: mean={frac.mean():.4f} std={frac.std():.4f} "
            f"(p10={_pct(frac,10):.4f}, p50={_pct(frac,50):.4f}, p90={_pct(frac,90):.4f})"
        )
        print(f"      Toward real (frac>0): {moved_toward:.1f}%  |  Overshoot (frac>1): {overshoot:.1f}%")
        print(
            f"      rel_orth = ||delta_orth||/||gap||: mean={rel_orth.mean():.4f} std={rel_orth.std():.4f} "
            f"(p50={_pct(rel_orth,50):.4f}, p90={_pct(rel_orth,90):.4f})"
        )

    # ==================== [4] MMD ====================
    print(f"\n    [4] MMD (Maximum Mean Discrepancy - 更敏感的分布距离):")

    # 使用子采样计算 MMD（避免内存问题）
    max_mmd_samples = int(min(500, len(h_real), len(h_fake)))
    idx_real = rng.choice(len(h_real), max_mmd_samples, replace=False)
    idx_fake = rng.choice(len(h_fake), max_mmd_samples, replace=False)
    mmd_fake = compute_mmd(h_fake[idx_fake], h_real[idx_real])

    if h_corrected is not None:
        max_mmd_samples_c = int(min(500, len(h_real), len(h_corrected)))
        idx_real_c = rng.choice(len(h_real), max_mmd_samples_c, replace=False)
        idx_corr = rng.choice(len(h_corrected), max_mmd_samples_c, replace=False)
        mmd_corrected = compute_mmd(h_corrected[idx_corr], h_real[idx_real_c])
    else:
        mmd_corrected = 0

    mmd_improvement = (mmd_fake - mmd_corrected) / (mmd_fake + 1e-8) * 100

    print(f"      MMD (h_fake vs h_real): {mmd_fake:.6f}")
    print(f"      MMD (h_corrected vs h_real): {mmd_corrected:.6f}")
    print(f"      MMD improvement: {mmd_improvement:.1f}%")

    # ==================== [5] Two-Sample Tests / Distribution Distances ====================
    print(f"\n    [5] Two-Sample Tests / Distribution Distances:")

    # C2ST
    c2st_fake = compute_c2st(h_real, h_fake)
    c2st_corrected = compute_c2st(h_real, h_corrected) if h_corrected is not None else None
    if c2st_fake is not None:
        auc_str = f"{c2st_fake['auc']:.3f}" if c2st_fake['auc'] is not None else "N/A"
        print(f"      C2ST acc (h_fake vs h_real): {c2st_fake['acc']:.3f}, auc={auc_str}")
    if c2st_corrected is not None:
        auc_str = f"{c2st_corrected['auc']:.3f}" if c2st_corrected['auc'] is not None else "N/A"
        print(f"      C2ST acc (h_corrected vs h_real): {c2st_corrected['acc']:.3f}, auc={auc_str}")

    # SWD
    swd_fake = compute_swd(h_real, h_fake)
    swd_corrected = compute_swd(h_real, h_corrected) if h_corrected is not None else None
    if swd_fake is not None:
        print(f"      SWD (h_fake vs h_real): {swd_fake:.6f}")
    if swd_corrected is not None:
        swd_improvement = (swd_fake - swd_corrected) / (swd_fake + 1e-8) * 100 if swd_fake is not None else 0
        print(f"      SWD (h_corrected vs h_real): {swd_corrected:.6f}")
        print(f"      SWD improvement: {swd_improvement:.1f}%")

    # Frechet Distance (PCA-reduced)
    frechet_fake = compute_frechet_distance(h_real, h_fake)
    frechet_corrected = compute_frechet_distance(h_real, h_corrected) if h_corrected is not None else None
    if frechet_fake is not None:
        print(f"      Frechet (PCA) (h_fake vs h_real): {frechet_fake:.6f}")
    if frechet_corrected is not None:
        frechet_improvement = (frechet_fake - frechet_corrected) / (frechet_fake + 1e-8) * 100 if frechet_fake is not None else 0
        print(f"      Frechet (PCA) (h_corrected vs h_real): {frechet_corrected:.6f}")
        print(f"      Frechet improvement: {frechet_improvement:.1f}%")

    # ==================== 可视化 ====================
    os.makedirs(output_dir, exist_ok=True)

    # 子采样用于可视化
    n_common = min(len(h_real), len(h_fake), len(h_corrected) if h_corrected is not None else len(h_real))
    max_vis_samples = int(min(max_vis_tokens, n_common))
    if max_vis_samples <= 0:
        print("    Warning: No tokens available for visualization, skipping plots.")
        return
    vis_idx = rng.choice(n_common, max_vis_samples, replace=False)

    h_real_vis = h_real[vis_idx]
    h_fake_vis = h_fake[vis_idx]
    h_corrected_vis = h_corrected[vis_idx] if h_corrected is not None else None

    # ----- PCA -----
    print("    Running PCA...")
    pca = PCA(n_components=2)
    all_data = np.vstack([h_real_vis, h_fake_vis] + ([h_corrected_vis] if h_corrected_vis is not None else []))
    pca.fit(all_data)

    real_pca = pca.transform(h_real_vis)
    fake_pca = pca.transform(h_fake_vis)
    corrected_pca = pca.transform(h_corrected_vis) if h_corrected_vis is not None else None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='h_real', s=10)
    ax.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_pca is not None:
        ax.scatter(corrected_pca[:, 0], corrected_pca[:, 1], alpha=0.5, label='h_corrected', s=10)
    ax.legend()
    ax.set_title(f'Layer {layer_idx} - PCA')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}_pca.png'), dpi=150)
    plt.close()
    print(f"    Saved to {output_dir}/distribution_shift_layer{layer_idx}_pca.png")

    # ----- Per-sample distance plot -----
    if (
        per_sample_l2_fake is not None
        and per_sample_l2_corrected is not None
        and per_sample_cos_fake is not None
        and per_sample_cos_corrected is not None
    ):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(per_sample_l2_fake, bins=50, alpha=0.5, label='h_fake -> h_real', density=True)
        axes[0].hist(per_sample_l2_corrected, bins=50, alpha=0.5, label='h_corrected -> h_real', density=True)
        axes[0].legend()
        axes[0].set_title(f'Layer {layer_idx} - Token-wise L2 Distance')
        axes[0].set_xlabel('L2 Distance')

        axes[1].hist(per_sample_cos_fake, bins=50, alpha=0.5, label='h_fake -> h_real', density=True)
        axes[1].hist(per_sample_cos_corrected, bins=50, alpha=0.5, label='h_corrected -> h_real', density=True)
        axes[1].legend()
        axes[1].set_title(f'Layer {layer_idx} - Token-wise Cosine Distance')
        axes[1].set_xlabel('Cosine Distance')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'per_sample_distances_layer{layer_idx}.png'), dpi=150)
        plt.close()
        print(f"    Saved token-wise distance plot to {output_dir}/per_sample_distances_layer{layer_idx}.png")
    else:
        print("    Skipping token-wise distance plot (requires aligned arrays).")

    # ----- 1D Projection (LDA) -----
    # 用 LDA 找到最能区分 h_real 和 h_fake 的方向
    labels = np.array([0] * len(h_real_vis) + [1] * len(h_fake_vis))
    X_for_lda = np.vstack([h_real_vis, h_fake_vis])

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_for_lda, labels)

    proj_real = lda.transform(h_real_vis).flatten()
    proj_fake = lda.transform(h_fake_vis).flatten()
    proj_corrected = lda.transform(h_corrected_vis).flatten() if h_corrected_vis is not None else None

    # Cohen's d
    d_fake = abs(cohens_d(proj_fake, proj_real))
    d_corrected = abs(cohens_d(proj_corrected, proj_real)) if proj_corrected is not None else 0

    print(f"    1D projection (lda) effect sizes:")
    print(f"      Cohen's d (h_fake vs h_real): {d_fake:.3f}")
    print(f"      Cohen's d (h_corrected vs h_real): {d_corrected:.3f}")

    # 方差分析
    var_proj_real = proj_real.var()
    var_proj_fake = proj_fake.var()
    var_proj_corrected = proj_corrected.var() if proj_corrected is not None else 0

    print(f"    1D projection variance (解释平滑度):")
    print(f"      Var(proj_real): {var_proj_real:.4f}")
    print(f"      Var(proj_fake): {var_proj_fake:.4f}")
    print(f"      Var(proj_corrected): {var_proj_corrected:.4f}")
    print(f"      Std(proj_real): {np.sqrt(var_proj_real):.4f}")
    print(f"      Std(proj_fake): {np.sqrt(var_proj_fake):.4f}")
    print(f"      Std(proj_corrected): {np.sqrt(var_proj_corrected):.4f}")
    print(f"      Var ratio (corrected/real): {var_proj_corrected / (var_proj_real + 1e-8):.2f}x")
    print(f"      Var ratio (corrected/fake): {var_proj_corrected / (var_proj_fake + 1e-8):.2f}x")

    # ===== 修正量分析 =====
    if proj_corrected is not None:
        # 计算每个样本的修正量和方向
        mean_real = proj_real.mean()
        mean_fake = proj_fake.mean()
        mean_corrected = proj_corrected.mean()

        # 修正量（在 LDA 方向上）
        correction = proj_corrected - proj_fake  # 每个样本的修正量
        total_gap = mean_real - mean_fake  # fake 到 real 的总距离

        # 修正方向：正值表示向 real 方向移动
        correction_toward_real = correction * np.sign(total_gap)

        # 统计
        mean_correction = correction.mean()
        mean_correction_toward_real = correction_toward_real.mean()
        pct_toward_real = (correction_toward_real > 0).mean() * 100
        correction_ratio = mean_correction_toward_real / (abs(total_gap) + 1e-8) * 100

        print(f"    Adapter 修正分析 (1D LDA 方向):")
        print(f"      h_real 中心: {mean_real:.3f}")
        print(f"      h_fake 中心: {mean_fake:.3f}")
        print(f"      h_corrected 中心: {mean_corrected:.3f}")
        print(f"      fake->real 距离: {total_gap:.3f}")
        print(f"      fake->corrected 移动: {mean_correction:.3f}")
        print(f"      向 real 方向移动: {mean_correction_toward_real:.3f} ({correction_ratio:.1f}% of gap)")
        print(f"      向 real 方向移动的样本比例: {pct_toward_real:.1f}%")

    # 1D projection 可视化（增强版）
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：分布直方图
    ax = axes[0]
    series = [
        (r"$h_{real}$ (Full Model)", proj_real, "gray"),
        (r"$h_{fake}$ (Pruned w/o Adapter)", proj_fake, "orange"),
    ]
    if proj_corrected is not None:
        series.append((r"$h_{corrected}$ (ACP Final)", proj_corrected, "green"))

    _plot_1d_density(
        ax,
        series,
        bins=60,
        title=f"Layer {layer_idx}: 1D Projection (LDA dir) Density",
        xlabel="Projection value",
    )

    # 添加中心线（均值）
    ax.axvline(float(np.mean(proj_real)), color="gray", linestyle="--", linewidth=2, alpha=0.8)
    ax.axvline(float(np.mean(proj_fake)), color="orange", linestyle="--", linewidth=2, alpha=0.8)
    if proj_corrected is not None:
        ax.axvline(float(np.mean(proj_corrected)), color="green", linestyle="--", linewidth=2, alpha=0.8)

    # 右图：修正方向分析
    ax2 = axes[1]
    if proj_corrected is not None:
        # 绘制修正量分布
        ax2.hist(correction_toward_real, bins=50, alpha=0.7, color='blue', density=True)
        ax2.axvline(0, color='red', linestyle='-', linewidth=2, label='No correction')
        ax2.axvline(correction_toward_real.mean(), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {correction_toward_real.mean():.3f}')

        # 获取当前y轴范围
        y_max = ax2.get_ylim()[1]

        ax2.legend(fontsize=10)
        ax2.set_title(f'Layer {layer_idx}: Adapter Correction Direction\n(Positive = Toward Real)', fontsize=14)
        ax2.set_xlabel('Correction Amount (toward real)', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)

        # 添加文字说明
        textstr = f'Gap (fake→real): {total_gap:.2f}\nCorrection: {mean_correction_toward_real:.2f} ({correction_ratio:.1f}%)\nSamples toward real: {pct_toward_real:.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}_proj1d.png'), dpi=150)
    plt.close()
    print(f"    Saved to {output_dir}/distribution_shift_layer{layer_idx}_proj1d.png")

    # ----- t-SNE -----
    print("    Running t-SNE (note: axis scale has no physical meaning)...")
    all_vis = np.vstack([h_real_vis, h_fake_vis] + ([h_corrected_vis] if h_corrected_vis is not None else []))
    tsne_data = all_vis

    # Preprocess for stability: standardize then PCA-reduce.
    # This avoids t-SNE being dominated by raw feature scale and makes runs more comparable.
    if tsne_standardize:
        scaler = StandardScaler()
        tsne_data = scaler.fit_transform(tsne_data)

    if tsne_pca_dim and tsne_pca_dim > 0 and tsne_data.shape[1] > tsne_pca_dim:
        pca_tsne = PCA(n_components=min(tsne_pca_dim, tsne_data.shape[1]), random_state=seed)
        tsne_data = pca_tsne.fit_transform(tsne_data)

    # For non-euclidean metrics, sklearn requires method='exact'.
    tsne_method = "barnes_hut" if tsne_metric == "euclidean" else "exact"
    perplexity = min(30.0, max(5.0, (len(tsne_data) - 1) / 3.0))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
        metric=tsne_metric,
        method=tsne_method,
    )
    all_tsne = tsne.fit_transform(tsne_data)

    n_real = len(h_real_vis)
    n_fake = len(h_fake_vis)
    real_tsne = all_tsne[:n_real]
    fake_tsne = all_tsne[n_real:n_real + n_fake]
    corrected_tsne = all_tsne[n_real + n_fake:] if h_corrected_vis is not None else None

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='h_real', s=10)
    ax.scatter(fake_tsne[:, 0], fake_tsne[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_tsne is not None:
        ax.scatter(corrected_tsne[:, 0], corrected_tsne[:, 1], alpha=0.5, label='h_corrected', s=10)
    ax.legend()
    ax.set_title(f'Layer {layer_idx} - t-SNE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}.png'), dpi=150)
    plt.close()
    print(f"    Saved t-SNE to {output_dir}/distribution_shift_layer{layer_idx}.png")

    # ----- DA comparison (Domain Adaptation 视角) -----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PCA
    axes[0, 0].scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='h_real', s=10)
    axes[0, 0].scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_pca is not None:
        axes[0, 0].scatter(corrected_pca[:, 0], corrected_pca[:, 1], alpha=0.5, label='h_corrected', s=10)
    axes[0, 0].legend()
    axes[0, 0].set_title('PCA')

    # t-SNE
    axes[0, 1].scatter(real_tsne[:, 0], real_tsne[:, 1], alpha=0.5, label='h_real', s=10)
    axes[0, 1].scatter(fake_tsne[:, 0], fake_tsne[:, 1], alpha=0.5, label='h_fake', s=10)
    if corrected_tsne is not None:
        axes[0, 1].scatter(corrected_tsne[:, 0], corrected_tsne[:, 1], alpha=0.5, label='h_corrected', s=10)
    axes[0, 1].legend()
    axes[0, 1].set_title('t-SNE')

    # 1D projection
    axes[1, 0].hist(proj_real, bins=50, alpha=0.5, label='h_real', density=True)
    axes[1, 0].hist(proj_fake, bins=50, alpha=0.5, label='h_fake', density=True)
    if proj_corrected is not None:
        axes[1, 0].hist(proj_corrected, bins=50, alpha=0.5, label='h_corrected', density=True)
    axes[1, 0].legend()
    axes[1, 0].set_title(f'1D LDA Projection (d_fake={d_fake:.2f}, d_corr={d_corrected:.2f})')

    # Per-sample L2 distance
    if per_sample_l2_fake is not None and per_sample_l2_corrected is not None:
        axes[1, 1].hist(per_sample_l2_fake, bins=50, alpha=0.5, label='h_fake -> h_real', density=True)
        axes[1, 1].hist(per_sample_l2_corrected, bins=50, alpha=0.5, label='h_corrected -> h_real', density=True)
        axes[1, 1].legend()
        axes[1, 1].set_title('Token-wise L2 Distance')
    else:
        axes[1, 1].axis("off")
        axes[1, 1].set_title('Token-wise L2 Distance (skipped)')

    plt.suptitle(f'Layer {layer_idx} - Distribution Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'distribution_shift_layer{layer_idx}_da.png'), dpi=150)
    plt.close()
    print(f"    Saved DA comparison to {output_dir}/distribution_shift_layer{layer_idx}_da.png")


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Using config: {args.config}")
    print(f"Using seed: {args.seed}")

    # Avoid overwriting outputs between no_layernorm/layernorm runs.
    default_tag = "raw" if args.no_layernorm else "ln"
    run_tag = args.tag.strip() or default_tag
    out_dir = args.output_dir if args.flat_output else os.path.join(args.output_dir, run_tag)
    print(f"Output dir: {out_dir} (tag={run_tag}, flat={args.flat_output})")

    # 加载模型
    model, processor = load_model_and_processor(args.checkpoint, args.config, device)

    # 加载样本
    print(f"\nLoading {args.num_samples} samples...")
    samples = load_samples(args.num_samples, args.config)
    print(f"Loaded {len(samples)} samples")

    # 收集 h 向量
    apply_ln = not args.no_layernorm
    h_data = collect_h_vectors(model, processor, samples, device, apply_next_layernorm=apply_ln)

    # === Training-aligned summary (mean/var/cos/var_ratio) ===
    # This matches the training log's repair stats more directly than t-SNE.
    report_rows = []
    for layer_idx in sorted(h_data.keys()):
        h_real = h_data[layer_idx].get("h_real")
        h_fake = h_data[layer_idx].get("h_fake")
        h_corr = h_data[layer_idx].get("h_corrected")

        if h_fake is not None:
            s = _mean_var_alignment_stats(h_fake, h_real)
            if s:
                report_rows.append({"layer": int(layer_idx), "mode": "fake_vs_real", **s})
        if h_corr is not None:
            s = _mean_var_alignment_stats(h_corr, h_real)
            if s:
                report_rows.append({"layer": int(layer_idx), "mode": "corrected_vs_real", **s})

    csv_path = _save_layerwise_alignment_report(report_rows, out_dir)
    _plot_layerwise_alignment_report(report_rows, out_dir)
    if report_rows:
        print(f"\nSaved training-aligned layerwise report to {csv_path}")
        print(f"Saved plot to {out_dir}/layerwise_repair_alignment.png")

    # 分析和可视化
    print("\nGenerating visualizations...")
    for layer_idx in sorted(h_data.keys()):
        analyze_and_visualize(
            h_data[layer_idx],
            layer_idx,
            out_dir,
            seed=args.seed,
            max_vis_tokens=args.max_vis_tokens,
            tsne_pca_dim=args.tsne_pca_dim,
            tsne_standardize=(not args.no_tsne_standardize),
            tsne_metric=args.tsne_metric,
        )

    print(f"\nDone! Visualizations saved to {out_dir}")


if __name__ == "__main__":
    main()
