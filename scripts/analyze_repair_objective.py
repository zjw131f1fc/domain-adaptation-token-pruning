#!/usr/bin/env python
"""Paper-style visualization for the *repair objective* (distribution alignment) layer-by-layer.

This script is meant for the paper visualization experiment:
    "Adapter (delayed repair) repairs the pruning-induced representation gap."

Key idea: evaluate the SAME objective that we train with (engine/train_utils.py):
    - teacher: keep_all pruning (no repair)
    - student OFF: normal pruning (repair disabled)
    - student ON : normal pruning (repair enabled)
and measure distribution alignment on **gen_answer tokens** at each decoder layer:
    total = mean_mse + var_weight * var_mse   (loss_type='mean_var')
    or total = token_mse                      (loss_type='mse')

Outputs (all saved into one folder = --output_dir):
    - CSV tables: layerwise + summary + repair-layer local effect + C2ST (optional)
    - Paper-ready figures: PDF + PNG
    - PAPER_VISUALS.md: what each figure/table means + caveats

Recommended usage (same checkpoint, toggle repair):
    CUDA_VISIBLE_DEVICES=0 python scripts/analyze_repair_objective.py \\
      --checkpoint outputs/tasks/20260301-2248_vqa-vqav2_llava157b_250c/checkpoints/checkpoint_final.pt \\
      --config configs/vision_token_pruning.yaml \\
      --num_samples 64 --batch_size 1 --split test \\
      --capture_layers all \\
      --output_dir outputs/visualizations/repair_paper_250c_n64 \\
      --paper_size double
"""

import os

# NOTE: force-set cache env vars because some clusters pre-set HF_HOME to a non-writable shared path.
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Prefer offline / local cache to make the analysis robust on clusters with restricted network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import re
import json
import math
import argparse
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# repo root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Trained checkpoint (.pt)")
    p.add_argument("--config", type=str, default="configs/vision_token_pruning.yaml", help="Config yaml")
    p.add_argument(
        "--model_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HF model id (for base weights + processor)",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Where to save ALL outputs")
    p.add_argument("--device", type=str, default="cuda:0", help="cuda:0 / cpu")
    p.add_argument("--split", type=str, default="test", choices=["train", "test"], help="Dataset split")
    p.add_argument("--num_samples", type=int, default=64, help="How many samples to evaluate")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size (batch_size=1 recommended)")
    p.add_argument("--max_length", type=int, default=1024, help="Tokenizer max_length")
    p.add_argument(
        "--capture_layers",
        type=str,
        default="all",
        help=(
            "Layers to capture, formats: 'all' | '0-31' | '0:32' | '13,22,29'. "
            "Use 'all' for per-layer curves."
        ),
    )
    p.add_argument(
        "--loss_type",
        type=str,
        default="",
        help="Override repair_loss_type (default: read from config.method_settings.repair_loss_type)",
    )
    p.add_argument(
        "--var_weight",
        type=float,
        default=float("nan"),
        help="Override repair_var_weight (default: read from config.method_settings.repair_var_weight)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed (PCA/C2ST split)")
    p.add_argument(
        "--paper_size",
        type=str,
        default="double",
        choices=["single", "double"],
        help="Figure width: single-column or double-column",
    )
    p.add_argument("--no_c2st", action="store_true", help="Skip C2ST AUC metrics (faster)")
    p.add_argument("--no_pca", action="store_true", help="Skip PCA scatter figures")
    p.add_argument("--no_lda", action="store_true", help="Skip per-layer LDA C2ST metrics/plots (faster)")
    p.add_argument(
        "--pca_topk",
        type=int,
        default=2,
        help="Add top-k gain layers (OFF-ON) into PCA/C2ST layer set (in addition to repair layers).",
    )
    p.add_argument("--pca_max_layers", type=int, default=6, help="Upper bound on PCA/C2ST layers (avoid many files)")
    p.add_argument(
        "--exclude_layers",
        type=str,
        default="",
        help=(
            "Exclude these layer indices from *figures* (and PCA/C2ST selection). "
            "Formats: '31' | '0-31' | '0:32' | '13,22,29'. "
            "CSV tables still include all captured layers."
        ),
    )
    p.add_argument(
        "--exclude_last_layer",
        action="store_true",
        help="Shortcut: exclude the final decoder layer from figures (often noisy / dominated by head).",
    )
    return p.parse_args()


def _set_paper_style(paper_size: str) -> Tuple[float, float]:
    """Return (width_in, base_height_in) and set rcParams for paper-like look."""
    if paper_size == "single":
        width = 3.4
        base_h = 2.2
        font = 8
    else:
        width = 6.8
        base_h = 2.6
        font = 9

    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": font,
            "axes.labelsize": font,
            "axes.titlesize": font,
            "legend.fontsize": font - 1,
            "xtick.labelsize": font - 1,
            "ytick.labelsize": font - 1,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )
    return width, base_h


def _save_fig(fig: plt.Figure, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{name}.pdf")
    png_path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print(f"Saved figure: {pdf_path}")


def _parse_layer_list(s: str, *, num_layers: Optional[int]) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    s_lower = s.lower()
    if s_lower in {"all", "every"}:
        if num_layers is None:
            raise ValueError("capture_layers='all' requires num_layers.")
        return list(range(int(num_layers)))

    def _add_range(out: List[int], start: int, end_exclusive: int) -> None:
        if end_exclusive <= start:
            return
        out.extend(list(range(start, end_exclusive)))

    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        # negative index support (e.g., "-1" == last layer)
        if part.startswith("-") and part[1:].isdigit():
            if num_layers is None:
                raise ValueError(f"Negative layer index '{part}' requires num_layers.")
            idx = int(num_layers) + int(part)
            out.append(int(idx))
            continue
        if part.lower() in {"last", "end"}:
            if num_layers is None:
                raise ValueError(f"Layer keyword '{part}' requires num_layers.")
            out.append(int(num_layers) - 1)
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            a = a.strip()
            b = b.strip()
            start = int(a) if a else 0
            if b:
                end = int(b)
            else:
                if num_layers is None:
                    raise ValueError(f"Open-ended range '{part}' requires num_layers.")
                end = int(num_layers)
            _add_range(out, start, end)
        elif "-" in part:
            a, b = part.split("-", 1)
            start = int(a.strip())
            end_inclusive = int(b.strip())
            _add_range(out, start, end_inclusive + 1)
        else:
            out.append(int(part))

    seen = set()
    dedup = []
    for x in out:
        # guard valid range (avoid silently adding weird negative/overflow)
        if num_layers is not None and (int(x) < 0 or int(x) >= int(num_layers)):
            continue
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def load_processor(model_path: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    return processor


def _infer_adapter_layers_from_state_dict(adapter_state_dict: Dict[str, Any]) -> List[int]:
    if not adapter_state_dict:
        return []
    layers = set()
    for k in adapter_state_dict.keys():
        m = re.match(r"^adapters\.(\d+)\.", str(k))
        if m:
            layers.add(int(m.group(1)))
    return sorted(layers)


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: torch.device, model_path: str):
    """Load PrunableLlava model; auto-enable repair adapter only if checkpoint contains its weights."""
    from transformers import LlavaForConditionalGeneration
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
    from engine.configs.loader import load_config

    config = load_config(override_file=config_path)
    method_cfg = config["method_settings"]

    meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_has_repair = ("repair_context_encoder_state_dict" in meta) and ("repair_adapter_state_dict" in meta)
    use_repair_adapter = bool(method_cfg.get("use_repair_adapter", False) and ckpt_has_repair)
    if bool(method_cfg.get("use_repair_adapter", False)) and not ckpt_has_repair:
        print("Note: config requests repair adapter, but checkpoint has no repair weights; disabling repair.")

    # Prefer checkpoint-inferred repair layers if available (avoid config/checkpoint mismatch).
    repair_layers_cfg = list(method_cfg.get("repair_layers", None) or [])
    repair_source_layers_cfg = method_cfg.get("repair_source_layers", None)
    repair_layers_for_model = repair_layers_cfg
    repair_source_layers_for_model = repair_source_layers_cfg
    if ckpt_has_repair:
        inferred = _infer_adapter_layers_from_state_dict(meta.get("repair_adapter_state_dict", {}) or {})
        if inferred:
            if repair_layers_cfg and (sorted([int(x) for x in repair_layers_cfg]) != inferred):
                print(f"Note: config repair_layers={repair_layers_cfg} != checkpoint repair_layers={inferred}; using checkpoint.")
            repair_layers_for_model = inferred
            if repair_source_layers_cfg is not None and len(list(repair_source_layers_cfg)) != len(inferred):
                print(
                    "Note: config repair_source_layers length mismatches checkpoint repair_layers; "
                    "disabling explicit mapping (auto-pick nearest pruning layer)."
                )
                repair_source_layers_for_model = None

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)

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
        repair_layers=repair_layers_for_model,
        repair_source_layers=repair_source_layers_for_model,
        repair_bottleneck_dim=method_cfg.get("repair_bottleneck_dim", 512),
        repair_dropout=0.0,
        repair_mask_encoder_type=method_cfg.get("repair_mask_encoder_type", "attention"),
        repair_use_pruned_info=method_cfg.get("repair_use_pruned_info", True),
        repair_alpha_init=method_cfg.get("repair_alpha_init", 0.1),
    )
    model.freeze_base_model()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "pruner_state_dict" in ckpt:
        model.pruner_manager.load_state_dict(ckpt["pruner_state_dict"])
        print("Loaded pruner_state_dict")
    if model.use_repair_adapter:
        model.repair_context_encoder.load_state_dict(ckpt["repair_context_encoder_state_dict"])
        model.repair_adapter_manager.load_state_dict(ckpt["repair_adapter_state_dict"])
        print("Loaded repair_context_encoder_state_dict + repair_adapter_state_dict")

    model.eval()
    return model, config


def _infer_num_decoder_layers(model) -> int:
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
    raise RuntimeError("Could not infer num decoder layers from model.")


def load_samples(num_samples: int, *, config_path: str, split: str) -> List[Dict[str, Any]]:
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset
    from itertools import islice

    config = load_config(override_file=config_path)
    bundle = load_dataset(config)
    dataset = bundle["splits"][split]
    return list(islice(dataset, int(num_samples)))


def _iter_batches(samples: List[Dict[str, Any]], batch_size: int) -> List[List[Dict[str, Any]]]:
    out = []
    for i in range(0, len(samples), batch_size):
        out.append(samples[i : i + batch_size])
    return out


def _extract_pooled_per_sample(capture_entry: Dict[str, torch.Tensor]) -> List[np.ndarray]:
    """Return pooled vectors per sample: list of (D,) float32 arrays, length=batch_size."""
    h = capture_entry["h"]  # (b,L,D)
    m = capture_entry["mask"]  # (b,L)
    if h.dim() != 3 or m.dim() != 2:
        raise ValueError(f"Unexpected capture shapes: h={tuple(h.shape)} mask={tuple(m.shape)}")
    b, _, d = h.shape
    pooled: List[np.ndarray] = []
    for i in range(b):
        valid = m[i] > 0.5
        if valid.sum().item() <= 0:
            pooled.append(np.zeros((d,), dtype=np.float32))
            continue
        vec = h[i][valid].float().mean(dim=0)
        pooled.append(vec.detach().cpu().numpy().astype(np.float32, copy=False))
    return pooled


def _flatten_capture_tokens(
    capture_entry: Dict[str, torch.Tensor],
    *,
    mask_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Flatten (b,L,D) -> (N,D) using mask (b,L), preserving sample/token order.

    This mirrors the training helper `_flatten_masked` but stays local to this script.
    """
    h = capture_entry["h"]
    m = capture_entry["mask"] if mask_override is None else mask_override
    if h.dim() != 3 or m.dim() != 2:
        raise ValueError(f"Unexpected capture shapes: h={tuple(h.shape)} mask={tuple(m.shape)}")
    b, L, d = h.shape
    h2 = h.reshape(b * L, d)
    m2 = m.reshape(b * L).to(dtype=torch.bool)
    if int(m2.sum().item()) <= 0:
        return h2[:0]
    return h2[m2]


@dataclass
class StreamingVectorMoments:
    """Streaming mean/var (unbiased=False) for vectors, aggregated over tokens."""

    n: int = 0
    sum: Optional[np.ndarray] = None  # float64, (D,)
    sumsq: Optional[np.ndarray] = None  # float64, (D,)

    def update(self, X: torch.Tensor) -> None:
        """Update with X of shape (N,D)."""
        if X is None:
            return
        if X.dim() != 2:
            raise ValueError(f"StreamingVectorMoments expects X=(N,D), got {tuple(X.shape)}")
        N = int(X.shape[0])
        if N <= 0:
            return
        x_f32 = X.float()
        x_sum = x_f32.sum(dim=0).double().cpu().numpy()
        x_sumsq = (x_f32 * x_f32).sum(dim=0).double().cpu().numpy()
        if self.sum is None:
            self.sum = x_sum
            self.sumsq = x_sumsq
        else:
            self.sum += x_sum
            self.sumsq += x_sumsq
        self.n += N

    def mean_var(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.n <= 0 or self.sum is None or self.sumsq is None:
            raise ValueError("StreamingVectorMoments is empty.")
        mean = self.sum / float(self.n)
        var = self.sumsq / float(self.n) - mean * mean
        # numeric guard
        var = np.maximum(var, 0.0)
        return mean, var


@dataclass
class StreamingMSE:
    """Streaming token-wise MSE between paired tensors (aligned tokens)."""

    sse: float = 0.0
    n_elem: int = 0

    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        if X is None or Y is None:
            return
        if X.shape != Y.shape:
            raise ValueError(f"StreamingMSE requires same shapes, got {tuple(X.shape)} vs {tuple(Y.shape)}")
        if X.numel() <= 0:
            return
        diff = (X.float() - Y.float())
        self.sse += float((diff * diff).sum().detach().cpu().item())
        self.n_elem += int(diff.numel())

    def value(self) -> float:
        if self.n_elem <= 0:
            return float("nan")
        return float(self.sse / float(self.n_elem))


@dataclass
class StreamingScalarMoments:
    """Streaming mean/std for scalars (for diagnostics like ||delta||)."""

    n: int = 0
    sum: float = 0.0
    sumsq: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        if x is None or x.numel() <= 0:
            return
        x_f32 = x.float()
        self.sum += float(x_f32.sum().detach().cpu().item())
        self.sumsq += float((x_f32 * x_f32).sum().detach().cpu().item())
        self.n += int(x_f32.numel())

    def mean_std(self) -> Tuple[float, float]:
        if self.n <= 0:
            return float("nan"), float("nan")
        mean = self.sum / float(self.n)
        var = self.sumsq / float(self.n) - mean * mean
        var = max(float(var), 0.0)
        return float(mean), float(math.sqrt(var))


def _c2st_auc(
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    seed: int,
) -> Dict[str, float]:
    """Classifier two-sample test using logistic regression; returns AUC/ACC.

    Interpretation:
        AUC ~ 0.5 => hard to distinguish => distributions are close.
        AUC >> 0.5 => easy to distinguish => distributions differ.
    """
    n = min(len(X0), len(X1))
    if n < 8:
        return {"auc": float("nan"), "acc": float("nan"), "n": float(n)}
    X0 = X0[:n]
    X1 = X1[:n]
    X = np.concatenate([X0, X1], axis=0)
    y = np.asarray([0] * n + [1] * n, dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=int(seed),
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=1)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob))
    acc = float(accuracy_score(y_test, (prob > 0.5).astype(np.int64)))
    return {"auc": auc, "acc": acc, "n": float(n)}


def _lda_c2st_auc(
    X0: np.ndarray,
    X1: np.ndarray,
    *,
    seed: int,
) -> Dict[str, float]:
    """LDA-based classifier two-sample test (C2ST); returns AUC/ACC.

    Interpretation is the same as logistic-regression C2ST:
        AUC ~ 0.5 => indistinguishable => distributions are close.
        AUC >> 0.5 => easy to distinguish => distributions differ.
    """
    n = min(len(X0), len(X1))
    if n < 8:
        return {"auc": float("nan"), "acc": float("nan"), "n": float(n)}
    X0 = X0[:n]
    X1 = X1[:n]
    X = np.concatenate([X0, X1], axis=0)
    y = np.asarray([0] * n + [1] * n, dtype=np.int64)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=int(seed),
        stratify=y,
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LinearDiscriminantAnalysis(solver="svd")
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob))
    acc = float(accuracy_score(y_test, clf.predict(X_test)))
    return {"auc": auc, "acc": acc, "n": float(n)}


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError(
            f"CUDA requested ({args.device}) but torch.cuda.is_available() is False. "
            "Run on a CUDA-enabled node or set --device cpu."
        )
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    width_in, base_h_in = _set_paper_style(args.paper_size)

    print(f"Loading model from checkpoint: {args.checkpoint}")
    model, config = load_model_from_checkpoint(args.checkpoint, args.config, device, args.model_path)
    processor = load_processor(args.model_path)

    num_layers = _infer_num_decoder_layers(model)
    capture_layers = _parse_layer_list(args.capture_layers, num_layers=num_layers)
    if not capture_layers:
        raise ValueError("Empty capture_layers. Use --capture_layers all or provide a list/range.")

    method_cfg = config["method_settings"]
    pruning_layers = [int(x) for x in (method_cfg.get("pruning_layers", []) or [])]
    repair_layers = [int(x) for x in (getattr(model, "repair_layers", None) or [])] if bool(getattr(model, "use_repair_adapter", False)) else []

    loss_type = (args.loss_type or str(method_cfg.get("repair_loss_type", "mean_var"))).strip()
    var_weight = float(method_cfg.get("repair_var_weight", 1.0)) if math.isnan(float(args.var_weight)) else float(args.var_weight)

    target_token_num = method_cfg.get("target_token_num", None)
    teacher_pruning_mode = method_cfg.get("teacher_pruning_mode", "keep_all")

    # Exclude layers from paper figures (e.g., last layer often dominated by head effects)
    excluded_layers = set(_parse_layer_list(args.exclude_layers, num_layers=num_layers)) if str(args.exclude_layers).strip() else set()
    if bool(getattr(args, "exclude_last_layer", False)):
        excluded_layers.add(int(num_layers) - 1)

    # Save meta for reproducibility
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cmd": " ".join(sys.argv),
        "checkpoint": args.checkpoint,
        "config": args.config,
        "model_path": args.model_path,
        "device": str(args.device),
        "dataset": str(config["dataset_settings"]["name"]),
        "split": args.split,
        "num_samples_requested": int(args.num_samples),
        "batch_size": int(args.batch_size),
        "max_length": int(args.max_length),
        "num_decoder_layers": int(num_layers),
        "capture_layers": [int(x) for x in capture_layers],
        "excluded_layers_for_figures": sorted([int(x) for x in excluded_layers]),
        "pruning_layers": [int(x) for x in pruning_layers],
        "use_repair_adapter": bool(getattr(model, "use_repair_adapter", False)),
        "repair_layers": [int(x) for x in repair_layers],
        "objective": {"loss_type": loss_type, "var_weight": float(var_weight)},
        "teacher_pruning_mode": str(teacher_pruning_mode),
        "target_token_num": target_token_num if target_token_num is None else int(target_token_num),
    }
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Loading samples: split={args.split} num_samples={args.num_samples}")
    samples = load_samples(args.num_samples, config_path=args.config, split=args.split)
    batches = _iter_batches(samples, int(args.batch_size))
    print(f"Prepared {len(samples)} samples => {len(batches)} batches (batch_size={args.batch_size})")

    from engine.data_utils import preprocess_batch

    # Also store pooled vectors for PCA/C2ST (per-sample, float32).
    # PCA/C2ST is qualitative/auxiliary.
    pooled: Dict[int, Dict[str, List[np.ndarray]]] = {
        int(l): {"teacher": [], "off": [], "on": []} for l in capture_layers
    }

    # Objective aggregation buffers (token-aggregate, stable even when each answer is short).
    moments = {
        int(l): {
            "teacher": StreamingVectorMoments(),
            "off": StreamingVectorMoments(),
            "on": StreamingVectorMoments(),
        }
        for l in capture_layers
    }
    token_mse_acc = {
        int(l): {
            "off": StreamingMSE(),
            "on": StreamingMSE(),
        }
        for l in capture_layers
    }

    def _forward_once(*, batch_prep: Dict[str, Any], pruning_mode: str, apply_repair: bool) -> Any:
        inputs = batch_prep["inputs"]
        return model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values", None),
            attention_mask=inputs.get("attention_mask", None),
            vision_start=batch_prep["vision_start"],
            vision_end=batch_prep["vision_end"],
            question_starts=batch_prep["question_starts"],
            question_ends=batch_prep["question_ends"],
            answer_starts=batch_prep["answer_starts"],
            answer_ends=batch_prep["answer_ends"],
            return_pruning_info=False,
            pruning_mode=pruning_mode,
            target_token_num=target_token_num,
            apply_repair=apply_repair,
            capture_layers=capture_layers,
        )

    # Main loop
    used_samples = 0
    for bi, batch in enumerate(batches):
        batch_prep = preprocess_batch(batch, processor, device, max_length=int(args.max_length), mode="train")

        with torch.no_grad():
            out_teacher = _forward_once(batch_prep=batch_prep, pruning_mode=str(teacher_pruning_mode), apply_repair=False)
            out_off = _forward_once(batch_prep=batch_prep, pruning_mode="normal", apply_repair=False)
            out_on = _forward_once(batch_prep=batch_prep, pruning_mode="normal", apply_repair=True)

        cap_teacher = getattr(out_teacher, "captured", None) or {}
        cap_off = (getattr(out_off, "captured_for_repair", None) or getattr(out_off, "captured", None) or {}) or {}
        cap_on = (getattr(out_on, "captured_for_repair", None) or getattr(out_on, "captured", None) or {}) or {}

        bsz = int(batch_prep["inputs"]["input_ids"].shape[0])

        # Store pooled vectors (for PCA/C2ST) + objective aggregation buffers
        for layer in capture_layers:
            if layer not in cap_teacher or layer not in cap_off or layer not in cap_on:
                continue

            t_entry = cap_teacher[layer]
            off_entry = cap_off[layer]
            on_entry = cap_on[layer]

            # pooled vectors (for PCA/C2ST)
            pooled[layer]["teacher"].extend(_extract_pooled_per_sample(t_entry))
            pooled[layer]["off"].extend(_extract_pooled_per_sample(off_entry))
            pooled[layer]["on"].extend(_extract_pooled_per_sample(on_entry))

            # Use a common mask so that teacher/off/on see the same token set.
            m_common = (t_entry["mask"] > 0.5) & (off_entry["mask"] > 0.5) & (on_entry["mask"] > 0.5)

            t_tok = _flatten_capture_tokens(t_entry, mask_override=m_common)
            off_tok = _flatten_capture_tokens(off_entry, mask_override=m_common)
            on_tok = _flatten_capture_tokens(on_entry, mask_override=m_common)

            # Update moments
            moments[layer]["teacher"].update(t_tok)
            moments[layer]["off"].update(off_tok)
            moments[layer]["on"].update(on_tok)

            # token-level (paired) MSE
            token_mse_acc[layer]["off"].update(off_tok, t_tok)
            token_mse_acc[layer]["on"].update(on_tok, t_tok)

        used_samples += bsz
        if (bi + 1) % 10 == 0 or (bi + 1) == len(batches):
            print(f"Processed {min(used_samples, len(samples))}/{len(samples)} samples...")

        # Reduce peak memory
        del out_teacher, out_off, out_on
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ===== Aggregate layerwise objective =====
    rows: List[Dict[str, Any]] = []

    def _compute_mode(layer: int, mode: str, mt: np.ndarray, vt: np.ndarray) -> Tuple[float, float, float, float]:
        if moments[layer][mode].n <= 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        ms, vs = moments[layer][mode].mean_var()
        mean_mse = float(np.mean((ms - mt) ** 2))
        var_mse = float(np.mean((vs - vt) ** 2))
        token_mse = float(token_mse_acc[layer][mode].value())
        if loss_type == "mse":
            total = token_mse
        else:
            total = mean_mse + float(var_weight) * var_mse
        return total, mean_mse, var_mse, token_mse

    for layer in capture_layers:
        if moments[layer]["teacher"].n <= 0 or moments[layer]["off"].n <= 0 or moments[layer]["on"].n <= 0:
            continue
        mt, vt = moments[layer]["teacher"].mean_var()

        total_off, mean_off, var_off, tok_off = _compute_mode(layer, "off", mt, vt)
        total_on, mean_on, var_on, tok_on = _compute_mode(layer, "on", mt, vt)

        rows.append(
            {
                "layer": int(layer),
                "n_tokens": int(moments[layer]["teacher"].n),
                "is_pruning_layer": int(layer in pruning_layers),
                "is_repair_layer": int(layer in repair_layers),
                "total_off": float(total_off),
                "mean_mse_off": float(mean_off),
                "var_mse_off": float(var_off),
                "token_mse_off": float(tok_off),
                "total_on": float(total_on),
                "mean_mse_on": float(mean_on),
                "var_mse_on": float(var_on),
                "token_mse_on": float(tok_on),
                "gain_off_minus_on": float(total_off - total_on),
            }
    )

    rows = sorted(rows, key=lambda r: int(r["layer"]))
    layer_csv = os.path.join(output_dir, "repair_objective_layerwise.csv")
    if not rows:
        raise RuntimeError("No layerwise rows computed. Check capture_layers / dataset preprocessing.")
    with open(layer_csv, "w", newline="", encoding="utf-8") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved CSV: {layer_csv}")

    # Summary numbers (mean over layers; also report repair-layers-only if available)
    key_off, key_on = "total_off", "total_on"

    def _summarize(scope: str) -> Optional[Dict[str, Any]]:
        if scope == "repair_layers":
            scope_rows = [r for r in rows if int(r.get("is_repair_layer", 0)) == 1]
        else:
            scope_rows = list(rows)
        if not scope_rows:
            return None
        off_vals = np.asarray([float(r[key_off]) for r in scope_rows], dtype=np.float64)
        on_vals = np.asarray([float(r[key_on]) for r in scope_rows], dtype=np.float64)
        gain_vals = off_vals - on_vals
        return {
            "metric": "total",
            "layers_scope": scope,
            "mean_off": float(np.nanmean(off_vals)),
            "mean_on": float(np.nanmean(on_vals)),
            "mean_gain_off_minus_on": float(np.nanmean(gain_vals)),
            "loss_type": loss_type,
            "var_weight": float(var_weight),
            "num_samples_used": int(len(samples)),
            "num_layers_used": int(len(scope_rows)),
        }

    summary_rows: List[Dict[str, Any]] = []
    s_all = _summarize("all_layers")
    if s_all is not None:
        summary_rows.append(s_all)
    s_rep = _summarize("repair_layers")
    if s_rep is not None:
        summary_rows.append(s_rep)

    summary_csv = os.path.join(output_dir, "repair_objective_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        import csv

        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved CSV: {summary_csv}")

    if s_all is not None:
        print(f"\nMean total (OFF): {s_all['mean_off']:.6f}")
        print(f"Mean total (ON ): {s_all['mean_on']:.6f}")
        print(f"Mean gain (OFF-ON, +good): {s_all['mean_gain_off_minus_on']:.6f}")
    if s_rep is not None:
        print(f"\nMean total (OFF, repair layers): {s_rep['mean_off']:.6f}")
        print(f"Mean total (ON , repair layers): {s_rep['mean_on']:.6f}")
        print(f"Mean gain (OFF-ON, repair layers): {s_rep['mean_gain_off_minus_on']:.6f}")

    # Repair-layer table (OFF vs ON), derived from the layerwise table
    row_by_layer = {int(r["layer"]): r for r in rows}
    repair_rows: List[Dict[str, Any]] = []
    for layer in repair_layers:
        r = row_by_layer.get(int(layer), None)
        if r is None:
            continue
        repair_rows.append(
            {
                "repair_layer": int(layer),
                "n_tokens": int(r.get("n_tokens", 0)),
                "total_off": float(r["total_off"]),
                "total_on": float(r["total_on"]),
                "gain_off_minus_on": float(r["gain_off_minus_on"]),
            }
        )
    repair_csv = os.path.join(output_dir, "repair_layers_effect.csv")
    if repair_rows:
        with open(repair_csv, "w", newline="", encoding="utf-8") as f:
            import csv

            w = csv.DictWriter(f, fieldnames=list(repair_rows[0].keys()))
            w.writeheader()
            w.writerows(repair_rows)
        print(f"Saved CSV: {repair_csv}")

    # ===== Figures =====
    plot_rows = [r for r in rows if int(r["layer"]) not in excluded_layers]
    if not plot_rows:
        raise RuntimeError("All layers excluded from figures. Check --exclude_layers / --exclude_last_layer.")

    layers = [int(r["layer"]) for r in plot_rows]
    off_curve = np.asarray([float(r["total_off"]) for r in plot_rows], dtype=np.float64)
    on_curve = np.asarray([float(r["total_on"]) for r in plot_rows], dtype=np.float64)

    gain_off_on_curve = off_curve - on_curve

    def _vlines(ax):
        for l in pruning_layers:
            ax.axvline(int(l), color="k", linestyle="--", alpha=0.18, linewidth=1.0)
        for l in repair_layers:
            ax.axvline(int(l), color="tab:green", linestyle="--", alpha=0.28, linewidth=1.2)

    # Fig: total objective curves (log-scale + linear-scale)
    for yscale in ("linear", "log"):
        fig, ax = plt.subplots(figsize=(width_in, base_h_in))
        ax.plot(layers, off_curve, marker="o", color="tab:red", label="OFF (no repair)")
        ax.plot(layers, on_curve, marker="o", color="tab:blue", label="ON (repair)")

        _vlines(ax)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Repair objective (total)")
        ax.set_title(f"Layerwise gap to teacher on gen_answer tokens ({loss_type}, var_w={var_weight:g})")
        ax.legend(frameon=False, ncol=1)
        if yscale == "log":
            ax.set_yscale("log")
        _save_fig(fig, output_dir, f"fig_repair_objective_layerwise_{yscale}")

    # Fig: gains
    fig, ax = plt.subplots(figsize=(width_in, base_h_in))
    ax.plot(layers, gain_off_on_curve, marker="o", color="tab:purple", label="Gain = OFF - ON (global)")
    ax.axhline(0.0, color="k", linewidth=1.0, alpha=0.4)
    _vlines(ax)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Gain (higher is better)")
    ax.set_title("Repair gains across layers")
    ax.legend(frameon=False, ncol=1)
    _save_fig(fig, output_dir, "fig_repair_gain_layerwise")

    # Fig: decomposition (mean + var)
    mean_off = np.asarray([float(r["mean_mse_off"]) for r in plot_rows], dtype=np.float64)
    mean_on = np.asarray([float(r["mean_mse_on"]) for r in plot_rows], dtype=np.float64)
    var_off = np.asarray([float(r["var_mse_off"]) for r in plot_rows], dtype=np.float64) * float(var_weight)
    var_on = np.asarray([float(r["var_mse_on"]) for r in plot_rows], dtype=np.float64) * float(var_weight)
    fig, axes = plt.subplots(2, 1, figsize=(width_in, base_h_in * 1.6), sharex=True)
    axes[0].plot(layers, mean_off, marker="o", color="tab:red", label="OFF mean_mse")
    axes[0].plot(layers, mean_on, marker="o", color="tab:blue", label="ON mean_mse")
    axes[0].set_ylabel("mean_mse")
    axes[0].legend(frameon=False)
    axes[1].plot(layers, var_off, marker="o", color="tab:red", label=f"OFF {var_weight:g}×var_mse")
    axes[1].plot(layers, var_on, marker="o", color="tab:blue", label=f"ON {var_weight:g}×var_mse")
    axes[1].set_ylabel(f"{var_weight:g}×var_mse")
    axes[1].set_xlabel("Layer")
    for ax in axes:
        _vlines(ax)
    axes[0].set_title("Repair objective decomposition (mean / var)")
    _save_fig(fig, output_dir, "fig_repair_objective_decomposition")

    # Fig: repair layers effect (OFF vs ON)
    if repair_rows:
        plot_repair_rows = [r for r in repair_rows if int(r["repair_layer"]) not in excluded_layers]
        xs = [int(r["repair_layer"]) for r in plot_repair_rows]
        if xs:
            off_y = np.asarray([float(r["total_off"]) for r in plot_repair_rows], dtype=np.float64)
            on_y = np.asarray([float(r["total_on"]) for r in plot_repair_rows], dtype=np.float64)
            gain_y = np.asarray([float(r["gain_off_minus_on"]) for r in plot_repair_rows], dtype=np.float64)
            x = np.arange(len(xs))

            fig, axes = plt.subplots(1, 2, figsize=(width_in, base_h_in), gridspec_kw={"width_ratios": [1.2, 1.0]})
            w = 0.36
            axes[0].bar(x - w / 2, off_y, width=w, label="OFF", color="tab:red", alpha=0.9)
            axes[0].bar(x + w / 2, on_y, width=w, label="ON", color="tab:blue", alpha=0.9)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([str(v) for v in xs])
            axes[0].set_xlabel("Repair layer")
            axes[0].set_ylabel("Objective (total)")
            axes[0].set_title("Repair layers: OFF vs ON")
            axes[0].legend(frameon=False)

            axes[1].bar(x, gain_y, width=0.55, color="tab:green", alpha=0.9)
            axes[1].axhline(0.0, color="k", linewidth=1.0, alpha=0.4)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([str(v) for v in xs])
            axes[1].set_xlabel("Repair layer")
            axes[1].set_ylabel("Gain (OFF - ON)")
            axes[1].set_title("Gain at repair layers")

            _save_fig(fig, output_dir, "fig_repair_layers_offon")

    # ===== LDA C2ST (per-layer) =====
    if not args.no_lda:
        lda_rows: List[Dict[str, Any]] = []
        for layer in capture_layers:
            if layer not in pooled:
                continue
            if not pooled[layer]["teacher"] or not pooled[layer]["off"] or not pooled[layer]["on"]:
                continue
            Xt = np.stack(pooled[layer]["teacher"], axis=0)
            Xoff = np.stack(pooled[layer]["off"], axis=0)
            Xon = np.stack(pooled[layer]["on"], axis=0)
            n = min(len(Xt), len(Xoff), len(Xon))
            Xt = Xt[:n]
            Xoff = Xoff[:n]
            Xon = Xon[:n]
            res_off = _lda_c2st_auc(Xt, Xoff, seed=int(args.seed))
            res_on = _lda_c2st_auc(Xt, Xon, seed=int(args.seed))
            lda_rows.append(
                {
                    "layer": int(layer),
                    "auc_teacher_vs_off": float(res_off["auc"]),
                    "acc_teacher_vs_off": float(res_off["acc"]),
                    "auc_teacher_vs_on": float(res_on["auc"]),
                    "acc_teacher_vs_on": float(res_on["acc"]),
                    "delta_auc_off_minus_on": float(res_off["auc"] - res_on["auc"]),
                    "n_per_group": int(res_off["n"]),
                }
            )

        if lda_rows:
            lda_csv = os.path.join(output_dir, "lda_layerwise.csv")
            with open(lda_csv, "w", newline="", encoding="utf-8") as f:
                import csv

                w = csv.DictWriter(f, fieldnames=list(lda_rows[0].keys()))
                w.writeheader()
                w.writerows(lda_rows)
            print(f"Saved CSV: {lda_csv}")

            # curve plot (exclude noisy layers if requested)
            lda_plot = [r for r in lda_rows if int(r["layer"]) not in excluded_layers]
            lda_plot = [
                r
                for r in lda_plot
                if (not math.isnan(float(r["auc_teacher_vs_off"]))) and (not math.isnan(float(r["auc_teacher_vs_on"])))
            ]
            if lda_plot:
                xs = [int(r["layer"]) for r in lda_plot]
                auc_off = [float(r["auc_teacher_vs_off"]) for r in lda_plot]
                auc_on = [float(r["auc_teacher_vs_on"]) for r in lda_plot]
                fig, ax = plt.subplots(figsize=(width_in, base_h_in))
                ax.plot(xs, auc_off, marker="o", color="tab:red", label="Teacher vs OFF")
                ax.plot(xs, auc_on, marker="o", color="tab:blue", label="Teacher vs ON")
                ax.axhline(0.5, color="k", linewidth=1.0, alpha=0.4)
                _vlines(ax)
                ax.set_xlabel("Layer")
                ax.set_ylabel("LDA C2ST AUC (0.5=indistinguishable)")
                ax.set_title("Per-layer distribution separability (LDA, pooled)")
                ax.legend(frameon=False)
                _save_fig(fig, output_dir, "fig_lda_auc_layerwise")

    # ===== PCA + C2ST on selected layers =====
    # choose layers: repair_layers + topk global gain
    selected_layers: List[int] = []
    selected_set = set()
    for l in repair_layers:
        if int(l) in excluded_layers:
            continue
        if l in pooled:
            selected_set.add(int(l))
    # top-k by mean gain OFF-ON
    by_gain = sorted(plot_rows, key=lambda r: float(r["gain_off_minus_on"]), reverse=True)
    for r in by_gain:
        if len(selected_set) >= int(args.pca_max_layers):
            break
        layer = int(r["layer"])
        if layer in excluded_layers:
            continue
        if layer in selected_set:
            continue
        if int(args.pca_topk) <= 0:
            break
        selected_set.add(layer)
        if len([x for x in selected_set if x not in set(repair_layers)]) >= int(args.pca_topk):
            break
    selected_layers = sorted(selected_set)

    # PCA scatters
    if (not args.no_pca) and selected_layers:
        for layer in selected_layers:
            Xt = np.stack(pooled[layer]["teacher"], axis=0)
            Xoff = np.stack(pooled[layer]["off"], axis=0)
            Xon = np.stack(pooled[layer]["on"], axis=0)
            n = min(len(Xt), len(Xoff), len(Xon))
            Xt = Xt[:n]
            Xoff = Xoff[:n]
            Xon = Xon[:n]
            X = np.concatenate([Xt, Xoff, Xon], axis=0)
            pca = PCA(n_components=2, random_state=int(args.seed))
            Z = pca.fit_transform(X)
            Zt = Z[:n]
            Zoff = Z[n : 2 * n]
            Zon = Z[2 * n : 3 * n]

            fig, ax = plt.subplots(figsize=(width_in, base_h_in))
            ax.scatter(Zt[:, 0], Zt[:, 1], s=14, alpha=0.75, label="Teacher (keep_all)", color="k")
            ax.scatter(Zoff[:, 0], Zoff[:, 1], s=14, alpha=0.55, label="Student OFF", color="tab:red")
            ax.scatter(Zon[:, 0], Zon[:, 1], s=14, alpha=0.55, label="Student ON", color="tab:blue")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            evr = pca.explained_variance_ratio_
            ax.set_title(f"PCA (pooled) @ layer {layer} | EVR={evr[0]:.2f},{evr[1]:.2f}")
            ax.legend(frameon=False, loc="best")
            ax.grid(True, alpha=0.25)
            _save_fig(fig, output_dir, f"fig_pca_pooled_layer{layer:02d}")

    # C2ST AUC
    if (not args.no_c2st) and selected_layers:
        c2st_rows = []
        for layer in selected_layers:
            Xt = np.stack(pooled[layer]["teacher"], axis=0)
            Xoff = np.stack(pooled[layer]["off"], axis=0)
            Xon = np.stack(pooled[layer]["on"], axis=0)
            res_off = _c2st_auc(Xt, Xoff, seed=int(args.seed))
            res_on = _c2st_auc(Xt, Xon, seed=int(args.seed))
            c2st_rows.append(
                {
                    "layer": int(layer),
                    "auc_teacher_vs_off": float(res_off["auc"]),
                    "acc_teacher_vs_off": float(res_off["acc"]),
                    "auc_teacher_vs_on": float(res_on["auc"]),
                    "acc_teacher_vs_on": float(res_on["acc"]),
                    "delta_auc_off_minus_on": float(res_off["auc"] - res_on["auc"]),
                    "n_per_group": int(res_off["n"]),
                }
            )

        c2st_csv = os.path.join(output_dir, "c2st_auc_selected_layers.csv")
        with open(c2st_csv, "w", newline="", encoding="utf-8") as f:
            import csv

            w = csv.DictWriter(f, fieldnames=list(c2st_rows[0].keys()))
            w.writeheader()
            w.writerows(c2st_rows)
        print(f"Saved CSV: {c2st_csv}")

        # bar plot
        xs = [int(r["layer"]) for r in c2st_rows]
        auc_off = [float(r["auc_teacher_vs_off"]) for r in c2st_rows]
        auc_on = [float(r["auc_teacher_vs_on"]) for r in c2st_rows]
        x = np.arange(len(xs))
        wbar = 0.38
        fig, ax = plt.subplots(figsize=(width_in, base_h_in))
        ax.bar(x - wbar / 2, auc_off, width=wbar, label="Teacher vs OFF", color="tab:red", alpha=0.9)
        ax.bar(x + wbar / 2, auc_on, width=wbar, label="Teacher vs ON", color="tab:blue", alpha=0.9)
        ax.axhline(0.5, color="k", linewidth=1.0, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in xs])
        ax.set_xlabel("Layer")
        ax.set_ylabel("C2ST AUC (0.5=indistinguishable)")
        ax.set_title("Distribution closeness test (pooled)")
        ax.legend(frameon=False)
        _save_fig(fig, output_dir, "fig_c2st_auc_selected_layers")

    # ===== Doc =====
    doc_path = os.path.join(output_dir, "PAPER_VISUALS.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("# Repair Objective Visualizations (Paper-ready)\n\n")
        f.write("This folder is produced by `scripts/analyze_repair_objective.py`.\n\n")
        f.write("## What is measured?\n")
        f.write(
            "- **Token region:** `gen_answer` tokens (teacher-forcing). In `PrunableLlavaForConditionalGeneration`, "
            "`gen_answer positions = [answer_start-1, answer_end-1)` (see `method/models/prunable_llava.py`).\n"
        )
        f.write(
            "- **Teacher:** same checkpoint weights, but **keep_all pruning** and `apply_repair=False`.\n"
            "- **Student OFF:** normal pruning, `apply_repair=False`.\n"
            "- **Student ON:** normal pruning, `apply_repair=True`.\n"
        )
        f.write("\n### Repair objective (same as training)\n")
        f.write(
            "- For each layer, collect teacher/student hidden states on gen_answer tokens, flatten valid tokens "
            "by mask, then compute:\n"
            "  - `mean_mse = MSE(mean(student), mean(teacher))`\n"
            "  - `var_mse  = MSE(var(student),  var(teacher))` (diag variance, unbiased=False)\n"
        )
        f.write(f"- If `loss_type = mean_var`: `total = mean_mse + {var_weight:g} * var_mse`.\n")
        f.write("- If `loss_type = mse`: `total = token_mse` (point-to-point MSE after padding).\n")
        f.write("\n## Tables\n")
        f.write("- `repair_objective_layerwise.csv`: per-layer objective (OFF/ON) + gains + decomposition.\n")
        f.write("- `repair_objective_summary.csv`: mean total (OFF/ON) over layers (also repair-layers-only).\n")
        if repair_rows:
            f.write("- `repair_layers_effect.csv`: repair layers only, OFF vs ON + gain.\n")
        if not args.no_lda:
            f.write("- `lda_layerwise.csv`: per-layer LDA C2ST metrics (Teacher vs OFF/ON).\n")
        if (not args.no_c2st) and selected_layers:
            f.write("- `c2st_auc_selected_layers.csv`: C2ST AUC on pooled vectors for selected layers.\n")
        f.write("\n## Figures (PDF + PNG)\n")
        f.write("- `fig_repair_objective_layerwise_linear.*`: layerwise `total` (linear y-axis).\n")
        f.write("- `fig_repair_objective_layerwise_log.*`: layerwise `total` (log y-axis; easier to see large reduction).\n")
        f.write("- `fig_repair_gain_layerwise.*`: gains across layers (OFF-ON).\n")
        f.write("- `fig_repair_objective_decomposition.*`: mean vs var components across layers.\n")
        if repair_rows:
            f.write("- `fig_repair_layers_offon.*`: repair layers only, OFF vs ON + gain bars.\n")
        if not args.no_lda:
            f.write("- `fig_lda_auc_layerwise.*`: per-layer LDA C2ST AUC curve (Teacher vs OFF/ON).\n")
        if (not args.no_pca) and selected_layers:
            f.write("- `fig_pca_pooled_layerXX.*`: PCA scatter (pooled per-sample vectors) at selected layers.\n")
        if (not args.no_c2st) and selected_layers:
            f.write("- `fig_c2st_auc_selected_layers.*`: C2ST AUC bars (Teacher vs OFF/ON).\n")
        f.write("\n## How to read the plots\n")
        f.write("- Lower `total` means student distribution is closer to teacher on gen_answer tokens.\n")
        f.write("- Positive `gain = OFF - ON` means repair helps.\n")
        f.write("\n## Notes / Caveats\n")
        f.write(
            "- This evaluation is **teacher-forcing** (uses ground-truth answer tokens), aligned with the training objective. "
            "It does not measure free generation quality directly.\n"
        )
        if excluded_layers:
            f.write(f"- Figures exclude layers: {sorted([int(x) for x in excluded_layers])} (CSV still contains all layers).\n")
        f.write(
            "- If the checkpoint has no repair weights, `ON` will be identical to `OFF` (no effect expected).\n"
        )
        f.write(
            "- Vertical markers in layerwise plots:\n"
            "  - black dashed: pruning layers\n"
            "  - green dashed: repair layers (inferred from checkpoint if available)\n"
        )
        f.write("\n## Repro\n")
        f.write("See `meta.json` for the full command + settings used.\n")
    print(f"Saved doc: {doc_path}")

    print(f"\nDone. All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
