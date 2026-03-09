#!/usr/bin/env python
"""Visualize binary vision-token pruning masks overlaid on the (processed) input image.

Outputs:
  - input_processed.png: the exact image fed to the vision encoder (decoded from pixel_values)
  - Lxx_keep.png / Lxx_prune.png: per-layer binary overlays
  - overview.png / overview.pdf:
      - single sample: a single-row panel (Input + pruned overlays)
      - multi samples: a grid panel (one row per sample/group)

This script is intentionally "batteries-included" and defaults to a known-good checkpoint.
You can still override paths via CLI if needed.
"""

from __future__ import annotations

import os

# NOTE: Keep env vars before importing transformers/torch to avoid accidental online downloads.
os.environ.setdefault("HF_HOME", "/data/users/zjw/huggingface_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import ast
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CHECKPOINT = (
    # "outputs/tasks/20260302-0000_vqa-vqav2_llava157b_acea/checkpoints/checkpoint_final.pt"
    "outputs/tasks/20260309-0032_vqa-vqav2_llava157b_084e/checkpoints/checkpoint_final.pt"
)
DEFAULT_CONFIG = "configs/vision_token_pruning.yaml"


def _parse_value(s: str) -> Any:
    s = s.strip()
    if s.lower() in {"none", "null"}:
        return None
    if s in {"True", "False"}:
        return s == "True"
    # numeric / list / dict / quoted string
    if s[:1] in {"[", "{", "(", "'", '"'} or s[:1].isdigit() or s[:1] == "-":
        try:
            return ast.literal_eval(s)
        except Exception:
            pass
    return s


def _parse_log_section_kv(log_path: Path, section_name: str) -> Dict[str, Any]:
    """Parse a '[Section Name]' block in our training logs into a dict."""
    lines = log_path.read_text(errors="ignore").splitlines()
    header = f"[{section_name}]"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            start = i + 1
            break
    if start is None:
        return {}

    out: Dict[str, Any] = {}
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            # blank line ends the section in our log format
            if out:
                break
            continue
        if stripped.startswith("[") and stripped.endswith("]") and stripped != header:
            break
        # skip separators
        if set(stripped) <= {"-"}:
            continue
        # parse "  key: value"
        if ":" not in stripped:
            continue
        key, val = stripped.split(":", 1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        out[key] = _parse_value(val)
    return out


def _find_task_log_from_checkpoint(checkpoint_path: Path) -> Optional[Path]:
    """Given .../tasks/<tag>/checkpoints/*.pt -> find .../tasks/<tag>/logs/*.log."""
    ckpt = checkpoint_path.resolve()
    if ckpt.parent.name != "checkpoints":
        return None
    task_dir = ckpt.parent.parent
    log_dir = task_dir / "logs"
    if not log_dir.is_dir():
        return None
    logs = sorted(log_dir.glob("*.log"))
    if not logs:
        return None
    # Usually there is exactly one log file; pick the largest as a heuristic.
    logs.sort(key=lambda p: p.stat().st_size, reverse=True)
    return logs[0]


def _build_config(
    config_path: str,
    checkpoint_path: Path,
    split: str,
    load_limit: int,
    force_backbone_name: Optional[str],
) -> Any:
    from engine.configs.loader import load_config

    overrides: Dict[str, Any] = {
        "config_settings": {"log_config_on_load": False},
        # speed: vqa-v2 supports fast_load_no_random, and we only need a small subset for visualization
        "dataset_settings": {"fast_load_no_random": True},
    }

    # If this checkpoint lives inside outputs/tasks/<tag>/..., parse the real settings from its log
    log_path = _find_task_log_from_checkpoint(checkpoint_path)
    if log_path is not None and log_path.exists():
        backbone_kv = _parse_log_section_kv(log_path, "Backbone Settings")
        method_kv = _parse_log_section_kv(log_path, "Method Settings")
        if backbone_kv:
            overrides.setdefault("backbone_settings", {})
            overrides["backbone_settings"].update({k: backbone_kv[k] for k in ("name",) if k in backbone_kv})
        if method_kv:
            overrides.setdefault("method_settings", {})
            # Keep all parsed keys; they are already in the expected types for most entries.
            overrides["method_settings"].update(method_kv)

    # Optional: force backbone name (useful if you move checkpoints around)
    if force_backbone_name:
        overrides.setdefault("backbone_settings", {})
        overrides["backbone_settings"]["name"] = str(force_backbone_name)

    config = load_config(override_file=config_path, override_dict=overrides, skip_auto_paths=True)

    # IMPORTANT: Ensure we only load the requested split (do not load train by accident).
    # load_config merges dicts and won't delete other splits from YAML, so we override after loading.
    config.dataset_settings["split"] = {str(split): max(int(load_limit), 1)}
    config.dataset_settings["fast_load_no_random"] = True

    # make sure config knows which checkpoint we loaded (purely informative)
    config.global_settings["checkpoint"] = str(checkpoint_path)
    return config


def _resolve_model_repo(backbone_name: str) -> str:
    mapping = {
        "llava-1.5-7b": "llava-hf/llava-1.5-7b-hf",
        "llava-1.5-13b": "llava-hf/llava-1.5-13b-hf",
    }
    return mapping.get(backbone_name, backbone_name)


def _dtype_from_str(s: str) -> torch.dtype:
    s = str(s or "float32").lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    return mapping.get(s, torch.float32)


def _pixel_values_to_pil(pixel_values: torch.Tensor, processor) -> Image.Image:
    """Decode (3,H,W) pixel_values back to an RGB image for visualization.

    This reconstructs the *processed* image (after resize/crop + normalization) so the 24x24 token grid aligns.
    """
    if pixel_values.dim() != 3 or pixel_values.shape[0] != 3:
        raise ValueError(f"pixel_values must be (3,H,W), got {tuple(pixel_values.shape)}")

    ip = getattr(processor, "image_processor", None)
    if ip is None:
        raise RuntimeError("processor has no image_processor; cannot decode pixel_values.")

    mean = torch.tensor(ip.image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(ip.image_std, dtype=torch.float32).view(3, 1, 1)
    x = pixel_values.detach().float().cpu()
    x = (x * std) + mean
    x = x.clamp(0.0, 1.0)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _overlay_mask(
    base_img: Image.Image,
    mask_grid: np.ndarray,
    color_rgb: Tuple[int, int, int],
    alpha: float = 0.38,
) -> Image.Image:
    """Overlay a binary (Hgrid,Wgrid) mask on an RGB image with nearest upsampling."""
    if base_img.mode != "RGB":
        base_img = base_img.convert("RGB")

    if mask_grid.ndim != 2:
        raise ValueError(f"mask_grid must be 2D, got shape {mask_grid.shape}")
    if mask_grid.dtype != np.float32 and mask_grid.dtype != np.float64:
        mask_grid = mask_grid.astype(np.float32)
    mask_grid = (mask_grid > 0.5).astype(np.uint8) * 255

    w, h = base_img.size
    mask_up = Image.fromarray(mask_grid, mode="L").resize((w, h), resample=Image.NEAREST)
    mask = (np.asarray(mask_up).astype(np.float32) / 255.0)[..., None]  # (H,W,1)

    base = np.asarray(base_img).astype(np.float32)
    overlay = np.zeros_like(base)
    overlay[..., 0] = color_rgb[0]
    overlay[..., 1] = color_rgb[1]
    overlay[..., 2] = color_rgb[2]

    a = float(alpha)
    out = base * (1.0 - a * mask) + overlay * (a * mask)
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def _mask_to_grid(mask_1d: np.ndarray) -> np.ndarray:
    mask_1d = np.asarray(mask_1d).reshape(-1)
    n = int(mask_1d.shape[0])
    side = int(math.isqrt(n))
    if side * side != n:
        raise ValueError(f"n_vision must be a square number, got n={n}")
    return mask_1d.reshape(side, side)


def _load_model_and_processor(config, checkpoint_path: Path, device: torch.device):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration

    method_cfg = config.method_settings
    backbone_cfg = config.backbone_settings
    global_cfg = config.global_settings

    torch_dtype = _dtype_from_str(global_cfg.get("dtype", "bfloat16"))
    backbone_name = backbone_cfg.get("name", "llava-1.5-7b")
    model_repo = _resolve_model_repo(backbone_name)

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_repo,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(model_repo, local_files_only=True)
    processor.tokenizer.padding_side = "right"
    base_model.processor = processor

    # Prefer eval_* knobs if present (these are what we use in analysis scripts too).
    eval_temp = float(method_cfg.get("eval_temperature", method_cfg.get("temperature", 0.2)))
    eval_thr = float(method_cfg.get("eval_pruning_threshold", method_cfg.get("pruning_threshold", 0.5)))

    # For visualization we want deterministic binary masks.
    use_gumbel_noise = False

    model = PrunableLlavaForConditionalGeneration(
        base_model=base_model,
        pruning_layers=list(method_cfg.get("pruning_layers", [4, 14, 24])),
        pruner_d_internal=int(method_cfg.get("pruner_d_internal", 512)),
        pruner_n_heads=int(method_cfg.get("pruner_n_heads", 4)),
        pruner_n_queries=int(method_cfg.get("pruner_n_queries", 16)),
        pruner_query_dropout=float(method_cfg.get("pruner_query_dropout", 0.0)),
        use_question_condition=bool(method_cfg.get("use_question_condition", False)),
        disc_d_hidden=int(method_cfg.get("disc_d_d", 256)),
        temperature=eval_temp,
        dropout=float(method_cfg.get("pruner_dropout", 0.0)),
        disc_use_spectral_norm=bool(method_cfg.get("disc_use_spectral_norm", False)),
        use_gumbel_noise=use_gumbel_noise,
        pruning_threshold=eval_thr,
        # repair adapter is irrelevant for pruning visualization (does not affect q2v in causal LM),
        # but we still construct it if the checkpoint provides weights.
        use_repair_adapter=bool(method_cfg.get("use_repair_adapter", False)),
        repair_layers=method_cfg.get("repair_layers", None),
        repair_source_layers=method_cfg.get("repair_source_layers", None),
        repair_bottleneck_dim=int(method_cfg.get("repair_bottleneck_dim", 512)),
        repair_dropout=float(method_cfg.get("repair_dropout", 0.0)),
        repair_mask_encoder_type=str(method_cfg.get("repair_mask_encoder_type", "attention")),
        repair_use_pruned_info=bool(method_cfg.get("repair_use_pruned_info", True)),
        repair_alpha_init=float(method_cfg.get("repair_alpha_init", 0.1)),
        repair_detach_input=bool(method_cfg.get("repair_detach_input", True)),
    )
    model.freeze_base_model()
    model.eval()
    model.set_temperature(eval_temp)
    model.set_pruning_threshold(eval_thr)
    model.set_use_gumbel_noise(False)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if "pruner_state_dict" in ckpt:
        model.pruner_manager.load_state_dict(ckpt["pruner_state_dict"])

    # Optional: load repair weights if present and model actually has the modules.
    if bool(getattr(model, "use_repair_adapter", False)):
        if ("repair_context_encoder_state_dict" in ckpt) and (getattr(model, "repair_context_encoder", None) is not None):
            model.repair_context_encoder.load_state_dict(ckpt["repair_context_encoder_state_dict"])
        if ("repair_adapter_state_dict" in ckpt) and (getattr(model, "repair_adapter_manager", None) is not None):
            model.repair_adapter_manager.load_state_dict(ckpt["repair_adapter_state_dict"])

    return model, processor


def _image_key(sample: Dict[str, Any]) -> Tuple[str, Any]:
    """Return a stable key to identify the underlying image (not the question)."""
    if "image_id" in sample:
        return ("image_id", sample["image_id"])
    if "image" in sample:
        return ("image", sample["image"])
    return ("unknown", None)


def _select_unique_sample_idxs(
    samples: List[Dict[str, Any]],
    requested_idxs: List[int],
    desired_count: int,
    *,
    search_start: int,
    max_search: int,
) -> Tuple[List[int], List[int]]:
    """Select indices with unique images; duplicates are replaced by later samples.

    Returns:
      selected_idxs, skipped_duplicate_idxs
    """
    n_total = len(samples)
    desired_count = max(int(desired_count), 0)
    max_search = max(int(max_search), 0)
    requested_idxs = [int(x) for x in requested_idxs]

    seen = set()
    selected: List[int] = []
    skipped: List[int] = []

    def _try_add(i: int) -> bool:
        if i < 0 or i >= n_total:
            return False
        key = _image_key(samples[i])
        if key in seen:
            return False
        seen.add(key)
        selected.append(i)
        return True

    for i in requested_idxs:
        if len(selected) >= desired_count:
            break
        if not _try_add(i):
            skipped.append(i)

    if len(selected) < desired_count:
        end = min(n_total, int(search_start) + max_search)
        for i in range(int(search_start), end):
            if len(selected) >= desired_count:
                break
            if i in requested_idxs:
                continue
            _try_add(i)

    return selected, skipped


def _save_overview(
    out_dir: Path,
    panels: List[Tuple[str, Image.Image]],
    n_layers: int,
) -> None:
    import matplotlib.pyplot as plt

    # panels: [("Input", img), ("L4", img), ("L14", img), ("L24", img)]
    # ECCV-ish compact figure defaults: clean, tight, no heavy fonts.
    plt.rcParams.update(
        {
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "figure.dpi": 200,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            # Make PDF text editable in most pipelines.
            "pdf.fonttype": 42,
        }
    )

    ncols = 1 + n_layers
    fig_w = 1.85 * ncols
    fig_h = 2.05
    fig, axes = plt.subplots(1, ncols, figsize=(fig_w, fig_h))
    if ncols == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_axis_off()
        if title:
            ax.set_title(title, pad=1.5)

    fig.subplots_adjust(left=0.0, right=1.0, top=0.92, bottom=0.0, wspace=0.02)
    fig.savefig(out_dir / "overview.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / "overview.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def _save_overview_grid(
    out_dir: Path,
    overview_rows: List[List[Tuple[str, Image.Image]]],
    n_layers: int,
    row_labels: Optional[List[str]] = None,
) -> None:
    """Save a multi-sample overview panel: one row per sample/group."""
    import matplotlib.pyplot as plt

    if not overview_rows:
        raise ValueError("overview_rows must be non-empty.")

    ncols = 1 + int(n_layers)
    nrows = int(len(overview_rows))

    for r, row in enumerate(overview_rows):
        if len(row) != ncols:
            raise ValueError(f"Row {r} has {len(row)} panels, expected {ncols}.")

    if row_labels is not None and len(row_labels) != nrows:
        raise ValueError(f"row_labels has {len(row_labels)} items, expected {nrows}.")

    # ECCV-ish compact figure defaults: clean, tight, no heavy fonts.
    plt.rcParams.update(
        {
            "font.size": 7.6,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.6,
            "figure.dpi": 200,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )

    cell_w = 1.85
    cell_h = 1.85
    fig_w = cell_w * ncols
    fig_h = cell_h * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    if nrows == 1:
        axes = np.asarray([axes])

    # Use first row titles as column titles; do not repeat per row.
    col_titles = [t for (t, _) in overview_rows[0]]

    for r in range(nrows):
        for c in range(ncols):
            title, img = overview_rows[r][c]
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_axis_off()

            if r == 0:
                ax.set_title(col_titles[c], pad=1.5)

            if c == 0 and row_labels is not None:
                # Put row label slightly outside the first column.
                ax.text(
                    -0.03,
                    0.5,
                    str(row_labels[r]),
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                )

    # Tight, paper-friendly layout
    left = 0.02 if row_labels is not None else 0.0
    fig.subplots_adjust(left=left, right=1.0, top=0.965, bottom=0.0, wspace=0.02, hspace=0.02)
    fig.savefig(out_dir / "overview.png", dpi=300, bbox_inches="tight", pad_inches=0.01)
    fig.savefig(out_dir / "overview.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    p.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument(
        "--sample_idxs",
        type=str,
        default="",
        help="Comma-separated sample indices (e.g. '0,1,2'). If set, overrides --sample_idx/--num_samples.",
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of consecutive samples to visualize starting from --sample_idx (ignored if --sample_idxs is set).",
    )
    p.add_argument(
        "--unique_images",
        action="store_true",
        help="Ensure each row uses a unique image (by image_id if available). Duplicates will be replaced by later samples.",
    )
    p.add_argument(
        "--max_search",
        type=int,
        default=200,
        help="When --unique_images is set, search up to this many additional samples to fill unique images.",
    )
    p.add_argument(
        "--layers",
        type=str,
        default="",
        help="Comma-separated pruning layers to visualize (e.g. '14,24'). Empty = use config pruning_layers (default: all).",
    )
    p.add_argument(
        "--skip_first_pruner",
        action="store_true",
        help="Skip the first pruning stage (paper-style, e.g. only show L14/L24 when pruning_layers=[4,14,24]).",
    )
    # Backward-compatible alias (kept intentionally; default behavior already shows all pruners).
    p.add_argument("--all_pruners", action="store_true", help="Alias: visualize all pruning_layers (default).")
    p.add_argument("--alpha", type=float, default=0.38, help="Overlay alpha for highlighted patches.")
    p.add_argument("--scale", type=int, default=4, help="Upscale factor for saved images (paper-friendly). Use 1 to disable.")
    p.add_argument("--out_dir", type=str, default="outputs/visualizations/pruning_overlays")
    p.add_argument("--force_backbone", type=str, default="", help="Force backbone_settings.name if log parsing is unavailable.")
    return p.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Determine sample indices to visualize.
    if args.sample_idxs.strip():
        if int(args.num_samples) != 1:
            print("[Note] --num_samples is ignored because --sample_idxs is set.")
        sample_idxs = [int(x.strip()) for x in args.sample_idxs.split(",") if x.strip()]
    else:
        start = int(args.sample_idx)
        n = max(int(args.num_samples), 1)
        sample_idxs = list(range(start, start + n))
    if not sample_idxs:
        raise ValueError("No sample indices provided.")

    base_max = max(int(x) for x in sample_idxs) + 1
    load_limit = base_max
    if args.unique_images and len(sample_idxs) > 1:
        load_limit = base_max + max(int(args.max_search), 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = _build_config(
        config_path=str(args.config),
        checkpoint_path=checkpoint_path,
        split=str(args.split),
        load_limit=load_limit,
        force_backbone_name=(args.force_backbone or None),
    )

    model, processor = _load_model_and_processor(config, checkpoint_path, device=device)

    # Load dataset once (avoid reshuffling/reloading per sample).
    from engine.datas.loader import load_dataset

    bundle = load_dataset(config)
    ds = bundle["splits"][str(args.split)]

    # Optional: enforce unique images (VQAv2 has multiple questions per image).
    if args.unique_images and len(sample_idxs) > 1:
        selected_idxs, skipped = _select_unique_sample_idxs(
            samples=ds.samples,
            requested_idxs=sample_idxs,
            desired_count=len(sample_idxs),
            search_start=base_max,
            max_search=int(args.max_search),
        )
        if skipped:
            skipped_str = ", ".join(str(i) for i in skipped[:20])
            print(
                f"[Note] Duplicate images detected in requested idxs; skipped: {skipped_str}"
                + (" ..." if len(skipped) > 20 else "")
            )
        if len(selected_idxs) < len(sample_idxs):
            print(
                f"[Warn] Only found {len(selected_idxs)}/{len(sample_idxs)} unique images within the search window. "
                f"Try increasing --max_search."
            )
        sample_idxs = selected_idxs

    # Pick layers to visualize
    if args.layers.strip():
        layers_to_viz = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    else:
        layers_to_viz = list(config.method_settings.get("pruning_layers", []))
        if args.skip_first_pruner:
            # Paper-style: often only show the last two pruning stages (e.g. L14/L24).
            layers_to_viz = layers_to_viz[1:]

    layers_to_viz = sorted({int(x) for x in layers_to_viz})
    if not layers_to_viz:
        raise ValueError("No layers selected for visualization.")

    out_dir = Path(args.out_dir)
    tag = checkpoint_path.parent.parent.name if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.stem
    # Output layout:
    # - Single sample: keep backward-compat path: .../<tag>_<split>_idxK/
    # - Multiple samples: use a run dir and subfolders: .../<tag>_<split>_idxK_nN/idx{...}/
    if len(sample_idxs) == 1:
        run_dir = out_dir / f"{tag}_{args.split}_idx{int(sample_idxs[0])}"
    else:
        run_dir = out_dir / f"{tag}_{args.split}_idx{int(sample_idxs[0])}_n{len(sample_idxs)}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Cache overview rows: each row = [Input, L4, L14, L24] (prune overlays only).
    overview_rows: List[List[Tuple[str, Image.Image]]] = []

    scale = max(int(args.scale), 1)

    from engine.data_utils import preprocess_batch

    for sample_idx in sample_idxs:
        sample = ds[int(sample_idx)]

        batch = preprocess_batch(
            [sample],
            processor=processor,
            device=device,
            max_length=int(config.trainer_settings.dl_settings.get("max_length", 1024)),
            mode="inference",
        )

        inputs = batch["inputs"]
        vision_start = int(batch["vision_start"])
        vision_end = int(batch["vision_end"])
        question_starts = [int(x) for x in batch["question_starts"]]
        question_ends = [int(x) for x in batch["question_ends"]]

        # Decode processed image for perfect grid alignment
        pixel_values = inputs["pixel_values"][0]  # (3,H,W)
        processed_img = _pixel_values_to_pil(pixel_values, processor)
        vis_img = processed_img
        if scale != 1:
            vis_img = processed_img.resize((processed_img.width * scale, processed_img.height * scale), resample=Image.BICUBIC)

        # Per-sample output dir
        if len(sample_idxs) == 1:
            save_dir = run_dir
        else:
            save_dir = run_dir / f"idx{int(sample_idx)}"
            save_dir.mkdir(parents=True, exist_ok=True)

        vis_img.save(save_dir / "input_processed.png")

        # Also save original image (uncropped) as reference
        try:
            orig_img = sample.get("image", None)
            if isinstance(orig_img, Image.Image):
                orig_img.convert("RGB").save(save_dir / "input_original.png")
        except Exception:
            pass

        with torch.no_grad():
            # autocast is safe but not strictly required; keep it simple on CPU.
            if device.type == "cuda":
                ctx = torch.autocast(
                    device_type="cuda",
                    dtype=_dtype_from_str(config.global_settings.get("dtype", "bfloat16")),
                )
            else:
                ctx = torch.autocast(device_type="cpu", dtype=torch.float32, enabled=False)
            with ctx:
                out = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    attention_mask=inputs.get("attention_mask", None),
                    vision_start=vision_start,
                    vision_end=vision_end,
                    question_starts=question_starts,
                    question_ends=question_ends,
                    answer_starts=None,
                    answer_ends=None,
                    return_pruning_info=True,
                    pruning_mode="normal",
                    apply_repair=False,
                )

        if out.pruning_infos is None:
            raise RuntimeError("Model returned no pruning_infos; check pruning_layers / forward route.")

        # Build one overview row: Input + prune overlays.
        row: List[Tuple[str, Image.Image]] = [("Input", vis_img)]

        # Still export per-layer keep overlays for debugging/ablation figures.
        for layer_idx in layers_to_viz:
            info = out.pruning_infos.get(int(layer_idx))
            if info is None:
                raise KeyError(
                    f"Layer {layer_idx} not found in pruning_infos. Available: {sorted(out.pruning_infos.keys())}"
                )
            mask = info["cumulative_mask"][0].detach().float().cpu().numpy()  # (576,)
            grid = _mask_to_grid(mask)

            np.save(save_dir / f"L{layer_idx}_cumulative_mask.npy", grid.astype(np.uint8))

            keep_img = _overlay_mask(vis_img, grid, color_rgb=(20, 180, 20), alpha=float(args.alpha))
            keep_img.save(save_dir / f"L{layer_idx}_keep.png")
            Image.fromarray((grid > 0.5).astype(np.uint8) * 255, mode="L").resize(vis_img.size, resample=Image.NEAREST).save(
                save_dir / f"L{layer_idx}_keep_mask.png"
            )

        for layer_idx in layers_to_viz:
            info = out.pruning_infos.get(int(layer_idx))
            mask = info["cumulative_mask"][0].detach().float().cpu().numpy()
            grid = _mask_to_grid(mask)
            pruned = (1.0 - grid).astype(np.float32)

            prune_img = _overlay_mask(vis_img, pruned, color_rgb=(200, 30, 30), alpha=float(args.alpha))
            prune_img.save(save_dir / f"L{layer_idx}_prune.png")
            Image.fromarray((pruned > 0.5).astype(np.uint8) * 255, mode="L").resize(vis_img.size, resample=Image.NEAREST).save(
                save_dir / f"L{layer_idx}_prune_mask.png"
            )
            row.append((f"L{layer_idx}", prune_img))

        overview_rows.append(row)

        # Quick per-sample summary
        print(f"[Saved] {save_dir}")
        for layer_idx in layers_to_viz:
            info = out.pruning_infos[int(layer_idx)]
            grid = _mask_to_grid(info["cumulative_mask"][0].detach().float().cpu().numpy())
            print(f"  idx{int(sample_idx)} L{layer_idx}: kept={grid.mean():.3f} ({int(grid.sum())}/{grid.size})")

    # Save combined overview panel if multiple samples; otherwise keep the single-sample behavior.
    if len(overview_rows) == 1:
        _save_overview(run_dir, panels=overview_rows[0], n_layers=len(layers_to_viz))
    else:
        _save_overview_grid(run_dir, overview_rows=overview_rows, n_layers=len(layers_to_viz))

    print(f"[Overview] {run_dir / 'overview.png'}")


if __name__ == "__main__":
    main()
