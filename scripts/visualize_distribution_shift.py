#!/usr/bin/env python
"""Compare distribution shift-to-real between THREE checkpoints.

This script compares two *trained variants* (A/B) against a separate "real" baseline model.

Goal:
    Compare
        (A) teacher-trained + adapter
    vs  (B) no-teacher + no-adapter
    by measuring how far their pruned (and optionally repaired) representations are from a "real" baseline.

Baseline (h_real):
    For a given input, run the *real baseline checkpoint* in keep-all mode (no pruning) and with apply_repair=False,
    then capture gen_answer hidden states at specified layers.

Compared representations (h_pred):
    For each checkpoint, run the model normally (pruning on; apply_repair auto-enabled if checkpoint contains repair weights),
    capture gen_answer hidden states at the same layers, and compute distance to h_real.

Usage:
    python scripts/visualize_distribution_shift.py \
        --checkpoint_real path/to/real.ckpt \
        --checkpoint_a path/to/a.ckpt \
        --checkpoint_b path/to/b.ckpt \
        --config_real configs/vision_token_pruning.yaml \
        --config_a configs/vision_token_pruning.yaml \
        --config_b configs/vision_token_pruning.yaml
"""

import os
os.environ["HF_HOME"] = "/data/users/zjw/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
from pathlib import Path
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# add repo root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint_real",
        type=str,
        required=True,
        help=(
            "Checkpoint Real (unpruned/ground-truth baseline). "
            "Special values: 'pretrained'/'hf'/'base' to use the pretrained base model as Real baseline (keep-all)."
        ),
    )
    p.add_argument("--checkpoint_a", type=str, required=True, help="Checkpoint A (e.g., teacher+adapter)")
    p.add_argument("--checkpoint_b", type=str, required=True, help="Checkpoint B (e.g., no-teacher+no-adapter)")
    p.add_argument("--label_real", type=str, default="Real(keep-all)", help="Legend label for real baseline")
    p.add_argument("--label_a", type=str, default="A(teacher+adapter)", help="Legend label for checkpoint A")
    p.add_argument("--label_b", type=str, default="B(no-teacher,no-adapter)", help="Legend label for checkpoint B")
    p.add_argument("--config_real", type=str, default="configs/vision_token_pruning.yaml", help="Config file for Real")
    p.add_argument("--config_a", type=str, default="configs/vision_token_pruning.yaml", help="Config file for A")
    p.add_argument("--config_b", type=str, default="configs/vision_token_pruning.yaml", help="Config file for B")
    p.add_argument("--num_samples", type=int, default=100, help="Number of samples (from train split)")
    p.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    p.add_argument("--output_dir", type=str, default="outputs/visualizations", help="Output directory")
    p.add_argument("--max_length", type=int, default=1024, help="Tokenizer max_length")
    p.add_argument(
        "--model_path",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="HF model id / local path for LLaVA base weights (used for processor and Real='pretrained').",
    )
    p.add_argument(
        "--capture_layers",
        type=str,
        default="",
        help=(
            "Layers to capture. Supported formats:\n"
            "  - Comma list: '13,22,29'\n"
            "  - Range: '0-31' (inclusive) or '0:32' (end-exclusive)\n"
            "  - 'all' (capture every decoder layer)\n"
            "Default: use config_real.method_settings.repair_layers else pruning_layers."
        ),
    )
    p.add_argument("--no_tsne", action="store_true", help="Skip t-SNE visualization (faster)")
    p.add_argument(
        "--summary_only",
        action="store_true",
        help="Only compute and plot layer-wise distance curves (no per-layer PCA/t-SNE/LDA figures). Recommended for --capture_layers all.",
    )
    p.add_argument(
        "--composite",
        action="store_true",
        help=(
            "Composite report: run layer-wise summary (recommended with --capture_layers all), "
            "plus a deep-dive on key layers (repair/pruning + largest gaps) with per-layer figures."
        ),
    )
    p.add_argument(
        "--deep_dive_layers",
        type=str,
        default="",
        help=(
            "Override deep-dive layers (same format as --capture_layers). "
            "If empty, uses config_real repair/pruning layers (+/- neighbors) plus top-k gap layers."
        ),
    )
    p.add_argument("--deep_dive_neighbors", type=int, default=1, help="Neighbor layers (+/-K) to include around key layers.")
    p.add_argument("--deep_dive_topk", type=int, default=5, help="Add top-k layers with largest (B-A) mean L2 gap to deep-dive.")
    p.add_argument(
        "--skip_repair_layers_a",
        type=str,
        default="",
        help="Comma-separated list of repair layer indices to skip for checkpoint A (e.g., '13' or '13,22'). Use 'first' to skip the first layer, 'last' to skip the last.",
    )
    p.add_argument(
        "--skip_repair_layers_b",
        type=str,
        default="",
        help="Comma-separated list of repair layer indices to skip for checkpoint B (e.g., '13' or '13,22'). Use 'first' to skip the first layer, 'last' to skip the last.",
    )
    return p.parse_args()


def _parse_layer_list(s: str, *, num_layers: Optional[int] = None) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    s_lower = s.lower()
    if s_lower in {"all", "every"}:
        if num_layers is None:
            raise ValueError("capture_layers='all' requires num_layers to be known.")
        return list(range(int(num_layers)))

    def _add_range(out: List[int], start: int, end_exclusive: int):
        if end_exclusive <= start:
            return
        out.extend(list(range(start, end_exclusive)))

    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
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
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup


def _parse_skip_repair_layers(skip_spec: str, all_repair_layers: List[int]) -> List[int]:
    """Parse skip_repair_layers specification.

    Args:
        skip_spec: Comma-separated layer indices, or 'first'/'last' keywords.
        all_repair_layers: All available repair layers from checkpoint (sorted).

    Returns:
        List of layer indices to skip.
    """
    skip_spec = (skip_spec or "").strip()
    if not skip_spec:
        return []

    if not all_repair_layers:
        return []

    skip_layers = []
    for part in skip_spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part == "first":
            skip_layers.append(all_repair_layers[0])
        elif part == "last":
            skip_layers.append(all_repair_layers[-1])
        else:
            try:
                skip_layers.append(int(part))
            except ValueError:
                print(f"  Warning: Invalid skip_repair_layers value '{part}', ignoring.")
    return list(set(skip_layers))  # Remove duplicates


def load_samples(num_samples: int, config_path: str):
    """Load samples from dataset configured in config_path."""
    from engine.configs.loader import load_config
    from engine.datas.loader import load_dataset
    from itertools import islice

    config = load_config(override_file=config_path)
    data_bundle = load_dataset(config)
    dataset = data_bundle["splits"]["train"]
    return list(islice(dataset, num_samples))


def load_processor(model_path: str):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "right"
    return processor


def preprocess_sample(sample, processor, max_length: int = 1024) -> Optional[Dict[str, Any]]:
    """CPU preprocessing into tensors + token span metadata."""
    image = sample["image"]
    question = sample["question"]
    answer = sample["answer"]

    prompt = f"USER: <image>\n{question}\nASSISTANT: {answer.capitalize()}"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = inputs["input_ids"]
    _, seq_len = input_ids.shape

    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    n_vision_tokens = 576

    image_positions = (input_ids[0] == image_token_id).nonzero(as_tuple=True)[0]
    if len(image_positions) > 0:
        vision_start = int(image_positions[0].item())
        vision_end = vision_start + n_vision_tokens
    else:
        vision_start = 1
        vision_end = vision_start + n_vision_tokens

    # locate "\nASSISTANT:"
    assistant_ids = processor.tokenizer.encode("\nASSISTANT:", add_special_tokens=False)
    if assistant_ids and assistant_ids[0] == 29871:
        assistant_ids = assistant_ids[1:]

    ids = input_ids[0].tolist()
    assistant_pos = None
    for j in range(len(ids) - len(assistant_ids) + 1):
        if ids[j : j + len(assistant_ids)] == assistant_ids:
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
        "inputs": inputs,  # CPU tensors
        "vision_start": vision_start,
        "vision_end": vision_end,
        "question_starts": question_starts,
        "question_ends": question_ends,
        "answer_starts": answer_starts,
        "answer_ends": answer_ends,
    }


def _move_inputs_to_device(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in inputs.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@contextmanager
def _temporary_keep_all_pruning(model):
    """Temporarily set pruning_threshold very low so y_soft > threshold => keep-all."""
    pruners = getattr(getattr(model, "pruner_manager", None), "pruners", None)
    if not pruners:
        yield
        return
    first_pruner = list(pruners.values())[0]
    old_threshold = float(first_pruner.pruning_threshold)
    try:
        model.pruner_manager.set_pruning_threshold(-1e9)
        yield
    finally:
        model.pruner_manager.set_pruning_threshold(old_threshold)


def _extract_flat_and_pooled(capture_entry) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract token-level (flat) and sample-level (pooled mean) vectors.

    Expected capture_entry format (current PrunableLlava):
        {"h": (b,L,D), "mask": (b,L)}
    """
    if capture_entry is None:
        return None, None

    if isinstance(capture_entry, dict) and ("h" in capture_entry) and ("mask" in capture_entry):
        h = capture_entry["h"]
        m = capture_entry["mask"]
    else:
        h = capture_entry
        m = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)

    if h.dim() != 3 or m.dim() != 2:
        raise ValueError(f"Unexpected capture shapes: h={tuple(h.shape)} mask={tuple(m.shape)}")

    b, _, _ = h.shape
    flat_list = []
    pooled_list = []
    for i in range(b):
        valid = m[i] > 0.5
        if valid.sum().item() <= 0:
            continue
        hi = h[i][valid]  # (Li,D)
        flat_list.append(hi)
        pooled_list.append(hi.mean(dim=0, keepdim=True))

    if not flat_list:
        return None, None

    flat = torch.cat(flat_list, dim=0).float().cpu().numpy()
    pooled = torch.cat(pooled_list, dim=0).float().cpu().numpy()
    return flat, pooled


def _extract_pooled_only(capture_entry) -> Optional[np.ndarray]:
    """Extract only sample-level pooled mean vectors: (N,D)."""
    if capture_entry is None:
        return None

    if isinstance(capture_entry, dict) and ("h" in capture_entry) and ("mask" in capture_entry):
        h = capture_entry["h"]
        m = capture_entry["mask"]
    else:
        h = capture_entry
        m = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)

    if h.dim() != 3 or m.dim() != 2:
        raise ValueError(f"Unexpected capture shapes: h={tuple(h.shape)} mask={tuple(m.shape)}")

    b, _, _ = h.shape
    pooled_list = []
    for i in range(b):
        valid = m[i] > 0.5
        if valid.sum().item() <= 0:
            continue
        hi = h[i][valid]  # (Li,D)
        pooled_list.append(hi.mean(dim=0, keepdim=True))
    if not pooled_list:
        return None
    return torch.cat(pooled_list, dim=0).float().cpu().numpy()


def _build_prunable_llava_from_pretrained(*, config_path: str, device: torch.device, model_path: str):
    """Build PrunableLlava from pretrained base weights (no checkpoint weights loaded)."""
    from transformers import LlavaForConditionalGeneration
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
    from engine.configs.loader import load_config

    config = load_config(override_file=config_path)
    method_cfg = config["method_settings"]

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
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
        # NOTE: Real baseline will always run with apply_repair=False in forward.
        # Keeping this flag from config is safe but not necessary.
        use_repair_adapter=bool(method_cfg.get("use_repair_adapter", False)),
        repair_layers=method_cfg.get("repair_layers", None),
        repair_source_layers=method_cfg.get("repair_source_layers", None),
        repair_bottleneck_dim=method_cfg.get("repair_bottleneck_dim", 512),
        repair_dropout=0.0,
        repair_mask_encoder_type=method_cfg.get("repair_mask_encoder_type", "attention"),
        repair_use_pruned_info=method_cfg.get("repair_use_pruned_info", True),
        repair_alpha_init=method_cfg.get("repair_alpha_init", 0.1),
    )
    model.freeze_base_model()
    model.eval()
    return model, config


def _infer_adapter_layers_from_state_dict(adapter_state_dict: Dict[str, Any]) -> List[int]:
    """Infer adapter layer indices from AdapterManager state dict keys.

    Expected key format:
        'adapters.{layer_idx}....'
    """
    if not adapter_state_dict:
        return []
    layers = set()
    for k in adapter_state_dict.keys():
        m = re.match(r"^adapters\.(\d+)\.", str(k))
        if m:
            layers.add(int(m.group(1)))
    return sorted(layers)


def _resolve_skip_repair_layers(checkpoint_path: str, skip_spec: str) -> List[int]:
    """Resolve skip_repair_layers by loading checkpoint and parsing spec.

    Args:
        checkpoint_path: Path to checkpoint file.
        skip_spec: Skip specification (e.g., 'first', '13', '13,22').

    Returns:
        List of layer indices to skip.
    """
    if not skip_spec.strip():
        return []

    try:
        meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "repair_adapter_state_dict" not in meta:
            return []
        all_layers = _infer_adapter_layers_from_state_dict(meta["repair_adapter_state_dict"])
        return _parse_skip_repair_layers(skip_spec, all_layers)
    except Exception as e:
        print(f"  Warning: Failed to resolve skip_repair_layers from {checkpoint_path}: {e}")
        return []


def load_model_from_checkpoint(checkpoint_path: str, config_path: str, device: torch.device, model_path: str, skip_repair_layers: Optional[List[int]] = None):
    """Load PrunableLlava model; auto-enable repair adapter only if checkpoint contains its weights.

    Args:
        skip_repair_layers: List of repair layer indices to skip loading (e.g., [13] to skip layer 13).
    """
    from transformers import LlavaForConditionalGeneration
    from method.models.prunable_llava import PrunableLlavaForConditionalGeneration
    from engine.configs.loader import load_config

    config = load_config(override_file=config_path)
    method_cfg = config["method_settings"]

    meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_has_repair = ("repair_context_encoder_state_dict" in meta) and ("repair_adapter_state_dict" in meta)
    use_repair_adapter = bool(method_cfg.get("use_repair_adapter", False) and ckpt_has_repair)
    if bool(method_cfg.get("use_repair_adapter", False)) and not ckpt_has_repair:
        print("  Note: config requests repair adapter, but checkpoint has no repair weights; disabling repair.")

    # If checkpoint contains repair adapter weights, prefer its layer indices to avoid config/checkpoint mismatch.
    repair_layers_cfg = list(method_cfg.get("repair_layers", None) or [])
    repair_source_layers_cfg = method_cfg.get("repair_source_layers", None)
    repair_layers_for_model = repair_layers_cfg
    repair_source_layers_for_model = repair_source_layers_cfg
    if ckpt_has_repair:
        inferred = _infer_adapter_layers_from_state_dict(meta.get("repair_adapter_state_dict", {}) or {})
        if inferred:
            if repair_layers_cfg and (sorted([int(x) for x in repair_layers_cfg]) != inferred):
                print(f"  Note: config repair_layers={repair_layers_cfg} != checkpoint repair_layers={inferred}; using checkpoint.")
            repair_layers_for_model = inferred

            # Apply skip_repair_layers filter
            if skip_repair_layers:
                original_layers = repair_layers_for_model
                repair_layers_for_model = [l for l in repair_layers_for_model if l not in skip_repair_layers]
                print(f"  Note: Skipping repair layers {skip_repair_layers}. Using layers {repair_layers_for_model} (original: {original_layers})")
                if not repair_layers_for_model:
                    print("  Warning: All repair layers were skipped; disabling repair adapter.")
                    use_repair_adapter = False

            # Only keep explicit source mapping if it matches inferred layers length.
            if repair_source_layers_cfg is not None and len(list(repair_source_layers_cfg)) != len(repair_layers_for_model):
                print(
                    "  Note: config repair_source_layers length mismatches checkpoint repair_layers; "
                    "disabling explicit mapping (will auto-pick nearest pruning layer)."
                )
                repair_source_layers_for_model = None

    base_model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=True,
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
        print("  Loaded pruner_state_dict")
    if model.use_repair_adapter:
        model.repair_context_encoder.load_state_dict(ckpt["repair_context_encoder_state_dict"])

        # Selectively load repair adapter weights (skip specified layers)
        full_adapter_state = ckpt["repair_adapter_state_dict"]
        if skip_repair_layers:
            filtered_state = {}
            for key, value in full_adapter_state.items():
                # Check if this key belongs to a skipped layer
                # Expected key format: 'adapters.{layer_idx}....'
                m = re.match(r"^adapters\.(\d+)\.", str(key))
                if m:
                    layer_idx = int(m.group(1))
                    if layer_idx in skip_repair_layers:
                        continue  # Skip this layer
                filtered_state[key] = value
            model.repair_adapter_manager.load_state_dict(filtered_state, strict=False)
            print(f"  Loaded repair_adapter_state_dict (skipped layers: {skip_repair_layers})")
        else:
            model.repair_adapter_manager.load_state_dict(full_adapter_state)
            print("  Loaded repair_context_encoder_state_dict + repair_adapter_state_dict")

    model.eval()
    return model, config


def _infer_num_decoder_layers(model) -> int:
    """Infer number of decoder layers for LLaVA/LLaMA backbone."""
    # Prefer config-based inference (stable across HF versions / module layouts).
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
    raise RuntimeError(
        "Could not infer num decoder layers from model (tried config.text_config.* and common module paths)."
    )


def collect_pooled_only(
    *,
    model,
    prepared_samples: List[Optional[Dict[str, Any]]],
    device: torch.device,
    capture_layers: List[int],
    apply_repair: Optional[bool],
    keep_all: bool,
    label: str,
) -> Dict[int, np.ndarray]:
    """Collect only pooled vectors (per sample mean over gen_answer tokens) at capture_layers.

    Returns:
        {layer: pooled (N,D)}
    """
    pooled_all = defaultdict(list)
    print(f"Collecting POOLED layers={capture_layers} | keep_all={keep_all} | apply_repair={apply_repair} | {label}")
    for i, prep in enumerate(prepared_samples):
        if prep is None:
            continue
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(prepared_samples)}...")

        inputs = _move_inputs_to_device(prep["inputs"], device)
        with torch.no_grad():
            if keep_all:
                with _temporary_keep_all_pruning(model):
                    out = model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs.get("pixel_values", None),
                        attention_mask=inputs.get("attention_mask", None),
                        vision_start=prep["vision_start"],
                        vision_end=prep["vision_end"],
                        question_starts=prep["question_starts"],
                        question_ends=prep["question_ends"],
                        answer_starts=prep["answer_starts"],
                        answer_ends=prep["answer_ends"],
                        return_pruning_info=False,
                        apply_repair=apply_repair,
                        capture_layers=capture_layers,
                    )
            else:
                out = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values", None),
                    attention_mask=inputs.get("attention_mask", None),
                    vision_start=prep["vision_start"],
                    vision_end=prep["vision_end"],
                    question_starts=prep["question_starts"],
                    question_ends=prep["question_ends"],
                    answer_starts=prep["answer_starts"],
                    answer_ends=prep["answer_ends"],
                    return_pruning_info=False,
                    apply_repair=apply_repair,
                    capture_layers=capture_layers,
                )

        cap = getattr(out, "captured", None) or {}
        for layer_idx in capture_layers:
            if layer_idx not in cap:
                continue
            pooled = _extract_pooled_only(cap[layer_idx])
            if pooled is not None:
                pooled_all[layer_idx].append(pooled)

    result = {}
    for layer_idx in capture_layers:
        if not pooled_all[layer_idx]:
            continue
        result[layer_idx] = np.concatenate(pooled_all[layer_idx], axis=0)
        print(f"  Layer {layer_idx}: pooled={result[layer_idx].shape}")
    return result


def _save_layerwise_curves(
    *,
    capture_layers: List[int],
    real: Dict[int, np.ndarray],
    a: Dict[int, np.ndarray],
    b: Dict[int, np.ndarray],
    output_dir: str,
    label_a: str,
    label_b: str,
    repair_layers: Optional[List[int]] = None,
    pruning_layers: Optional[List[int]] = None,
):
    import csv

    os.makedirs(output_dir, exist_ok=True)
    rows = []
    xs = []
    mean_l2_a = []
    mean_l2_b = []
    mean_cos_a = []
    mean_cos_b = []
    mean_l2_unit_a = []
    mean_l2_unit_b = []
    mean_norm_r = []
    mean_norm_a = []
    mean_norm_b = []

    for layer_idx in capture_layers:
        Xr = real.get(layer_idx, None)
        Xa = a.get(layer_idx, None)
        Xb = b.get(layer_idx, None)
        if Xr is None or Xa is None or Xb is None:
            continue
        n = min(len(Xr), len(Xa), len(Xb))
        if n <= 0:
            continue
        Xr = Xr[:n]
        Xa = Xa[:n]
        Xb = Xb[:n]

        center_real = Xr.mean(axis=0)
        center_a = Xa.mean(axis=0)
        center_b = Xb.mean(axis=0)
        center_l2_a = float(np.linalg.norm(center_a - center_real))
        center_l2_b = float(np.linalg.norm(center_b - center_real))
        center_cos_a = float(_cosine_distance(center_a[None, :], center_real[None, :])[0])
        center_cos_b = float(_cosine_distance(center_b[None, :], center_real[None, :])[0])

        per_l2_a = np.linalg.norm(Xa - Xr, axis=1)
        per_l2_b = np.linalg.norm(Xb - Xr, axis=1)
        per_cos_a = _cosine_distance(Xa, Xr)
        per_cos_b = _cosine_distance(Xb, Xr)

        # scale diagnostics
        nr = np.linalg.norm(Xr, axis=1)
        na = np.linalg.norm(Xa, axis=1)
        nb = np.linalg.norm(Xb, axis=1)

        def _unit(x):
            nrm = np.linalg.norm(x, axis=1, keepdims=True)
            return x / (nrm + 1e-8)

        per_l2_unit_a = np.linalg.norm(_unit(Xa) - _unit(Xr), axis=1)
        per_l2_unit_b = np.linalg.norm(_unit(Xb) - _unit(Xr), axis=1)

        rows.append(
            {
                "layer": layer_idx,
                "n": n,
                "center_l2_a": center_l2_a,
                "center_l2_b": center_l2_b,
                "center_cos_a": center_cos_a,
                "center_cos_b": center_cos_b,
                "mean_l2_a": float(per_l2_a.mean()),
                "std_l2_a": float(per_l2_a.std()),
                "mean_l2_b": float(per_l2_b.mean()),
                "std_l2_b": float(per_l2_b.std()),
                "mean_cos_a": float(per_cos_a.mean()),
                "std_cos_a": float(per_cos_a.std()),
                "mean_cos_b": float(per_cos_b.mean()),
                "std_cos_b": float(per_cos_b.std()),
                "mean_l2_unit_a": float(per_l2_unit_a.mean()),
                "std_l2_unit_a": float(per_l2_unit_a.std()),
                "mean_l2_unit_b": float(per_l2_unit_b.mean()),
                "std_l2_unit_b": float(per_l2_unit_b.std()),
                "mean_norm_real": float(nr.mean()),
                "mean_norm_a": float(na.mean()),
                "mean_norm_b": float(nb.mean()),
            }
        )

        xs.append(layer_idx)
        mean_l2_a.append(float(per_l2_a.mean()))
        mean_l2_b.append(float(per_l2_b.mean()))
        mean_cos_a.append(float(per_cos_a.mean()))
        mean_cos_b.append(float(per_cos_b.mean()))
        mean_l2_unit_a.append(float(per_l2_unit_a.mean()))
        mean_l2_unit_b.append(float(per_l2_unit_b.mean()))
        mean_norm_r.append(float(nr.mean()))
        mean_norm_a.append(float(na.mean()))
        mean_norm_b.append(float(nb.mean()))

    csv_path = os.path.join(output_dir, "layerwise_distances.csv")
    if not rows:
        print("No layer-wise rows to save.")
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved layer-wise CSV: {csv_path}")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(xs, mean_l2_a, marker="o", label=f"{label_a} -> real")
    axes[0].plot(xs, mean_l2_b, marker="o", label=f"{label_b} -> real")
    axes[0].set_ylabel("Mean L2 (pooled)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(xs, mean_cos_a, marker="o", label=f"{label_a} -> real")
    axes[1].plot(xs, mean_cos_b, marker="o", label=f"{label_b} -> real")
    axes[1].set_ylabel("Mean Cosine Dist (pooled)")
    axes[1].set_xlabel("Layer")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    for ax in axes:
        if pruning_layers:
            for l in pruning_layers:
                ax.axvline(l, color="k", linestyle="--", alpha=0.15)
        if repair_layers:
            for l in repair_layers:
                ax.axvline(l, color="tab:green", linestyle="--", alpha=0.25)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "layerwise_distance_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved layer-wise curves: {fig_path}")

    # Unit-normalized L2 (scale-invariant-ish)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xs, mean_l2_unit_a, marker="o", label=f"{label_a} -> real")
    ax.plot(xs, mean_l2_unit_b, marker="o", label=f"{label_b} -> real")
    ax.set_ylabel("Mean L2 (unit-normalized pooled)")
    ax.set_xlabel("Layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if pruning_layers:
        for l in pruning_layers:
            ax.axvline(l, color="k", linestyle="--", alpha=0.15)
    if repair_layers:
        for l in repair_layers:
            ax.axvline(l, color="tab:green", linestyle="--", alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "layerwise_distance_curves_unitnorm.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved unit-norm curves: {fig_path}")

    # Norm curves (diagnostics for L2 blow-up)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(xs, mean_norm_r, marker="o", label="real ||h||")
    ax.plot(xs, mean_norm_a, marker="o", label=f"{label_a} ||h||")
    ax.plot(xs, mean_norm_b, marker="o", label=f"{label_b} ||h||")
    ax.set_ylabel("Mean ||h|| (pooled)")
    ax.set_xlabel("Layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if pruning_layers:
        for l in pruning_layers:
            ax.axvline(l, color="k", linestyle="--", alpha=0.15)
    if repair_layers:
        for l in repair_layers:
            ax.axvline(l, color="tab:green", linestyle="--", alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "layerwise_norm_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved norm curves: {fig_path}")

    # Gap curve (B-A), positive means A is closer to real on average.
    gap_l2 = [b - a for a, b in zip(mean_l2_a, mean_l2_b)]
    gap_cos = [b - a for a, b in zip(mean_cos_a, mean_cos_b)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(xs, gap_l2, marker="o", color="tab:purple")
    axes[0].axhline(0.0, color="k", linewidth=1, alpha=0.4)
    axes[0].set_ylabel("Gap Mean L2 (B - A)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(xs, gap_cos, marker="o", color="tab:purple")
    axes[1].axhline(0.0, color="k", linewidth=1, alpha=0.4)
    axes[1].set_ylabel("Gap Mean Cos Dist (B - A)")
    axes[1].set_xlabel("Layer")
    axes[1].grid(True, alpha=0.3)
    for ax in axes:
        if pruning_layers:
            for l in pruning_layers:
                ax.axvline(l, color="k", linestyle="--", alpha=0.15)
        if repair_layers:
            for l in repair_layers:
                ax.axvline(l, color="tab:green", linestyle="--", alpha=0.25)
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "layerwise_gap_curves.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved gap curves: {fig_path}")

    return rows


def _save_repair_layer_bars(
    *,
    rows: List[Dict[str, Any]],
    repair_layers: List[int],
    output_dir: str,
    label_a: str,
    label_b: str,
):
    os.makedirs(output_dir, exist_ok=True)
    # index rows by layer
    by_layer = {int(r["layer"]): r for r in rows}
    layers = [int(l) for l in repair_layers if int(l) in by_layer]
    if not layers:
        print("No repair layers present in computed rows; skip repair-layer bar plot.")
        return

    # Save a small CSV for convenience
    import csv
    csv_path = os.path.join(output_dir, "repair_layers_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "mean_l2_a",
                "std_l2_a",
                "mean_l2_b",
                "std_l2_b",
                "mean_cos_a",
                "std_cos_a",
                "mean_cos_b",
                "std_cos_b",
            ]
        )
        for l in layers:
            r = by_layer[l]
            writer.writerow(
                [
                    l,
                    float(r["mean_l2_a"]),
                    float(r["std_l2_a"]),
                    float(r["mean_l2_b"]),
                    float(r["std_l2_b"]),
                    float(r["mean_cos_a"]),
                    float(r["std_cos_a"]),
                    float(r["mean_cos_b"]),
                    float(r["std_cos_b"]),
                ]
            )
    print(f"Saved repair-layer CSV: {csv_path}")

    mean_a = [float(by_layer[l]["mean_l2_a"]) for l in layers]
    std_a = [float(by_layer[l]["std_l2_a"]) for l in layers]
    mean_b = [float(by_layer[l]["mean_l2_b"]) for l in layers]
    std_b = [float(by_layer[l]["std_l2_b"]) for l in layers]

    x = np.arange(len(layers))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, mean_a, width, yerr=std_a, capsize=3, label=label_a)
    ax.bar(x + width / 2, mean_b, width, yerr=std_b, capsize=3, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Repair layer")
    ax.set_ylabel("Mean L2 (pooled) ± std")
    ax.set_title("Repair layers: distance to real (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "repair_layers_l2_bar.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved repair-layer bar: {fig_path}")

    # Cosine distance bar
    mean_a = [float(by_layer[l]["mean_cos_a"]) for l in layers]
    std_a = [float(by_layer[l]["std_cos_a"]) for l in layers]
    mean_b = [float(by_layer[l]["mean_cos_b"]) for l in layers]
    std_b = [float(by_layer[l]["std_cos_b"]) for l in layers]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, mean_a, width, yerr=std_a, capsize=3, label=label_a)
    ax.bar(x + width / 2, mean_b, width, yerr=std_b, capsize=3, label=label_b)
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.set_xlabel("Repair layer")
    ax.set_ylabel("Mean Cosine Dist (pooled) ± std")
    ax.set_title("Repair layers: cosine distance to real (lower is better)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig_path = os.path.join(output_dir, "repair_layers_cos_bar.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved repair-layer bar: {fig_path}")


def collect_h_real_and_pred(
    model,
    prepared_samples: List[Optional[Dict[str, Any]]],
    device: torch.device,
    capture_layers: List[int],
    apply_repair_pred: bool,
    compute_real: bool,
) -> Dict[int, Dict[str, Optional[np.ndarray]]]:
    """Collect real/pred vectors at capture_layers.

    Returns:
        {layer: {"real_tokens","pred_tokens","real_pooled","pred_pooled"}}
    """
    real_tokens_all = defaultdict(list)
    pred_tokens_all = defaultdict(list)
    real_pooled_all = defaultdict(list)
    pred_pooled_all = defaultdict(list)

    print(f"Collecting layers={capture_layers} | apply_repair_pred={apply_repair_pred} | compute_real={compute_real}")
    for i, prep in enumerate(prepared_samples):
        if prep is None:
            continue
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(prepared_samples)}...")

        inputs = _move_inputs_to_device(prep["inputs"], device)

        with torch.no_grad():
            out_pred = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values", None),
                attention_mask=inputs.get("attention_mask", None),
                vision_start=prep["vision_start"],
                vision_end=prep["vision_end"],
                question_starts=prep["question_starts"],
                question_ends=prep["question_ends"],
                answer_starts=prep["answer_starts"],
                answer_ends=prep["answer_ends"],
                return_pruning_info=True,
                apply_repair=apply_repair_pred,
                capture_layers=capture_layers,
            )

            out_real = None
            if compute_real:
                with _temporary_keep_all_pruning(model):
                    out_real = model(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs.get("pixel_values", None),
                        attention_mask=inputs.get("attention_mask", None),
                        vision_start=prep["vision_start"],
                        vision_end=prep["vision_end"],
                        question_starts=prep["question_starts"],
                        question_ends=prep["question_ends"],
                        answer_starts=prep["answer_starts"],
                        answer_ends=prep["answer_ends"],
                        return_pruning_info=False,
                        apply_repair=False,
                        capture_layers=capture_layers,
                    )

        cap_pred = getattr(out_pred, "captured", None) or {}
        cap_real = (getattr(out_real, "captured", None) or {}) if out_real is not None else {}

        for layer_idx in capture_layers:
            if layer_idx in cap_pred:
                flat, pooled = _extract_flat_and_pooled(cap_pred[layer_idx])
                if flat is not None:
                    pred_tokens_all[layer_idx].append(flat)
                if pooled is not None:
                    pred_pooled_all[layer_idx].append(pooled)
            if compute_real and (layer_idx in cap_real):
                flat, pooled = _extract_flat_and_pooled(cap_real[layer_idx])
                if flat is not None:
                    real_tokens_all[layer_idx].append(flat)
                if pooled is not None:
                    real_pooled_all[layer_idx].append(pooled)

    result = {}
    for layer_idx in capture_layers:
        pred_tokens = np.concatenate(pred_tokens_all[layer_idx], axis=0) if pred_tokens_all[layer_idx] else None
        pred_pooled = np.concatenate(pred_pooled_all[layer_idx], axis=0) if pred_pooled_all[layer_idx] else None
        real_tokens = np.concatenate(real_tokens_all[layer_idx], axis=0) if real_tokens_all[layer_idx] else None
        real_pooled = np.concatenate(real_pooled_all[layer_idx], axis=0) if real_pooled_all[layer_idx] else None
        if pred_tokens is None:
            continue
        result[layer_idx] = {
            "pred_tokens": pred_tokens,
            "pred_pooled": pred_pooled,
            "real_tokens": real_tokens,
            "real_pooled": real_pooled,
        }
        print(
            f"  Layer {layer_idx}: pred_tokens={pred_tokens.shape if pred_tokens is not None else None}, "
            f"real_tokens={real_tokens.shape if real_tokens is not None else None}"
        )
    return result


def collect_h_real_only(
    model,
    prepared_samples: List[Optional[Dict[str, Any]]],
    device: torch.device,
    capture_layers: List[int],
) -> Dict[int, Dict[str, Optional[np.ndarray]]]:
    """Collect only real vectors: keep-all + apply_repair=False."""
    real_tokens_all = defaultdict(list)
    real_pooled_all = defaultdict(list)

    print(f"Collecting REAL layers={capture_layers} (keep-all, apply_repair=False)")
    for i, prep in enumerate(prepared_samples):
        if prep is None:
            continue
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(prepared_samples)}...")

        inputs = _move_inputs_to_device(prep["inputs"], device)
        with torch.no_grad():
            with _temporary_keep_all_pruning(model):
                out_real = model(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values", None),
                    attention_mask=inputs.get("attention_mask", None),
                    vision_start=prep["vision_start"],
                    vision_end=prep["vision_end"],
                    question_starts=prep["question_starts"],
                    question_ends=prep["question_ends"],
                    answer_starts=prep["answer_starts"],
                    answer_ends=prep["answer_ends"],
                    return_pruning_info=False,
                    apply_repair=False,
                    capture_layers=capture_layers,
                )

        cap_real = getattr(out_real, "captured", None) or {}
        for layer_idx in capture_layers:
            if layer_idx not in cap_real:
                continue
            flat, pooled = _extract_flat_and_pooled(cap_real[layer_idx])
            if flat is not None:
                real_tokens_all[layer_idx].append(flat)
            if pooled is not None:
                real_pooled_all[layer_idx].append(pooled)

    result = {}
    for layer_idx in capture_layers:
        real_tokens = np.concatenate(real_tokens_all[layer_idx], axis=0) if real_tokens_all[layer_idx] else None
        real_pooled = np.concatenate(real_pooled_all[layer_idx], axis=0) if real_pooled_all[layer_idx] else None
        if real_tokens is None:
            continue
        result[layer_idx] = {
            "real_tokens": real_tokens,
            "real_pooled": real_pooled,
        }
        print(f"  Layer {layer_idx}: real_tokens={real_tokens.shape}")
    return result


def collect_h_pred_only(
    model,
    prepared_samples: List[Optional[Dict[str, Any]]],
    device: torch.device,
    capture_layers: List[int],
    apply_repair_pred: bool,
) -> Dict[int, Dict[str, Optional[np.ndarray]]]:
    """Collect only pred vectors (normal pruning + optional repair)."""
    pred_tokens_all = defaultdict(list)
    pred_pooled_all = defaultdict(list)

    print(f"Collecting PRED layers={capture_layers} | apply_repair_pred={apply_repair_pred}")
    for i, prep in enumerate(prepared_samples):
        if prep is None:
            continue
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(prepared_samples)}...")

        inputs = _move_inputs_to_device(prep["inputs"], device)
        with torch.no_grad():
            out_pred = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs.get("pixel_values", None),
                attention_mask=inputs.get("attention_mask", None),
                vision_start=prep["vision_start"],
                vision_end=prep["vision_end"],
                question_starts=prep["question_starts"],
                question_ends=prep["question_ends"],
                answer_starts=prep["answer_starts"],
                answer_ends=prep["answer_ends"],
                return_pruning_info=True,
                apply_repair=apply_repair_pred,
                capture_layers=capture_layers,
            )

        cap_pred = getattr(out_pred, "captured", None) or {}
        for layer_idx in capture_layers:
            if layer_idx not in cap_pred:
                continue
            flat, pooled = _extract_flat_and_pooled(cap_pred[layer_idx])
            if flat is not None:
                pred_tokens_all[layer_idx].append(flat)
            if pooled is not None:
                pred_pooled_all[layer_idx].append(pooled)

    result = {}
    for layer_idx in capture_layers:
        pred_tokens = np.concatenate(pred_tokens_all[layer_idx], axis=0) if pred_tokens_all[layer_idx] else None
        pred_pooled = np.concatenate(pred_pooled_all[layer_idx], axis=0) if pred_pooled_all[layer_idx] else None
        if pred_tokens is None:
            continue
        result[layer_idx] = {
            "pred_tokens": pred_tokens,
            "pred_pooled": pred_pooled,
        }
        print(f"  Layer {layer_idx}: pred_tokens={pred_tokens.shape}")
    return result


# ===== metrics =====

def _cosine_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 1 - np.sum(x * y, axis=-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + eps)


def compute_mmd(X, Y, gamma=1.0):
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
    return (K_XX.sum() / (m * m) + K_YY.sum() / (n * n) - 2 * K_XY.sum() / (m * n))


def _subsample_pair(X, Y, max_samples=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = min(len(X), len(Y), max_samples)
    if n <= 0:
        return None, None
    idx_x = rng.choice(len(X), n, replace=False)
    idx_y = rng.choice(len(Y), n, replace=False)
    return X[idx_x], Y[idx_y]


def compute_c2st(X, Y, test_size=0.3, seed=42):
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

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = None
    return {"acc": acc, "auc": auc}


def compute_swd(X, Y, n_projections=100, seed=42):
    Xs, Ys = _subsample_pair(X, Y, max_samples=2000, seed=seed)
    if Xs is None:
        return None
    rng = np.random.default_rng(seed)
    d = Xs.shape[1]
    dirs = rng.normal(size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8
    dists = []
    for v in dirs:
        proj_x = np.sort(Xs @ v)
        proj_y = np.sort(Ys @ v)
        dists.append(np.mean(np.abs(proj_x - proj_y)))
    return float(np.mean(dists))


def _sqrtm_psd(mat):
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0, None)
    return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T


def compute_frechet_distance(X, Y, pca_dim=64, seed=42):
    Xs, Ys = _subsample_pair(X, Y, max_samples=5000, seed=seed)
    if Xs is None:
        return None
    d = Xs.shape[1]
    pca_dim = min(pca_dim, d, len(Xs) - 1, len(Ys) - 1)
    if pca_dim < 2:
        return None
    pca = PCA(n_components=pca_dim, random_state=seed)
    Z = pca.fit_transform(np.vstack([Xs, Ys]))
    Zx = Z[: len(Xs)]
    Zy = Z[len(Xs) :]
    mu_x = Zx.mean(axis=0)
    mu_y = Zy.mean(axis=0)
    cov_x = np.cov(Zx, rowvar=False)
    cov_y = np.cov(Zy, rowvar=False)
    cov_x_sqrt = _sqrtm_psd(cov_x)
    cov_prod = cov_x_sqrt @ cov_y @ cov_x_sqrt
    cov_mean = _sqrtm_psd(cov_prod)
    diff = mu_x - mu_y
    return float(diff @ diff + np.trace(cov_x + cov_y - 2 * cov_mean))


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)


def analyze_and_visualize_compare(
    *,
    layer_idx: int,
    h_real_tokens: np.ndarray,
    h_a_tokens: np.ndarray,
    h_b_tokens: np.ndarray,
    h_real_pooled: Optional[np.ndarray],
    h_a_pooled: Optional[np.ndarray],
    h_b_pooled: Optional[np.ndarray],
    output_dir: str,
    label_a: str,
    label_b: str,
    run_tsne: bool,
):
    os.makedirs(output_dir, exist_ok=True)

    if h_real_pooled is None or h_a_pooled is None or h_b_pooled is None:
        # fallback to token-level
        h_real_pooled = h_real_tokens
        h_a_pooled = h_a_tokens
        h_b_pooled = h_b_tokens
        pooled_name = "token"
    else:
        pooled_name = "sample"

    print(f"\n=== Layer {layer_idx} ===")
    print(f"  token-level: real={h_real_tokens.shape}, A={h_a_tokens.shape}, B={h_b_tokens.shape}")
    print(f"  {pooled_name}-level: real={h_real_pooled.shape}, A={h_a_pooled.shape}, B={h_b_pooled.shape}")

    # [1] Center distance (pooled)
    center_real = h_real_pooled.mean(axis=0)
    center_a = h_a_pooled.mean(axis=0)
    center_b = h_b_pooled.mean(axis=0)
    l2_a = float(np.linalg.norm(center_a - center_real))
    l2_b = float(np.linalg.norm(center_b - center_real))
    cos_a = float(_cosine_distance(center_a[None, :], center_real[None, :])[0])
    cos_b = float(_cosine_distance(center_b[None, :], center_real[None, :])[0])
    print(f"  [Center/{pooled_name}] L2: {label_a}={l2_a:.4f} | {label_b}={l2_b:.4f}")
    print(f"  [Center/{pooled_name}] Cos: {label_a}={cos_a:.4f} | {label_b}={cos_b:.4f}")

    # [2] Per-vector distance (pooled)
    n = min(len(h_real_pooled), len(h_a_pooled), len(h_b_pooled))
    Xr = h_real_pooled[:n]
    Xa = h_a_pooled[:n]
    Xb = h_b_pooled[:n]
    per_l2_a = np.linalg.norm(Xa - Xr, axis=1)
    per_l2_b = np.linalg.norm(Xb - Xr, axis=1)
    per_cos_a = _cosine_distance(Xa, Xr)
    per_cos_b = _cosine_distance(Xb, Xr)
    print(f"  [Mean±Std/{pooled_name}] L2: {label_a}={per_l2_a.mean():.4f}±{per_l2_a.std():.4f} | {label_b}={per_l2_b.mean():.4f}±{per_l2_b.std():.4f}")
    print(f"  [Mean±Std/{pooled_name}] Cos: {label_a}={per_cos_a.mean():.4f}±{per_cos_a.std():.4f} | {label_b}={per_cos_b.mean():.4f}±{per_cos_b.std():.4f}")

    # [3] Token-level distribution distances
    print("  [Token] Shape:")
    std_real = float(h_real_tokens.std(axis=1).mean())
    std_a = float(h_a_tokens.std(axis=1).mean())
    std_b = float(h_b_tokens.std(axis=1).mean())
    var_real = float(h_real_tokens.var(axis=0).mean())
    var_a = float(h_a_tokens.var(axis=0).mean())
    var_b = float(h_b_tokens.var(axis=0).mean())
    print(f"    mean-std: real={std_real:.4f} | {label_a}={std_a:.4f} (ratio={std_a/(std_real+1e-8):.3f}) | {label_b}={std_b:.4f} (ratio={std_b/(std_real+1e-8):.3f})")
    print(f"    mean-var: real={var_real:.4f} | {label_a}={var_a:.4f} | {label_b}={var_b:.4f}")

    n_tok = h_real_tokens.shape[0]
    m = min(500, n_tok, h_a_tokens.shape[0], h_b_tokens.shape[0])
    idx = np.random.choice(n_tok, m, replace=False)
    mmd_a = compute_mmd(h_a_tokens[idx], h_real_tokens[idx])
    mmd_b = compute_mmd(h_b_tokens[idx], h_real_tokens[idx])
    print(f"  [Token] MMD: {label_a}={mmd_a:.6f} | {label_b}={mmd_b:.6f}")

    c2st_a = compute_c2st(h_real_tokens, h_a_tokens)
    c2st_b = compute_c2st(h_real_tokens, h_b_tokens)
    if c2st_a is not None:
        auc_str = f"{c2st_a['auc']:.3f}" if c2st_a["auc"] is not None else "N/A"
        print(f"  [Token] C2ST: {label_a} acc={c2st_a['acc']:.3f} auc={auc_str}")
    if c2st_b is not None:
        auc_str = f"{c2st_b['auc']:.3f}" if c2st_b["auc"] is not None else "N/A"
        print(f"  [Token] C2ST: {label_b} acc={c2st_b['acc']:.3f} auc={auc_str}")

    swd_a = compute_swd(h_real_tokens, h_a_tokens)
    swd_b = compute_swd(h_real_tokens, h_b_tokens)
    if swd_a is not None and swd_b is not None:
        print(f"  [Token] SWD: {label_a}={swd_a:.6f} | {label_b}={swd_b:.6f}")

    frechet_a = compute_frechet_distance(h_real_tokens, h_a_tokens)
    frechet_b = compute_frechet_distance(h_real_tokens, h_b_tokens)
    if frechet_a is not None and frechet_b is not None:
        print(f"  [Token] Frechet(PCA): {label_a}={frechet_a:.6f} | {label_b}={frechet_b:.6f}")

    # ===== Visualizations (token-level) =====
    max_vis = min(1000, len(h_real_tokens), len(h_a_tokens), len(h_b_tokens))
    vis_idx = np.random.choice(len(h_real_tokens), max_vis, replace=False)
    real_vis = h_real_tokens[vis_idx]
    a_vis = h_a_tokens[vis_idx]
    b_vis = h_b_tokens[vis_idx]

    # PCA
    pca = PCA(n_components=2)
    pca.fit(np.vstack([real_vis, a_vis, b_vis]))
    real_pca = pca.transform(real_vis)
    a_pca = pca.transform(a_vis)
    b_pca = pca.transform(b_vis)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(real_pca[:, 0], real_pca[:, 1], s=10, alpha=0.5, label="real", color="gray")
    ax.scatter(a_pca[:, 0], a_pca[:, 1], s=10, alpha=0.5, label=label_a)
    ax.scatter(b_pca[:, 0], b_pca[:, 1], s=10, alpha=0.5, label=label_b)
    ax.legend()
    ax.set_title(f"Layer {layer_idx} - PCA (token-level)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dist_compare_layer{layer_idx}_pca.png"), dpi=150)
    plt.close()

    # t-SNE (optional)
    if run_tsne:
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        emb = tsne.fit_transform(np.vstack([real_vis, a_vis, b_vis]))
        n0 = len(real_vis)
        n1 = len(a_vis)
        emb_r = emb[:n0]
        emb_a = emb[n0 : n0 + n1]
        emb_b = emb[n0 + n1 :]
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(emb_r[:, 0], emb_r[:, 1], s=10, alpha=0.5, label="real", color="gray")
        ax.scatter(emb_a[:, 0], emb_a[:, 1], s=10, alpha=0.5, label=label_a)
        ax.scatter(emb_b[:, 0], emb_b[:, 1], s=10, alpha=0.5, label=label_b)
        ax.legend()
        ax.set_title(f"Layer {layer_idx} - t-SNE (token-level)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"dist_compare_layer{layer_idx}_tsne.png"), dpi=150)
        plt.close()

    # 1D LDA projection (3-class)
    lda = LinearDiscriminantAnalysis(n_components=1)
    X = np.vstack([real_vis, a_vis, b_vis])
    y = np.array([0] * len(real_vis) + [1] * len(a_vis) + [2] * len(b_vis))
    proj = lda.fit_transform(X, y).reshape(-1)
    proj_r = proj[: len(real_vis)]
    proj_a = proj[len(real_vis) : len(real_vis) + len(a_vis)]
    proj_b = proj[len(real_vis) + len(a_vis) :]
    d_a = abs(cohens_d(proj_a, proj_r))
    d_b = abs(cohens_d(proj_b, proj_r))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(proj_r, bins=60, alpha=0.5, label="real", density=True, color="gray")
    ax.hist(proj_a, bins=60, alpha=0.5, label=f"{label_a} (d={d_a:.2f})", density=True)
    ax.hist(proj_b, bins=60, alpha=0.5, label=f"{label_b} (d={d_b:.2f})", density=True)
    ax.legend()
    ax.set_title(f"Layer {layer_idx} - 1D LDA (token-level)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dist_compare_layer{layer_idx}_lda1d.png"), dpi=150)
    plt.close()

    # per-sample (pooled) distance hists
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(per_l2_a, bins=60, alpha=0.5, label=f"{label_a} -> real", density=True)
    axes[0].hist(per_l2_b, bins=60, alpha=0.5, label=f"{label_b} -> real", density=True)
    axes[0].legend()
    axes[0].set_title(f"Layer {layer_idx} - {pooled_name}-level L2 distance")
    axes[1].hist(per_cos_a, bins=60, alpha=0.5, label=f"{label_a} -> real", density=True)
    axes[1].hist(per_cos_b, bins=60, alpha=0.5, label=f"{label_b} -> real", density=True)
    axes[1].legend()
    axes[1].set_title(f"Layer {layer_idx} - {pooled_name}-level cosine distance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dist_compare_layer{layer_idx}_{pooled_name}_dists.png"), dpi=150)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Checkpoint Real: {args.checkpoint_real}")
    print(f"Checkpoint A: {args.checkpoint_a}")
    print(f"Checkpoint B: {args.checkpoint_b}")
    print(f"Config Real: {args.config_real}")
    print(f"Config A: {args.config_a}")
    print(f"Config B: {args.config_b}")

    # Resolve skip_repair_layers for A and B
    print("\n=== Resolving skip_repair_layers ===")
    skip_repair_layers_a = _resolve_skip_repair_layers(args.checkpoint_a, args.skip_repair_layers_a)
    skip_repair_layers_b = _resolve_skip_repair_layers(args.checkpoint_b, args.skip_repair_layers_b)
    if skip_repair_layers_a:
        print(f"  Checkpoint A: will skip repair layers {skip_repair_layers_a}")
    if skip_repair_layers_b:
        print(f"  Checkpoint B: will skip repair layers {skip_repair_layers_b}")

    # load samples (use real config for dataset settings)
    samples = load_samples(args.num_samples, args.config_real)
    print(f"Loaded {len(samples)} samples")

    processor = load_processor(args.model_path)

    # preprocess once on CPU
    prepared = []
    for s in samples:
        prepared.append(preprocess_sample(s, processor, max_length=args.max_length))

    # Run Real (baseline)
    print("\n=== Load & run Real baseline ===")
    if str(args.checkpoint_real).strip().lower() in {"pretrained", "hf", "base"}:
        print(f"  Real baseline source: pretrained ({args.model_path})")
        model_real, _ = _build_prunable_llava_from_pretrained(
            config_path=args.config_real,
            device=device,
            model_path=args.model_path,
        )
    else:
        model_real, _ = load_model_from_checkpoint(args.checkpoint_real, args.config_real, device, args.model_path, skip_repair_layers=None)
    num_layers_total = _infer_num_decoder_layers(model_real)

    # Resolve capture layers (may depend on model depth when using 'all' / open ranges)
    # Composite defaults to capturing all layers for the summary curve.
    capture_layers_spec = (args.capture_layers or "").strip()
    if args.composite and not capture_layers_spec:
        capture_layers_spec = "all"
    if capture_layers_spec:
        capture_layers = _parse_layer_list(capture_layers_spec, num_layers=num_layers_total)
    else:
        from engine.configs.loader import load_config

        cfg_real = load_config(override_file=args.config_real)
        ms = cfg_real["method_settings"]
        capture_layers = list(ms.get("repair_layers", []) or [])
        if not capture_layers:
            capture_layers = list(ms.get("pruning_layers", []) or [])

    capture_layers = [int(x) for x in capture_layers]
    if not capture_layers:
        raise ValueError("No capture_layers provided and config_real has empty repair_layers/pruning_layers.")
    print(f"Capture layers: {capture_layers}")

    # Load config_real for annotations / defaults
    from engine.configs.loader import load_config

    cfg_real = load_config(override_file=args.config_real)
    ms_real = cfg_real.get("method_settings", {}) or {}
    repair_layers_cfg = [int(x) for x in (ms_real.get("repair_layers", []) or [])]
    pruning_layers_cfg = [int(x) for x in (ms_real.get("pruning_layers", []) or [])]

    if args.summary_only or args.composite:
        os.makedirs(args.output_dir, exist_ok=True)
        # Real pooled (keep-all, no repair)
        real_pooled = collect_pooled_only(
            model=model_real,
            prepared_samples=prepared,
            device=device,
            capture_layers=capture_layers,
            apply_repair=False,
            keep_all=True,
            label=args.label_real,
        )
        del model_real
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # A pooled
        print("\n=== Load & run A (summary) ===")
        model_a, _ = load_model_from_checkpoint(args.checkpoint_a, args.config_a, device, args.model_path, skip_repair_layers=skip_repair_layers_a)
        apply_repair_a = bool(getattr(model_a, "use_repair_adapter", False))
        a_pooled = collect_pooled_only(
            model=model_a,
            prepared_samples=prepared,
            device=device,
            capture_layers=capture_layers,
            apply_repair=apply_repair_a,
            keep_all=False,
            label=args.label_a,
        )
        del model_a
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # B pooled
        print("\n=== Load & run B (summary) ===")
        model_b, _ = load_model_from_checkpoint(args.checkpoint_b, args.config_b, device, args.model_path, skip_repair_layers=skip_repair_layers_b)
        apply_repair_b = bool(getattr(model_b, "use_repair_adapter", False))
        b_pooled = collect_pooled_only(
            model=model_b,
            prepared_samples=prepared,
            device=device,
            capture_layers=capture_layers,
            apply_repair=apply_repair_b,
            keep_all=False,
            label=args.label_b,
        )
        del model_b
        if device.type == "cuda":
            torch.cuda.empty_cache()

        rows = _save_layerwise_curves(
            capture_layers=capture_layers,
            real=real_pooled,
            a=a_pooled,
            b=b_pooled,
            output_dir=args.output_dir,
            label_a=args.label_a,
            label_b=args.label_b,
            repair_layers=repair_layers_cfg,
            pruning_layers=pruning_layers_cfg,
        ) or []

        if repair_layers_cfg:
            _save_repair_layer_bars(
                rows=rows,
                repair_layers=repair_layers_cfg,
                output_dir=args.output_dir,
                label_a=args.label_a,
                label_b=args.label_b,
            )

        if args.summary_only and not args.composite:
            print(f"\nDone. Saved summary to {args.output_dir}")
            return

        # Composite deep-dive: per-layer figures on selected key layers
        print("\n=== Composite deep-dive ===")
        key_layers = set()
        key_layers.update(repair_layers_cfg)
        key_layers.update(pruning_layers_cfg)
        # Add top-k gap layers from summary rows
        if rows:
            sorted_by_gap = sorted(
                rows,
                key=lambda r: float(r["mean_l2_b"]) - float(r["mean_l2_a"]),
                reverse=True,
            )
            for r in sorted_by_gap[: max(int(args.deep_dive_topk), 0)]:
                key_layers.add(int(r["layer"]))

        # Neighbor expansion
        nb = max(int(args.deep_dive_neighbors), 0)
        expanded = set()
        for l in key_layers:
            for d in range(-nb, nb + 1):
                expanded.add(int(l) + d)
        key_layers = {l for l in expanded if 0 <= int(l) < int(num_layers_total)}

        # Override with user-specified deep_dive_layers
        deep_spec = (args.deep_dive_layers or "").strip()
        if deep_spec:
            key_layers = set(_parse_layer_list(deep_spec, num_layers=num_layers_total))

        key_layers = sorted([int(x) for x in key_layers])
        if not key_layers:
            print("  No deep-dive layers selected; composite mode will stop after summary.")
            print(f"\nDone. Saved composite summary to {args.output_dir}")
            return

        deep_dir = os.path.join(args.output_dir, "deep_dive")
        os.makedirs(deep_dir, exist_ok=True)
        print(f"Deep-dive layers: {key_layers}")

        # Real (tokens) for deep-dive layers
        print("\n=== Load & run Real baseline (deep-dive) ===")
        if str(args.checkpoint_real).strip().lower() in {"pretrained", "hf", "base"}:
            model_real2, _ = _build_prunable_llava_from_pretrained(
                config_path=args.config_real,
                device=device,
                model_path=args.model_path,
            )
        else:
            model_real2, _ = load_model_from_checkpoint(args.checkpoint_real, args.config_real, device, args.model_path, skip_repair_layers=None)
        data_real = collect_h_real_only(
            model=model_real2,
            prepared_samples=prepared,
            device=device,
            capture_layers=key_layers,
        )
        del model_real2
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # A deep-dive
        print("\n=== Load & run A (deep-dive) ===")
        model_a2, _ = load_model_from_checkpoint(args.checkpoint_a, args.config_a, device, args.model_path, skip_repair_layers=skip_repair_layers_a)
        apply_repair_a2 = bool(getattr(model_a2, "use_repair_adapter", False))
        data_a = collect_h_pred_only(
            model=model_a2,
            prepared_samples=prepared,
            device=device,
            capture_layers=key_layers,
            apply_repair_pred=apply_repair_a2,
        )
        del model_a2
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # B deep-dive
        print("\n=== Load & run B (deep-dive) ===")
        model_b2, _ = load_model_from_checkpoint(args.checkpoint_b, args.config_b, device, args.model_path, skip_repair_layers=skip_repair_layers_b)
        apply_repair_b2 = bool(getattr(model_b2, "use_repair_adapter", False))
        data_b = collect_h_pred_only(
            model=model_b2,
            prepared_samples=prepared,
            device=device,
            capture_layers=key_layers,
            apply_repair_pred=apply_repair_b2,
        )
        del model_b2
        if device.type == "cuda":
            torch.cuda.empty_cache()

        run_tsne = not args.no_tsne
        for layer_idx in key_layers:
            if layer_idx not in data_real or layer_idx not in data_a or layer_idx not in data_b:
                print(f"Skip layer {layer_idx}: missing data")
                continue
            real_tokens = data_real[layer_idx]["real_tokens"]
            real_pooled = data_real[layer_idx]["real_pooled"]
            if real_tokens is None:
                print(f"Skip layer {layer_idx}: missing real baseline")
                continue
            analyze_and_visualize_compare(
                layer_idx=layer_idx,
                h_real_tokens=real_tokens,
                h_a_tokens=data_a[layer_idx]["pred_tokens"],
                h_b_tokens=data_b[layer_idx]["pred_tokens"],
                h_real_pooled=real_pooled,
                h_a_pooled=data_a[layer_idx]["pred_pooled"],
                h_b_pooled=data_b[layer_idx]["pred_pooled"],
                output_dir=deep_dir,
                label_a=args.label_a,
                label_b=args.label_b,
                run_tsne=run_tsne,
            )

        print(f"\nDone. Saved composite report to {args.output_dir}")
        return

    data_real = collect_h_real_only(
        model=model_real,
        prepared_samples=prepared,
        device=device,
        capture_layers=capture_layers,
    )
    del model_real
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run A
    print("\n=== Load & run A ===")
    model_a, _ = load_model_from_checkpoint(args.checkpoint_a, args.config_a, device, args.model_path, skip_repair_layers=skip_repair_layers_a)
    apply_repair_a = bool(getattr(model_a, "use_repair_adapter", False))
    data_a = collect_h_pred_only(
        model=model_a,
        prepared_samples=prepared,
        device=device,
        capture_layers=capture_layers,
        apply_repair_pred=apply_repair_a,
    )
    del model_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Run B
    print("\n=== Load & run B ===")
    model_b, _ = load_model_from_checkpoint(args.checkpoint_b, args.config_b, device, args.model_path, skip_repair_layers=skip_repair_layers_b)
    apply_repair_b = bool(getattr(model_b, "use_repair_adapter", False))
    data_b = collect_h_pred_only(
        model=model_b,
        prepared_samples=prepared,
        device=device,
        capture_layers=capture_layers,
        apply_repair_pred=apply_repair_b,
    )
    del model_b
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Analyze
    os.makedirs(args.output_dir, exist_ok=True)
    run_tsne = not args.no_tsne
    for layer_idx in capture_layers:
        if layer_idx not in data_real or layer_idx not in data_a or layer_idx not in data_b:
            print(f"Skip layer {layer_idx}: missing data")
            continue

        real_tokens = data_real[layer_idx]["real_tokens"]
        real_pooled = data_real[layer_idx]["real_pooled"]
        if real_tokens is None:
            print(f"Skip layer {layer_idx}: missing real baseline")
            continue

        analyze_and_visualize_compare(
            layer_idx=layer_idx,
            h_real_tokens=real_tokens,
            h_a_tokens=data_a[layer_idx]["pred_tokens"],
            h_b_tokens=data_b[layer_idx]["pred_tokens"],
            h_real_pooled=real_pooled,
            h_a_pooled=data_a[layer_idx]["pred_pooled"],
            h_b_pooled=data_b[layer_idx]["pred_pooled"],
            output_dir=args.output_dir,
            label_a=args.label_a,
            label_b=args.label_b,
            run_tsne=run_tsne,
        )

    print(f"\nDone. Saved figures to {args.output_dir}")


if __name__ == "__main__":
    main()
