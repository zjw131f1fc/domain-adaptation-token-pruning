#!/usr/bin/env python3
"""
Latency benchmark (ms) for LLaVA token-pruning variants.

Goals:
  - Measure model-side latency (exclude processor/preprocess overhead).
  - Support hard mode (physical pruning) with deployed delayed-repair adapter.
  - Report wall-clock latency (ms) and, if CUDA is available, GPU event time (ms).

Typical usage (single GPU):
  python scripts/benchmark_latency.py \\
    --config configs/sweeps/vision_token_pruning_target64.yaml \\
    --checkpoint outputs/checkpoints/checkpoint_final.pt \\
    --mode hard \\
    --max-new-tokens 1 \\
    --warmup 10 \\
    --iters 50

Notes:
  - `--mode hard` calls `model.generate_with_hard_pruning()` (physical token deletion).
  - `--mode origin` calls `model.generate()` (no pruning).
  - For short generation latency, set `--max-new-tokens 1` (prefill + 1 token).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Keep the same defaults as eval scripts, but do not override user env if already set.
os.environ.setdefault("HF_HOME", "/data/users/zjw/huggingface_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if q <= 0:
        return float(sorted_vals[0])
    if q >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (q / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(sorted_vals[f])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


@dataclass(frozen=True)
class LatencyStats:
    n: int
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float


def _summarize_ms(values_ms: List[float]) -> LatencyStats:
    vals = [float(x) for x in values_ms]
    vals.sort()
    n = len(vals)
    mean = sum(vals) / max(n, 1)
    return LatencyStats(
        n=n,
        mean_ms=mean,
        p50_ms=_percentile(vals, 50),
        p90_ms=_percentile(vals, 90),
        p99_ms=_percentile(vals, 99),
        min_ms=vals[0] if vals else float("nan"),
        max_ms=vals[-1] if vals else float("nan"),
    )


def _load_checkpoint(model, checkpoint_path: str, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "pruner_state_dict" in ckpt:
        model.pruner_manager.load_state_dict(ckpt["pruner_state_dict"])
    if "disc_state_dict" in ckpt:
        model.disc_manager.load_state_dict(ckpt["disc_state_dict"])

    # delayed repair modules (deployed adapter)
    if getattr(model, "use_repair_adapter", False):
        if "repair_context_encoder_state_dict" in ckpt and getattr(model, "repair_context_encoder", None) is not None:
            model.repair_context_encoder.load_state_dict(ckpt["repair_context_encoder_state_dict"])
        if "repair_adapter_state_dict" in ckpt and getattr(model, "repair_adapter_manager", None) is not None:
            model.repair_adapter_manager.load_state_dict(ckpt["repair_adapter_state_dict"])
    return ckpt


def _build_single_input_from_dataset(
    *,
    config: Any,
    processor,
    device: torch.device,
    max_length: int,
    split: str,
    sample_idx: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from engine.datas.loader import load_dataset
    from engine.data_utils import preprocess_batch

    bundle = load_dataset(config)
    if split not in bundle["splits"]:
        raise KeyError(f"Split {split!r} not in dataset bundle. Available: {list(bundle['splits'].keys())}")
    dataset = bundle["splits"][split]
    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx out of range: {sample_idx}, dataset size={len(dataset)}")
    sample = dataset[sample_idx]
    pre = preprocess_batch([sample], processor, device, max_length=max_length, mode="inference")
    return pre, sample


def _build_single_input_dummy(
    *,
    processor,
    device: torch.device,
    max_length: int,
    question: str,
    image_size: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from engine.data_utils import preprocess_batch
    from PIL import Image

    # A deterministic synthetic image (cheap to generate, good enough for latency micro-benchmark).
    img = Image.effect_noise((int(image_size), int(image_size)), 64.0).convert("RGB")
    sample = {"image": img, "question": str(question)}
    pre = preprocess_batch([sample], processor, device, max_length=max_length, mode="inference")
    return pre, sample


def _run_one(
    *,
    model,
    mode: str,
    pre: Dict[str, Any],
    max_new_tokens: int,
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    inputs = pre["inputs"]

    if mode == "hard":
        output, stats = model.generate_with_hard_pruning(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            vision_start=pre["vision_start"],
            vision_end=pre["vision_end"],
            question_starts=pre["question_starts"],
            question_ends=pre["question_ends"],
            max_new_tokens=int(max_new_tokens),
        )
        return output, stats

    if mode == "hard_baseline":
        # Same hard-route (manual decode loop) but:
        # - keep_all: no physical pruning
        # - no pruner calls
        # - no delayed repair adapter calls
        output, stats = model.generate_with_hard_pruning(
            input_ids=inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            vision_start=pre["vision_start"],
            vision_end=pre["vision_end"],
            question_starts=pre["question_starts"],
            question_ends=pre["question_ends"],
            hard_pruning_mode="keep_all",
            apply_pruner=False,
            apply_repair=False,
            max_new_tokens=int(max_new_tokens),
        )
        return output, stats

    if mode == "origin":
        output = model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )
        return output, None

    raise ValueError(f"Unknown mode: {mode!r} (expected 'hard' or 'origin')")


def _bench(
    *,
    model,
    mode: str,
    pre: Dict[str, Any],
    max_new_tokens: int,
    warmup: int,
    iters: int,
) -> Dict[str, Any]:
    model.eval()

    # warmup
    with torch.inference_mode():
        for _ in range(int(warmup)):
            _run_one(model=model, mode=mode, pre=pre, max_new_tokens=max_new_tokens)

    wall_ms: List[float] = []
    cuda_ms: List[float] = []

    use_cuda_timing = torch.cuda.is_available() and (next(model.parameters()).is_cuda)
    start_event = torch.cuda.Event(enable_timing=True) if use_cuda_timing else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda_timing else None

    with torch.inference_mode():
        for _ in range(int(iters)):
            if use_cuda_timing:
                torch.cuda.synchronize()
                assert start_event is not None and end_event is not None
                start_event.record()

            t0 = time.perf_counter()
            _, stats = _run_one(model=model, mode=mode, pre=pre, max_new_tokens=max_new_tokens)

            if use_cuda_timing:
                assert end_event is not None
                end_event.record()
                end_event.synchronize()
                cuda_ms.append(float(start_event.elapsed_time(end_event)))

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            wall_ms.append((t1 - t0) * 1000.0)

    out: Dict[str, Any] = {
        "mode": mode,
        "max_new_tokens": int(max_new_tokens),
        "warmup": int(warmup),
        "iters": int(iters),
        "wall_ms": _summarize_ms(wall_ms).__dict__,
    }
    if cuda_ms:
        out["cuda_ms"] = _summarize_ms(cuda_ms).__dict__

    if stats is not None:
        # Keep only lightweight keys; floats/ints only.
        compact = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float))}
        out["kept_stats"] = compact

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark latency (ms) for hard/origin inference.")
    parser.add_argument("--config", type=str, default="configs/vision_token_pruning.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["hard", "hard_baseline", "origin", "both", "all"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)

    # input selection
    parser.add_argument("--use-dataset", action="store_true", help="Use dataset sample from config (more realistic).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=2048)

    # dummy input (when not using dataset)
    parser.add_argument("--question", type=str, default="What is in the image?")
    parser.add_argument("--image-size", type=int, default=336)

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config + model
    from engine.configs.loader import load_config
    from main_acp_ddp import load_model

    config = load_config(override_file=str(config_path))
    model, processor = load_model(config, device, local_rank=0)
    _load_checkpoint(model, str(ckpt_path), device)

    # eval settings for pruning behavior (make it deterministic)
    method_cfg = config.method_settings
    eval_temp = method_cfg.get("eval_temperature", method_cfg.get("temperature_min", 0.1))
    eval_threshold = method_cfg.get("eval_pruning_threshold", method_cfg.get("pruning_threshold", 0.5))
    model.set_temperature(float(eval_temp))
    model.set_pruning_threshold(float(eval_threshold))
    model.set_use_gumbel_noise(False)

    # build a single input (processor runs once; we do not time it)
    if args.use_dataset:
        pre, sample = _build_single_input_from_dataset(
            config=config,
            processor=processor,
            device=device,
            max_length=int(args.max_length),
            split=str(args.split),
            sample_idx=int(args.sample_idx),
        )
        sample_info = {"source": "dataset", "split": str(args.split), "sample_idx": int(args.sample_idx)}
    else:
        pre, sample = _build_single_input_dummy(
            processor=processor,
            device=device,
            max_length=int(args.max_length),
            question=str(args.question),
            image_size=int(args.image_size),
        )
        sample_info = {"source": "dummy", "image_size": int(args.image_size), "question": str(args.question)}

    # run benchmark
    if str(args.mode) == "both":
        modes = ["hard", "origin"]
    elif str(args.mode) == "all":
        modes = ["hard", "hard_baseline", "origin"]
    else:
        modes = [str(args.mode)]
    results_by_mode: Dict[str, Dict[str, Any]] = {}
    for m in modes:
        results_by_mode[m] = _bench(
            model=model,
            mode=m,
            pre=pre,
            max_new_tokens=int(args.max_new_tokens),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

    print("== Latency Benchmark ==")
    print(f"device: {device}")
    print(f"modes: {modes}, max_new_tokens={int(args.max_new_tokens)}")
    print(f"sample: {sample_info}")
    print(f"eval_temperature: {float(eval_temp)}, eval_threshold: {float(eval_threshold)}")

    def _print_one(label: str, r: Dict[str, Any]) -> None:
        print("")
        print(f"-- {label} --")
        wall = r["wall_ms"]
        print("[Wall-clock]")
        print(
            f"  n={wall['n']}  mean={wall['mean_ms']:.3f} ms  "
            f"p50={wall['p50_ms']:.3f}  p90={wall['p90_ms']:.3f}  p99={wall['p99_ms']:.3f}  "
            f"min={wall['min_ms']:.3f}  max={wall['max_ms']:.3f}"
        )
        if "cuda_ms" in r:
            cuda = r["cuda_ms"]
            print("[CUDA events]")
            print(
                f"  n={cuda['n']}  mean={cuda['mean_ms']:.3f} ms  "
                f"p50={cuda['p50_ms']:.3f}  p90={cuda['p90_ms']:.3f}  p99={cuda['p99_ms']:.3f}  "
                f"min={cuda['min_ms']:.3f}  max={cuda['max_ms']:.3f}"
            )

        if "kept_stats" in r:
            ks = r["kept_stats"]
            key_order = [
                "avg_kept_ratio",
                "original_n_vision",
                "final_n_kept",
            ]
            extras = []
            for k in key_order:
                if k in ks:
                    extras.append(f"{k}={ks[k]:.6f}" if isinstance(ks[k], float) else f"{k}={ks[k]}")
            if extras:
                print("[Kept Stats]")
                print("  " + ", ".join(extras))

    for m in modes:
        _print_one(m, results_by_mode[m])

    if "origin" in results_by_mode and len(results_by_mode) > 1:
        print("")
        print("[Speedup vs origin]")
        o = results_by_mode["origin"]
        o_mean = float(o["wall_ms"]["mean_ms"])
        o_cuda = float(o["cuda_ms"]["mean_ms"]) if "cuda_ms" in o else None

        for m in modes:
            if m == "origin":
                continue
            r = results_by_mode[m]
            m_mean = float(r["wall_ms"]["mean_ms"])
            if o_mean > 0 and m_mean > 0:
                speedup = o_mean / m_mean  # >1 means m is faster than origin
                ratio = m_mean / o_mean
                if ratio <= 1.0:
                    print(f"  {m}: wall_mean {speedup:.3f}x  (latency {(1.0 - ratio) * 100.0:.2f}% lower)")
                else:
                    print(f"  {m}: wall_mean {speedup:.3f}x  (latency {(ratio - 1.0) * 100.0:.2f}% higher)")

            if o_cuda is not None and "cuda_ms" in r:
                m_cuda = float(r["cuda_ms"]["mean_ms"])
                if o_cuda > 0 and m_cuda > 0:
                    speedup_cuda = o_cuda / m_cuda
                    ratio_cuda = m_cuda / o_cuda
                    if ratio_cuda <= 1.0:
                        print(f"  {m}: cuda_mean {speedup_cuda:.3f}x  (latency {(1.0 - ratio_cuda) * 100.0:.2f}% lower)")
                    else:
                        print(f"  {m}: cuda_mean {speedup_cuda:.3f}x  (latency {(ratio_cuda - 1.0) * 100.0:.2f}% higher)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
