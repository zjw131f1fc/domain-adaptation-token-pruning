#!/usr/bin/env python3
"""
Estimate an optimistic latency upper bound (ms) for vision-token pruning.

This script is for *estimation*, not correctness:
  - We keep the vision tower as a constant cost (same as origin).
  - We estimate the LLM cost under an "ideal packing" assumption:
      * compute image features once (not timed)
      * feed the language model a *shorter* sequence (text + K vision tokens)
      * keep the language-model forward path (SDPA/FlashAttention, etc.) untouched
  - We measure pruner + deployed delayed-repair adapter as "constant overhead" modules,
    and add them on top of the packed-LLM estimate.

Outputs:
  - origin latency (model.generate, max_new_tokens=1)
  - packed LLM latency for K in targets (prefill only, includes lm_head + argmax)
  - estimated non-LLM constant = origin - packed_llm(K=576)
  - pruner+adapter constant (measured)
  - predicted upper bound: nonLLM + packed_llm(K) + pruner_adapter_const
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

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

    # deployed delayed repair adapter
    if getattr(model, "use_repair_adapter", False):
        if "repair_context_encoder_state_dict" in ckpt and getattr(model, "repair_context_encoder", None) is not None:
            model.repair_context_encoder.load_state_dict(ckpt["repair_context_encoder_state_dict"])
        if "repair_adapter_state_dict" in ckpt and getattr(model, "repair_adapter_manager", None) is not None:
            model.repair_adapter_manager.load_state_dict(ckpt["repair_adapter_state_dict"])

    return ckpt


def _build_single_input_dummy(*, processor, device: torch.device, max_length: int, question: str, image_size: int):
    from engine.data_utils import preprocess_batch
    from PIL import Image

    img = Image.effect_noise((int(image_size), int(image_size)), 64.0).convert("RGB")
    sample = {"image": img, "question": str(question)}
    pre = preprocess_batch([sample], processor, device, max_length=max_length, mode="inference")
    return pre, sample


def _compute_image_features(*, base_mm, base_cfg, pixel_values: torch.Tensor) -> torch.Tensor:
    image_features = base_mm.get_image_features(
        pixel_values=pixel_values,
        vision_feature_layer=base_cfg.vision_feature_layer,
        vision_feature_select_strategy=base_cfg.vision_feature_select_strategy,
    )
    # transformers>=5.2: BaseModelOutputWithPooling, projected features in .pooler_output
    if not torch.is_tensor(image_features) and hasattr(image_features, "pooler_output"):
        image_features = image_features.pooler_output
    # list/tuple compat
    if isinstance(image_features, (list, tuple)):
        if len(image_features) > 0 and torch.is_tensor(image_features[0]) and image_features[0].dim() == 2:
            image_features = torch.stack(list(image_features), dim=0)
        else:
            image_features = torch.cat(list(image_features), dim=0)
    if not torch.is_tensor(image_features):
        raise TypeError(f"Unexpected image_features type: {type(image_features)}")
    return image_features


def _time_cuda(
    fn,
    *,
    warmup: int,
    iters: int,
) -> Dict[str, LatencyStats]:
    wall_ms: List[float] = []
    cuda_ms: List[float] = []

    use_cuda = torch.cuda.is_available()
    start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None

    with torch.inference_mode():
        for _ in range(int(warmup)):
            fn()

        for _ in range(int(iters)):
            if use_cuda:
                torch.cuda.synchronize()
                assert start_event is not None and end_event is not None
                start_event.record()
            t0 = time.perf_counter()
            fn()
            if use_cuda:
                assert end_event is not None
                end_event.record()
                end_event.synchronize()
                cuda_ms.append(float(start_event.elapsed_time(end_event)))
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            wall_ms.append((t1 - t0) * 1000.0)

    out: Dict[str, LatencyStats] = {"wall": _summarize_ms(wall_ms)}
    if cuda_ms:
        out["cuda"] = _summarize_ms(cuda_ms)
    return out


def _run_origin_generate(*, model, inputs: Dict[str, torch.Tensor], max_new_tokens: int) -> None:
    _ = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )


def _run_llm_prefill_one_token(
    *,
    llm,
    lm_head,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
) -> None:
    out = llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    hs = out.last_hidden_state
    logits = lm_head(hs[:, -1:, :])
    _ = logits.argmax(dim=-1)


def _parse_targets(arg: str) -> List[int]:
    items = []
    for part in str(arg).replace(" ", "").split(","):
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise ValueError("Empty --targets.")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate latency upper bound by packed-LLM + constants.")
    parser.add_argument("--config", type=str, default="configs/vision_token_pruning-64.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--targets", type=str, default="64,128,192", help="Comma-separated K (vision tokens kept).")
    parser.add_argument("--max-new-tokens", type=int, default=1, help="Only 1 is meaningful for this estimator.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--question", type=str, default="What is in the image?")
    parser.add_argument("--image-size", type=int, default=336)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {args.config}")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    targets = _parse_targets(args.targets)
    max_new_tokens = int(args.max_new_tokens)
    if max_new_tokens != 1:
        raise ValueError("This estimator currently targets max_new_tokens=1 (prefill + 1 token).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config + model
    from engine.configs.loader import load_config
    from main_acp_ddp import load_model

    config = load_config(override_file=str(config_path))
    model, processor = load_model(config, device, local_rank=0)
    _load_checkpoint(model, str(ckpt_path), device)

    # deterministic-ish
    method_cfg = config.method_settings
    eval_temp = method_cfg.get("eval_temperature", method_cfg.get("temperature_min", 0.1))
    eval_threshold = method_cfg.get("eval_pruning_threshold", method_cfg.get("pruning_threshold", 0.5))
    model.set_temperature(float(eval_temp))
    model.set_pruning_threshold(float(eval_threshold))
    model.set_use_gumbel_noise(False)
    model.eval()

    # build one input; processor runs once (not timed)
    pre, _sample = _build_single_input_dummy(
        processor=processor,
        device=device,
        max_length=int(args.max_length),
        question=str(args.question),
        image_size=int(args.image_size),
    )
    inputs = pre["inputs"]
    input_ids: torch.Tensor = inputs["input_ids"]
    attention_mask: torch.Tensor = inputs.get("attention_mask", torch.ones_like(input_ids))
    pixel_values: torch.Tensor = inputs["pixel_values"]

    vision_start = int(pre["vision_start"])
    vision_end = int(pre["vision_end"])
    n_vision = int(vision_end - vision_start)
    seq_len = int(input_ids.shape[1])
    n_text = int(seq_len - n_vision)

    # base model internals
    base = model.base_model
    base_mm = base.model
    llm = base_mm.language_model
    lm_head = base.lm_head

    # compute full image features once (not timed)
    with torch.inference_mode():
        image_features_full = _compute_image_features(base_mm=base_mm, base_cfg=base.config, pixel_values=pixel_values)
        # shape should be (batch, 576, hidden)
        if image_features_full.dim() != 3:
            raise ValueError(f"Expected image_features to be 3D (b,n,h), got shape={tuple(image_features_full.shape)}")
        image_features_full = image_features_full.to(device=device, dtype=next(base.parameters()).dtype)

    # Precompute text embeds (not timed)
    with torch.inference_mode():
        token_embed = base_mm.get_input_embeddings()
        before_ids = input_ids[:, :vision_start]
        after_ids = input_ids[:, vision_end:]
        before_embeds = token_embed(before_ids)
        after_embeds = token_embed(after_ids)

    def _make_packed_embeds(K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        K = int(K)
        if K < 0 or K > n_vision:
            raise ValueError(f"Invalid K={K}, expected 0..{n_vision}")
        embeds = torch.cat([before_embeds, image_features_full[:, :K, :], after_embeds], dim=1)
        attn = torch.ones(embeds.shape[0], embeds.shape[1], device=embeds.device, dtype=attention_mask.dtype)
        return embeds, attn

    # 1) origin total latency (vision + llm, optimized generate)
    origin_stats = _time_cuda(
        lambda: _run_origin_generate(model=model, inputs=inputs, max_new_tokens=max_new_tokens),
        warmup=int(args.warmup),
        iters=int(args.iters),
    )

    # 2) packed-LLM latency for K=576 (full length, but NO vision tower)
    embeds_full, attn_full = _make_packed_embeds(n_vision)
    llm_full_stats = _time_cuda(
        lambda: _run_llm_prefill_one_token(llm=llm, lm_head=lm_head, inputs_embeds=embeds_full, attention_mask=attn_full),
        warmup=int(args.warmup),
        iters=int(args.iters),
    )

    # estimated non-LLM constant (mostly vision tower + multimodal glue)
    origin_cuda_mean = origin_stats.get("cuda", origin_stats["wall"]).mean_ms
    llm_full_cuda_mean = llm_full_stats.get("cuda", llm_full_stats["wall"]).mean_ms
    non_llm_const_ms = float(origin_cuda_mean - llm_full_cuda_mean)

    # 3) pruner + deployed adapter "constant"
    dtype = next(model.parameters()).dtype
    hidden_size = int(model.hidden_size)
    q_len = int(pre["question_ends"][0] - pre["question_starts"][0])
    q_len = max(q_len, 1)

    # reuse image_features_full as a realistic vision_hidden (shape matches hidden_size)
    vision_hidden = image_features_full.to(dtype=dtype)
    if vision_hidden.shape[2] != hidden_size:
        # fallback: generate random with correct hidden size
        vision_hidden = torch.randn(vision_hidden.shape[0], n_vision, hidden_size, device=device, dtype=dtype)

    q2v_attn = torch.rand(vision_hidden.shape[0], n_vision, device=device, dtype=dtype)
    q2v_attn = q2v_attn / q2v_attn.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    question_hidden = torch.randn(vision_hidden.shape[0], q_len, hidden_size, device=device, dtype=dtype)
    question_lengths = torch.tensor([q_len] * int(vision_hidden.shape[0]), device=device, dtype=torch.long)

    # prebuild a "sparse-ish" cumulative mask and a single-token hidden for adapter
    keep_ratio = float(max(min(targets[0], n_vision), 1) / n_vision)
    cumulative_mask_const = torch.zeros(vision_hidden.shape[0], n_vision, device=device, dtype=dtype)
    n_keep_const = max(1, int(round(keep_ratio * n_vision)))
    cumulative_mask_const[:, :n_keep_const] = 1
    x_const = torch.randn(vision_hidden.shape[0], 1, hidden_size, device=device, dtype=dtype)

    def _run_pruner_and_adapter_const() -> None:
        # pruners + context encoders at pruning layers
        last_mask_emb = None
        last_pruned_emb = None
        for p_layer in getattr(model, "pruning_layers", []) or []:
            pruner = model.pruner_manager.get_pruner(int(p_layer))
            _mask, _info = pruner.forward_full(
                vision_hidden,
                q2v_attn,
                cumulative_vision_mask=cumulative_mask_const,
                question_hidden=question_hidden,
                question_lengths=question_lengths,
                n_pruned_tokens=0,
            )
            if getattr(model, "repair_context_encoder", None) is not None:
                last_mask_emb, last_pruned_emb = model.repair_context_encoder(vision_hidden, cumulative_mask_const)

        # adapters at repair layers (seq_len=1, "constant" wrt prompt length)
        if getattr(model, "repair_adapter_manager", None) is not None and getattr(model, "repair_layers", None):
            for r_layer in model.repair_layers:
                adapter = model.repair_adapter_manager.get_adapter(int(r_layer))
                _ = adapter(
                    x_const,
                    mask=None,
                    query=x_const,
                    mask_emb=last_mask_emb,
                    pruned_emb=last_pruned_emb,
                )

    const_stats = _time_cuda(_run_pruner_and_adapter_const, warmup=int(args.warmup), iters=int(args.iters))
    pruner_adapter_const_ms = const_stats.get("cuda", const_stats["wall"]).mean_ms

    # 4) packed LLM per target K
    packed_results: Dict[int, Dict[str, LatencyStats]] = {}
    for K in targets:
        embeds_k, attn_k = _make_packed_embeds(int(K))
        packed_results[int(K)] = _time_cuda(
            lambda embeds=embeds_k, attn=attn_k: _run_llm_prefill_one_token(
                llm=llm, lm_head=lm_head, inputs_embeds=embeds, attention_mask=attn
            ),
            warmup=int(args.warmup),
            iters=int(args.iters),
        )

    # printing
    print("== Latency Upper Bound Estimate ==")
    print(f"device: {device}")
    print(f"prompt: seq_len={seq_len}, n_text={n_text}, n_vision={n_vision}, vision_start={vision_start}")
    print(f"max_new_tokens={max_new_tokens}, warmup={int(args.warmup)}, iters={int(args.iters)}")
    print(f"targets(K): {targets}")
    print("")

    def _fmt(stats: Dict[str, LatencyStats]) -> str:
        s = stats.get("cuda", stats["wall"])
        return f"mean={s.mean_ms:.3f} ms (p50={s.p50_ms:.3f}, p90={s.p90_ms:.3f}, p99={s.p99_ms:.3f})"

    print(f"[origin generate] { _fmt(origin_stats) }")
    print(f"[packed llm K=576] { _fmt(llm_full_stats) }")
    print(f"[non-LLM constant] {non_llm_const_ms:.3f} ms  (= origin - packed_llm(576))")
    print(f"[pruner+adapter constant] { _fmt(const_stats) }")
    print("")

    origin_mean = origin_cuda_mean
    for K in targets:
        llm_k_mean = packed_results[int(K)].get("cuda", packed_results[int(K)]["wall"]).mean_ms
        pred = float(non_llm_const_ms + llm_k_mean + pruner_adapter_const_ms)
        speedup = origin_mean / pred if pred > 0 else float("nan")
        ratio = pred / origin_mean if origin_mean > 0 else float("nan")
        print(f"K={int(K):4d}  packed_llm={llm_k_mean:8.3f} ms  pred_total={pred:8.3f} ms  speedup_vs_origin={speedup:6.3f}x  (ratio={ratio:.3f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
