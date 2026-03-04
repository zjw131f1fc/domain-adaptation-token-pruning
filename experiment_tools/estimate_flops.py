#!/usr/bin/env python3
"""
FLOPs 估算工具（不依赖实际运行 / 不做 profiling）。

目标：
- 给定 pruning layers + (cumulative) kept ratios，估算 Transformer 的理论 FLOPs（Prefill + Decode）。
- 可选计入：CrossAttentionPruner、Delayed-Repair 相关模块的额外 FLOPs。

重要假设（请务必确认你要评估的部署路径）：
1) mode=physical（默认）：
   - 在 pruning layer 之后“物理删除” vision tokens（序列长度真的变短）
   - 后续层的 attention/MLP FLOPs 会显著下降（O(L^2) -> smaller L）
2) mode=mask_only：
   - 不物理删除 tokens，只做 post-softmax masking
   - Transformer 主干 FLOPs 基本不变（仍按完整 seq_len 计算），只有 pruner/adapter overhead 变化

说明：
- 这里的 FLOPs 是“理论乘加计数”近似：1 MAC（乘加）按 2 FLOPs 计。
- 不包含 softmax、LayerNorm、rope 等低阶开销（通常不是主要项）。

默认参数（可通过命令行覆盖）：
- 视觉侧常数项：CLIP ViT-L/14-336（24 层，seq_len=577 含 CLS）+ LLaVA mm_projector=mlp2x
- LLM 侧：LLaVA-1.5-7B（32 层）
- token 长度：text_len=10（问题+模板等文本 token，不含视觉），gen_len=1（答案生成 1 token）

历史参考（仍保留常量，便于对照）：
- VQA v2 val[0] inference prompt tokenize 后：n_vision=576，text_len=27，总 seq_len=603
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


VQAV2_VAL0_N_VISION = 576
VQAV2_VAL0_TEXT_LEN_INFER = 27
VQAV2_VAL0_SEQ_LEN_INFER = VQAV2_VAL0_N_VISION + VQAV2_VAL0_TEXT_LEN_INFER  # 603


@dataclass(frozen=True)
class VisionEncoderSpec:
    """A minimal ViT-like vision encoder spec (e.g., CLIP ViT-L/14-336)."""

    num_layers: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    # ViT sequence length includes CLS (patches + 1). For 336/14: 24*24+1=577.
    seq_len: int = 577
    # Patch embedding (conv/linear) parameters for a rough estimate.
    include_patch_embed: bool = False
    patch_size: int = 14
    in_channels: int = 3
    image_size: int = 336


def _paper_linear_fit_params(
    *,
    upper_token: float,
    upper_tflops: float,
    low_token: float,
    low_tflops: float,
) -> Tuple[float, float]:
    """Fit TFLOPs = a + b * token (two-point fit)."""
    if abs(upper_token - low_token) < 1e-12:
        raise ValueError("upper_token and low_token must be different for a linear fit.")
    b = (upper_tflops - low_tflops) / (upper_token - low_token)
    a = upper_tflops - b * upper_token
    return float(a), float(b)


def _paper_linear_tflops(
    token: float,
    *,
    upper_token: float,
    upper_tflops: float,
    low_token: float,
    low_tflops: float,
) -> float:
    a, b = _paper_linear_fit_params(
        upper_token=upper_token,
        upper_tflops=upper_tflops,
        low_token=low_token,
        low_tflops=low_tflops,
    )
    return a + b * float(token)


def _parse_csv_ints(s: str) -> List[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _parse_csv_floats(s: str) -> List[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def _format_flops(flops: float) -> str:
    # flops is a number (not necessarily int) in FLOPs
    if flops >= 1e12:
        return f"{flops / 1e12:.4f} TFLOPs"
    if flops >= 1e9:
        return f"{flops / 1e9:.4f} GFLOPs"
    if flops >= 1e6:
        return f"{flops / 1e6:.4f} MFLOPs"
    return f"{flops:.0f} FLOPs"


@dataclass(frozen=True)
class ModelSpec:
    num_layers: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008  # SwiGLU intermediate
    num_heads: int = 32
    num_kv_heads: int = 32
    head_dim: int = 128


@dataclass(frozen=True)
class PrunerSpec:
    d_internal: int = 512
    n_heads: int = 4
    n_queries: int = 16


@dataclass(frozen=True)
class RepairSpec:
    bottleneck_dim: int = 512
    # Mask encoder uses a small d_pos (as in repo)
    mask_pos_dim: int = 64
    # Whether adapter gets cached embeddings (mask_emb/pruned_emb)
    adapter_uses_cached_ctx: bool = True
    # Whether to include RepairContextEncoder overhead at pruning layers
    include_context_encoder: bool = True
    # Whether to include repair adapter overhead at repair layers
    include_repair_adapter: bool = True


def _validate_pruning_inputs(
    layers: Sequence[int],
    ratios: Sequence[float],
    total_layers: int,
) -> Tuple[List[int], List[float]]:
    if len(layers) != len(ratios):
        raise ValueError(f"layers length ({len(layers)}) must match ratios length ({len(ratios)})")
    pairs = list(zip([int(x) for x in layers], [float(r) for r in ratios]))
    pairs.sort(key=lambda x: x[0])

    sorted_layers = [p[0] for p in pairs]
    sorted_ratios = [p[1] for p in pairs]

    for i, layer_idx in enumerate(sorted_layers):
        if layer_idx < 0 or layer_idx >= total_layers:
            raise ValueError(f"Invalid pruning layer index {layer_idx}, expected 0..{total_layers-1}")
        if i > 0 and layer_idx == sorted_layers[i - 1]:
            raise ValueError(f"Duplicate pruning layer index: {layer_idx}")

    for r in sorted_ratios:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"Invalid cumulative ratio: {r}, expected [0,1]")

    # cumulative ratios should be non-increasing across depth (pruning accumulates)
    for i in range(1, len(sorted_ratios)):
        if sorted_ratios[i] > sorted_ratios[i - 1] + 1e-9:
            raise ValueError(
                "Cumulative ratios must be non-increasing with deeper layers. "
                f"Got {sorted_ratios[i-1]} -> {sorted_ratios[i]}."
            )

    return sorted_layers, sorted_ratios


# -----------------------------
# FLOPs formulas (theory)
# -----------------------------

def transformer_prefill_layer_flops(seq_len: int, spec: ModelSpec) -> float:
    """Theoretical FLOPs for one decoder layer prefill with full self-attention over seq_len tokens."""
    h = spec.hidden_size
    inter = spec.intermediate_size
    nh = spec.num_heads
    nkv = spec.num_kv_heads
    d = spec.head_dim

    # Projections: Q, K, V, O
    # Q: (seq, h) x (h, nh*d) -> 2*seq*h*(nh*d)
    q_proj = 2.0 * seq_len * h * (nh * d)
    # K/V: (seq,h) x (h, nkv*d) each
    kv_proj = 2.0 * 2.0 * seq_len * h * (nkv * d)
    # O: (seq, nh*d) x (nh*d, h)
    o_proj = 2.0 * seq_len * (nh * d) * h

    # Attention matmuls: QK^T and Attn*V
    # QK^T: (nh, seq, d) x (nh, d, seq) -> 2*nh*seq*seq*d
    qk = 2.0 * nh * seq_len * seq_len * d
    # Attn*V: same shape
    av = 2.0 * nh * seq_len * seq_len * d

    attn = q_proj + kv_proj + o_proj + qk + av

    # MLP (SwiGLU): gate, up, down each ~ 2*seq*h*inter
    mlp = 6.0 * seq_len * h * inter

    return attn + mlp


def transformer_decode_layer_flops(kv_len: int, spec: ModelSpec) -> float:
    """Theoretical FLOPs for one decoder layer for ONE generated token with KV cache length kv_len."""
    # For the new token: projections are seq_len=1
    h = spec.hidden_size
    inter = spec.intermediate_size
    nh = spec.num_heads
    nkv = spec.num_kv_heads
    d = spec.head_dim

    # projections for 1 token
    q_proj = 2.0 * 1.0 * h * (nh * d)
    kv_proj = 2.0 * 2.0 * 1.0 * h * (nkv * d)
    o_proj = 2.0 * 1.0 * (nh * d) * h

    # attention matmuls scale with kv_len (query_len=1)
    qk = 2.0 * nh * 1.0 * kv_len * d
    av = 2.0 * nh * 1.0 * kv_len * d

    attn = q_proj + kv_proj + o_proj + qk + av
    mlp = 6.0 * 1.0 * h * inter
    return attn + mlp


def vit_encoder_layer_flops(seq_len: int, spec: VisionEncoderSpec) -> float:
    """Approx FLOPs for one ViT encoder layer (full self-attention + 2-layer MLP)."""
    d = spec.hidden_size
    dff = spec.intermediate_size
    nh = spec.num_heads
    if d % nh != 0:
        raise ValueError(f"vision hidden_size ({d}) must be divisible by num_heads ({nh})")
    head_dim = d // nh

    # QKV + O projections: 4 linears with (seq, d) x (d, d)
    proj = 2.0 * seq_len * d * d * 4.0

    # Attention matmuls: QK^T and Attn*V
    qk = 2.0 * nh * seq_len * seq_len * head_dim
    av = 2.0 * nh * seq_len * seq_len * head_dim

    # MLP: two linears (d->dff) + (dff->d)
    mlp = 2.0 * seq_len * d * dff + 2.0 * seq_len * dff * d

    return proj + qk + av + mlp


def vit_patch_embed_flops(spec: VisionEncoderSpec) -> float:
    """Approx FLOPs for patch embedding (linear/conv) stage.

    We model it as: for each patch, a linear map from (p*p*in_ch) -> hidden_size.
    """
    if not spec.include_patch_embed:
        return 0.0
    if spec.patch_size <= 0 or spec.image_size <= 0:
        return 0.0
    if spec.image_size % spec.patch_size != 0:
        # Keep it simple: if image size isn't divisible, skip patch embed estimate.
        return 0.0
    n_patches = (spec.image_size // spec.patch_size) ** 2
    in_dim = spec.patch_size * spec.patch_size * spec.in_channels
    return 2.0 * n_patches * in_dim * spec.hidden_size


def llava_mm_projector_flops(
    n_vision_tokens: int,
    *,
    vision_dim: int,
    llm_hidden_size: int,
    projector_type: str,
) -> float:
    """Approx FLOPs for LLaVA mm_projector mapping vision features -> LLM hidden.

    projector_type:
      - 'linear': one Linear(vision_dim -> llm_hidden_size)
      - 'mlp2x': Linear(vision_dim -> llm_hidden_size) + Linear(llm_hidden_size -> llm_hidden_size)
    """
    if n_vision_tokens <= 0:
        return 0.0
    if projector_type == "none":
        return 0.0
    if projector_type == "linear":
        return 2.0 * n_vision_tokens * vision_dim * llm_hidden_size
    if projector_type == "mlp2x":
        first = 2.0 * n_vision_tokens * vision_dim * llm_hidden_size
        second = 2.0 * n_vision_tokens * llm_hidden_size * llm_hidden_size
        return first + second
    raise ValueError(f"Unknown projector_type: {projector_type}")


def cross_attention_pruner_flops(n_vision: int, model: ModelSpec, pruner: PrunerSpec) -> float:
    """Approx FLOPs for CrossAttentionPruner forward (vision_proj + MHA + token_scorer + query_agg)."""
    d_model = model.hidden_size
    d_internal = pruner.d_internal
    n_queries = pruner.n_queries
    n_heads = pruner.n_heads
    if d_internal % n_heads != 0:
        raise ValueError(f"pruner d_internal ({d_internal}) must be divisible by n_heads ({n_heads})")
    head_dim = d_internal // n_heads

    # vision_proj: (n_vision, d_model) -> (n_vision, d_internal)
    vision_proj = 2.0 * n_vision * d_model * d_internal

    # MultiheadAttention internal projections (approx like torch nn.MultiheadAttention):
    # Q proj for queries: (n_queries, d_internal) -> (n_queries, d_internal)
    q_proj = 2.0 * n_queries * d_internal * d_internal
    # K/V proj for vision tokens
    kv_proj = 2.0 * 2.0 * n_vision * d_internal * d_internal
    # attention matmuls
    qk = 2.0 * n_heads * n_queries * n_vision * head_dim
    av = 2.0 * n_heads * n_queries * n_vision * head_dim
    # output proj for queries
    o_proj = 2.0 * n_queries * d_internal * d_internal
    cross_attn = q_proj + kv_proj + qk + av + o_proj

    # token_scorer: Linear(d_internal->d_internal) + Linear(d_internal->1)
    token_scorer = 2.0 * n_vision * d_internal * d_internal + 2.0 * n_vision * d_internal

    # query aggregator: per token, linear (n_queries -> 1)
    query_agg = 2.0 * n_vision * n_queries

    return vision_proj + cross_attn + token_scorer + query_agg


def repair_context_encoder_flops(n_vision: int, model: ModelSpec, repair: RepairSpec) -> float:
    """Approx FLOPs for RepairContextEncoder (mask attention pooling + pruned aggregator proj). Batch assumed 1."""
    # MaskAttentionEncoder: attention pooling over (n_vision, d_pos) then out_proj (d_pos->bottleneck)
    d_pos = repair.mask_pos_dim
    bottleneck = repair.bottleneck_dim

    # attention scores (1 x d_pos) @ (d_pos x n_vision) and weighted sum
    # 2*n_vision*d_pos each, twice -> 4*n_vision*d_pos
    attn_pool = 4.0 * n_vision * d_pos
    out_proj = 2.0 * d_pos * bottleneck

    # pruned aggregator projection: (hidden_size -> bottleneck)
    pruned_proj = 2.0 * model.hidden_size * bottleneck

    return attn_pool + out_proj + pruned_proj


def delayed_repair_adapter_flops(seq_len: int, model: ModelSpec, repair: RepairSpec) -> float:
    """Approx FLOPs for LightweightAdapter when using cached ctx (mask_emb/pruned_emb).

    When cached ctx is provided, the adapter skips mask_encoder and pruned_aggregator.
    """
    h = model.hidden_size
    b = repair.bottleneck_dim

    # down + query_proj + up (all are linear)
    down = 2.0 * seq_len * h * b
    query_proj = 2.0 * seq_len * h * b
    up = 2.0 * seq_len * b * h

    # gamma_net + beta_net: two linears (b->b) applied on seq positions
    film = 2.0 * 2.0 * seq_len * b * b

    # If not cached, add mask_encoder + pruned_aggregator costs, but default is cached.
    extra = 0.0
    if not repair.adapter_uses_cached_ctx:
        # crude upper bound: mask_encoder ~ O(n_vision*d_pos + d_pos*b) + pruned aggregator proj
        # Note: n_vision is unknown here; caller should not use this mode unless they know what they want.
        extra = 0.0

    return down + query_proj + film + up + extra


@dataclass(frozen=True)
class FlopsResult:
    origin_prefill: float
    origin_decode: float
    origin_total: float
    pruned_prefill: float
    pruned_decode: float
    pruned_total: float
    reduction_pct: float
    speedup_ideal: float
    avg_vision_tokens_per_layer: float
    vision_encoder: float
    mm_projector: float


def estimate_flops(
    *,
    model: ModelSpec,
    n_vision: int,
    text_len: int,
    gen_len: int,
    pruning_layers: Sequence[int],
    cumulative_ratios: Sequence[float],
    mode: str,
    include_pruner: bool,
    pruner: PrunerSpec,
    repair_layers: Sequence[int],
    repair: RepairSpec,
    include_vision_encoder: bool,
    vision: VisionEncoderSpec,
    include_mm_projector: bool,
    mm_projector_type: str,
    mm_vision_dim: int,
) -> FlopsResult:
    total_layers = model.num_layers
    layers, ratios = _validate_pruning_inputs(pruning_layers, cumulative_ratios, total_layers)

    # --- vision encoder FLOPs (constant per example, independent of pruning schedule) ---
    vision_encoder = 0.0
    if include_vision_encoder:
        vision_encoder = vit_patch_embed_flops(vision) + sum(
            vit_encoder_layer_flops(vision.seq_len, vision) for _ in range(int(vision.num_layers))
        )

    mm_projector = 0.0
    if include_mm_projector:
        mm_projector = llava_mm_projector_flops(
            n_vision_tokens=int(n_vision),
            vision_dim=int(mm_vision_dim),
            llm_hidden_size=int(model.hidden_size),
            projector_type=str(mm_projector_type),
        )

    # --- origin FLOPs ---
    origin_seq = text_len + n_vision
    origin_prefill = vision_encoder + mm_projector + sum(
        transformer_prefill_layer_flops(origin_seq, model) for _ in range(total_layers)
    )

    # Decode: average kv length (prompt + average generated history)
    # avg_kv_len ~ prompt_len + gen_len/2
    origin_avg_kv = origin_seq + max(int(gen_len // 2), 0)
    origin_decode_per_token = sum(transformer_decode_layer_flops(origin_avg_kv, model) for _ in range(total_layers))
    origin_decode = origin_decode_per_token * max(int(gen_len), 0)
    origin_total = origin_prefill + origin_decode

    # --- pruned FLOPs ---
    pruned_prefill = vision_encoder + mm_projector
    pruned_decode = 0.0

    # physical mode: update current_n_vision after pruning layers
    # mask_only: keep seq_len constant for Transformer, but still compute pruner/repair overheads
    current_n_vision = int(n_vision)
    prev_cum = 1.0

    # Compute avg_vision_tokens_per_layer (32-layer average, aligned with your "Avg.Token" notion)
    # We treat cumulative ratio at layer idx applies to layers [layer_idx..next_prune-1], and pre layers keep 1.0.
    if not layers:
        avg_ratio = 1.0
    else:
        seg0 = layers[0]
        seg_lengths = [seg0]
        for i in range(len(layers) - 1):
            seg_lengths.append(layers[i + 1] - layers[i])
        seg_lengths.append(total_layers - layers[-1])
        weighted = float(seg_lengths[0]) * 1.0
        for i, r in enumerate(ratios):
            weighted += float(seg_lengths[i + 1]) * float(r)
        avg_ratio = weighted / float(total_layers)
    avg_vision_tokens_per_layer = avg_ratio * float(n_vision)

    layers_set = set(layers)
    repair_layers_set = set(int(x) for x in (repair_layers or []))

    # For mask_only, Transformer seq stays at origin_seq.
    fixed_seq_len = origin_seq

    # Prefill: layer-by-layer walk
    for layer_idx in range(total_layers):
        seq_len = (text_len + current_n_vision) if mode == "physical" else fixed_seq_len
        pruned_prefill += transformer_prefill_layer_flops(seq_len, model)

        # Extra modules at specific layers
        if include_pruner and (layer_idx in layers_set):
            pruned_prefill += cross_attention_pruner_flops(
                n_vision=current_n_vision if mode == "physical" else n_vision,
                model=model,
                pruner=pruner,
            )
            # delayed repair: context encoder runs at pruning layers to cache ctx
            if repair.include_context_encoder:
                pruned_prefill += repair_context_encoder_flops(
                    n_vision=current_n_vision if mode == "physical" else n_vision,
                    model=model,
                    repair=repair,
                )

        if repair.include_repair_adapter and (layer_idx in repair_layers_set):
            # delayed repair adapter runs at repair layers; FLOPs scale with seq_len at that layer
            pruned_prefill += delayed_repair_adapter_flops(seq_len=seq_len, model=model, repair=repair)

        # Update vision tokens AFTER pruning layer (only for physical mode)
        if mode == "physical" and (layer_idx in layers_set):
            # relative ratio = cum / prev_cum
            cum = ratios[layers.index(layer_idx)]
            rel = cum / max(prev_cum, 1e-12)
            prev_cum = cum
            current_n_vision = int(current_n_vision * rel)

    # Decode: approximate avg kv len depends on prompt length.
    if mode == "physical" and layers:
        final_n_vision = int(n_vision * ratios[-1])
    else:
        final_n_vision = int(n_vision)
    prompt_len_pruned = text_len + final_n_vision if mode == "physical" else origin_seq
    avg_kv_pruned = prompt_len_pruned + max(int(gen_len // 2), 0)

    pruned_decode_per_token_llm = sum(transformer_decode_layer_flops(avg_kv_pruned, model) for _ in range(total_layers))
    pruned_decode = pruned_decode_per_token_llm * max(int(gen_len), 0)

    # Add decode-time overhead for repair adapter at repair layers (seq_len=1 per token)
    # (If your deployment uses full-forward decode, this underestimates repair overhead.)
    if repair.include_repair_adapter and repair_layers_set and gen_len > 0:
        per_token_repair = 0.0
        for _ in repair_layers_set:
            per_token_repair += delayed_repair_adapter_flops(seq_len=1, model=model, repair=repair)
        pruned_decode += per_token_repair * int(gen_len)

    pruned_total = pruned_prefill + pruned_decode

    saved = origin_total - pruned_total
    reduction_pct = (saved / origin_total * 100.0) if origin_total > 0 else 0.0
    speedup = (origin_total / pruned_total) if pruned_total > 0 else float("inf")

    return FlopsResult(
        origin_prefill=origin_prefill,
        origin_decode=origin_decode,
        origin_total=origin_total,
        pruned_prefill=pruned_prefill,
        pruned_decode=pruned_decode,
        pruned_total=pruned_total,
        reduction_pct=float(reduction_pct),
        speedup_ideal=float(speedup),
        avg_vision_tokens_per_layer=float(avg_vision_tokens_per_layer),
        vision_encoder=float(vision_encoder),
        mm_projector=float(mm_projector),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Estimate theoretical FLOPs for token pruning configurations.")
    p.add_argument("--layers", type=str, default="", help="Pruning layer indices (0-based), comma-separated.")
    p.add_argument("--ratios", type=str, default="", help="Cumulative kept ratios (relative to original n_vision), comma-separated.")
    p.add_argument("--mode", type=str, default="physical", choices=["physical", "mask_only"], help="Pruning mode assumption.")

    p.add_argument(
        "--n-vision",
        type=int,
        default=VQAV2_VAL0_N_VISION,
        help=f"Number of vision tokens before pruning. Default={VQAV2_VAL0_N_VISION} (LLaVA-1.5 style).",
    )
    p.add_argument(
        "--text-len",
        type=int,
        default=10,
        help=(
            "Prompt text token count (excluding vision tokens). "
            "Default=10 (question~10 tokens)."
        ),
    )
    p.add_argument("--gen-len", type=int, default=1, help="Number of generated tokens (decode length). Default=1.")

    # model spec
    p.add_argument("--num-layers", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=4096)
    p.add_argument("--intermediate-size", type=int, default=11008)
    p.add_argument("--num-heads", type=int, default=32)
    p.add_argument("--num-kv-heads", type=int, default=32)
    p.add_argument("--head-dim", type=int, default=128)

    # pruner
    p.add_argument("--include-pruner", action="store_true", help="Include CrossAttentionPruner overhead.")
    p.add_argument("--pruner-d-internal", type=int, default=512)
    p.add_argument("--pruner-n-heads", type=int, default=4)
    p.add_argument("--pruner-n-queries", type=int, default=16)

    # repair
    p.add_argument("--repair-layers", type=str, default="", help="Repair adapter layer indices (0-based), comma-separated.")
    p.add_argument("--include-repair-context", action="store_true", help="Include RepairContextEncoder overhead at pruning layers.")
    p.add_argument("--include-repair-adapter", action="store_true", help="Include delayed repair adapter overhead at repair layers.")
    p.add_argument("--repair-bottleneck", type=int, default=512)
    p.add_argument("--repair-pos-dim", type=int, default=64)

    # vision encoder (ViT-like, e.g., CLIP ViT-L/14-336)
    p.add_argument(
        "--include-vision-encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include a ViT-like vision encoder FLOPs (constant). Default: enabled.",
    )
    p.add_argument("--vision-num-layers", type=int, default=24)
    p.add_argument("--vision-hidden-size", type=int, default=1024)
    p.add_argument("--vision-intermediate-size", type=int, default=4096)
    p.add_argument("--vision-num-heads", type=int, default=16)
    p.add_argument("--vision-seq-len", type=int, default=577, help="Vision encoder token count including CLS (default 577 for 336/14).")
    p.add_argument("--vision-image-size", type=int, default=336)
    p.add_argument("--vision-patch-size", type=int, default=14)
    p.add_argument(
        "--vision-include-patch-embed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include patch embedding FLOPs (small). Default: disabled.",
    )

    # mm_projector
    p.add_argument(
        "--include-mm-projector",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include LLaVA mm_projector FLOPs (constant). Default: enabled.",
    )
    p.add_argument("--mm-projector-type", type=str, default="mlp2x", choices=["none", "linear", "mlp2x"])
    p.add_argument("--mm-vision-dim", type=int, default=1024, help="Vision feature dim before projector (default 1024 for CLIP ViT-L).")

    # paper alignment helper (ToMe/SparseVLM tables often look linear in Avg.Token)
    p.add_argument(
        "--paper-fit",
        action="store_true",
        help=(
            "Also report a 'paper-aligned' TFLOPs value by fitting a linear model "
            "TFLOPs = a + b * token using two anchor points (upper + low). "
            "This is useful for matching ratios in papers where FLOPs are reported in a different accounting unit."
        ),
    )
    p.add_argument(
        "--paper-upper-token",
        type=float,
        default=576.0,
        help="Anchor token count for upper bound (default: 576).",
    )
    p.add_argument(
        "--paper-upper-tflops",
        type=float,
        default=4.62,
        help="Anchor TFLOPs at upper bound token (default: 4.62).",
    )
    p.add_argument(
        "--paper-low-token",
        type=float,
        default=64.0,
        help="Anchor token count for low point (default: 64).",
    )
    p.add_argument(
        "--paper-low-tflops",
        type=float,
        default=1.19,
        help="Anchor TFLOPs at low point token (default: 1.19).",
    )
    p.add_argument(
        "--paper-token",
        type=float,
        default=None,
        help=(
            "Override token used for paper-fit reporting. "
            "If omitted, uses avg_vision_tokens_per_layer from the pruning schedule."
        ),
    )

    args = p.parse_args()

    layers = _parse_csv_ints(args.layers)
    ratios = _parse_csv_floats(args.ratios)
    repair_layers = _parse_csv_ints(args.repair_layers)

    model = ModelSpec(
        num_layers=int(args.num_layers),
        hidden_size=int(args.hidden_size),
        intermediate_size=int(args.intermediate_size),
        num_heads=int(args.num_heads),
        num_kv_heads=int(args.num_kv_heads),
        head_dim=int(args.head_dim),
    )
    pruner = PrunerSpec(
        d_internal=int(args.pruner_d_internal),
        n_heads=int(args.pruner_n_heads),
        n_queries=int(args.pruner_n_queries),
    )
    repair = RepairSpec(
        bottleneck_dim=int(args.repair_bottleneck),
        mask_pos_dim=int(args.repair_pos_dim),
        include_context_encoder=bool(args.include_repair_context),
        include_repair_adapter=bool(args.include_repair_adapter),
        adapter_uses_cached_ctx=True,
    )
    vision = VisionEncoderSpec(
        num_layers=int(args.vision_num_layers),
        hidden_size=int(args.vision_hidden_size),
        intermediate_size=int(args.vision_intermediate_size),
        num_heads=int(args.vision_num_heads),
        seq_len=int(args.vision_seq_len),
        include_patch_embed=bool(args.vision_include_patch_embed),
        patch_size=int(args.vision_patch_size),
        image_size=int(args.vision_image_size),
    )

    res = estimate_flops(
        model=model,
        n_vision=int(args.n_vision),
        text_len=int(args.text_len),
        gen_len=int(args.gen_len),
        pruning_layers=layers,
        cumulative_ratios=ratios,
        mode=str(args.mode),
        include_pruner=bool(args.include_pruner),
        pruner=pruner,
        repair_layers=repair_layers,
        repair=repair,
        include_vision_encoder=bool(args.include_vision_encoder),
        vision=vision,
        include_mm_projector=bool(args.include_mm_projector),
        mm_projector_type=str(args.mm_projector_type),
        mm_vision_dim=int(args.mm_vision_dim),
    )

    print("== FLOPs Estimate ==")
    print(f"mode: {args.mode}")
    print(f"n_vision: {args.n_vision}, text_len: {args.text_len}, gen_len: {args.gen_len}")
    if int(args.n_vision) == VQAV2_VAL0_N_VISION and int(args.text_len) == VQAV2_VAL0_TEXT_LEN_INFER:
        print(f"preset: vqav2_val0_infer (prompt seq_len={VQAV2_VAL0_SEQ_LEN_INFER})")
    if layers:
        print(f"pruning_layers: {layers}")
        print(f"cumulative_ratios: {ratios}")
    if args.include_pruner:
        print(f"include_pruner: True (d_internal={pruner.d_internal}, n_heads={pruner.n_heads}, n_queries={pruner.n_queries})")
    else:
        print("include_pruner: False")
    if repair_layers and (args.include_repair_adapter or args.include_repair_context):
        print(f"repair_layers: {repair_layers}")
        print(f"include_repair_context: {bool(args.include_repair_context)}")
        print(f"include_repair_adapter: {bool(args.include_repair_adapter)} (bottleneck={repair.bottleneck_dim})")

    print("")
    print(f"avg_vision_tokens_per_layer (32-layer avg): {res.avg_vision_tokens_per_layer:.3f}")

    if bool(args.include_vision_encoder) or bool(args.include_mm_projector):
        print("")
        print("[Vision Side (constant)]")
        if bool(args.include_vision_encoder):
            print(
                "  vision_encoder: "
                f"{_format_flops(res.vision_encoder)} "
                f"(layers={vision.num_layers}, seq_len={vision.seq_len}, hidden={vision.hidden_size}, mlp={vision.intermediate_size})"
            )
        if bool(args.include_mm_projector):
            print(
                "  mm_projector:   "
                f"{_format_flops(res.mm_projector)} "
                f"(type={args.mm_projector_type}, n_vision={args.n_vision}, vision_dim={args.mm_vision_dim} -> llm_hidden={model.hidden_size})"
            )

    print("")
    print("[Origin]")
    print(f"  Prefill: {_format_flops(res.origin_prefill)}")
    print(f"  Decode:  {_format_flops(res.origin_decode)}")
    print(f"  Total:   {_format_flops(res.origin_total)}")

    print("")
    print("[Pruned]")
    print(f"  Prefill: {_format_flops(res.pruned_prefill)}")
    print(f"  Decode:  {_format_flops(res.pruned_decode)}")
    print(f"  Total:   {_format_flops(res.pruned_total)}")

    print("")
    print("[Comparison]")
    print(f"  Reduction: {res.reduction_pct:.2f}%")
    print(f"  Ideal speedup (FLOPs-based): {res.speedup_ideal:.3f}x")

    if bool(args.paper_fit):
        token_for_paper = float(args.paper_token) if args.paper_token is not None else float(res.avg_vision_tokens_per_layer)
        upper_token = float(args.paper_upper_token)
        upper_tflops = float(args.paper_upper_tflops)
        low_token = float(args.paper_low_token)
        low_tflops = float(args.paper_low_tflops)

        a, b = _paper_linear_fit_params(
            upper_token=upper_token,
            upper_tflops=upper_tflops,
            low_token=low_token,
            low_tflops=low_tflops,
        )
        fitted = _paper_linear_tflops(
            token_for_paper,
            upper_token=upper_token,
            upper_tflops=upper_tflops,
            low_token=low_token,
            low_tflops=low_tflops,
        )
        upper_val = _paper_linear_tflops(
            upper_token,
            upper_token=upper_token,
            upper_tflops=upper_tflops,
            low_token=low_token,
            low_tflops=low_tflops,
        )
        ratio = fitted / upper_val if upper_val != 0 else float("nan")

        print("")
        print("[Paper-Fit (linear TFLOPs = a + b*token)]")
        print(f"  anchors: ({upper_token:.0f} -> {upper_tflops:.4f}T), ({low_token:.0f} -> {low_tflops:.4f}T)")
        print(f"  fitted params: a={a:.6f}, b={b:.9f} (TFLOPs/token)")
        print(f"  token_used: {token_for_paper:.3f}")
        print(f"  paper_equiv: {fitted:.4f} TFLOPs (ratio vs upper: {ratio:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
