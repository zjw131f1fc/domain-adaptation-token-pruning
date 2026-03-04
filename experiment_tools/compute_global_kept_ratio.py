#!/usr/bin/env python3
"""
计算“对 32 层平均”的全局保留率（avg_kept_ratio）。

背景（与本仓库训练/评估一致）：
- pruning_layers 是 0-based 的 decoder layer index（例如 [1,13,25]）
- 传入的 kept ratio 应该是每个 pruning layer 处的 cumulative_mask.mean()（日志里的 L{layer}_kept）
- 全局平均保留率定义为：对所有层按层数加权平均

公式（total_layers=T）：
avg = (n0*1.0 + n1*r0 + n2*r1 + ... + nK*r{K-1}) / T
其中：
  n0 = pruning_layers[0]                      # 剪枝前的层数，默认 100% 保留
  n1 = pruning_layers[1] - pruning_layers[0]  # 第一段受 r0 影响
  ...
  nK = T - pruning_layers[-1]                 # 最后一段受 r_last 影响

示例：
  python experiment_tools/compute_global_kept_ratio.py --layers 1,13,25 --ratios 0.70,0.45,0.30
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class GlobalKeptResult:
    total_layers: int
    layers: List[int]
    ratios: List[float]
    segment_lengths: List[int]
    avg_kept_ratio: float


def _parse_csv_ints(s: str) -> List[int]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _parse_csv_floats(s: str) -> List[float]:
    parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def _validate_and_sort_pairs(
    layers: Sequence[int],
    ratios: Sequence[float],
    total_layers: int,
) -> Tuple[List[int], List[float]]:
    if total_layers <= 0:
        raise ValueError(f"total_layers must be > 0, got {total_layers}")

    if len(layers) != len(ratios):
        raise ValueError(f"layers length ({len(layers)}) must match ratios length ({len(ratios)})")

    if len(layers) == 0:
        return [], []

    pairs = list(zip([int(x) for x in layers], [float(r) for r in ratios]))
    pairs.sort(key=lambda x: x[0])

    sorted_layers = [p[0] for p in pairs]
    sorted_ratios = [p[1] for p in pairs]

    # basic validity
    for i, layer_idx in enumerate(sorted_layers):
        if layer_idx < 0 or layer_idx >= total_layers:
            raise ValueError(
                f"Invalid layer index: {layer_idx}. Expected 0 <= layer < total_layers({total_layers})."
            )
        if i > 0 and layer_idx == sorted_layers[i - 1]:
            raise ValueError(f"Duplicate layer index: {layer_idx}")

    for r in sorted_ratios:
        if not (0.0 <= r <= 1.0):
            raise ValueError(f"Invalid ratio: {r}. Expected 0.0 <= ratio <= 1.0.")

    return sorted_layers, sorted_ratios


def compute_global_avg_kept_ratio(
    *,
    pruning_layers: Sequence[int],
    cumulative_kept_ratios: Sequence[float],
    total_layers: int = 32,
) -> GlobalKeptResult:
    layers, ratios = _validate_and_sort_pairs(pruning_layers, cumulative_kept_ratios, total_layers)

    # No pruning layers => all layers keep 100%
    if not layers:
        return GlobalKeptResult(
            total_layers=int(total_layers),
            layers=[],
            ratios=[],
            segment_lengths=[int(total_layers)],
            avg_kept_ratio=1.0,
        )

    segment_lengths: List[int] = []
    segment_lengths.append(int(layers[0]))
    for i in range(len(layers) - 1):
        segment_lengths.append(int(layers[i + 1] - layers[i]))
    segment_lengths.append(int(total_layers - layers[-1]))

    if any(x < 0 for x in segment_lengths):
        raise ValueError(f"Computed negative segment length(s): {segment_lengths}. Check layers/total_layers.")
    if sum(segment_lengths) != total_layers:
        # 逻辑上必须满足，否则就是输入/边界有问题
        raise ValueError(
            f"Segment lengths must sum to total_layers={total_layers}, got {segment_lengths} (sum={sum(segment_lengths)})"
        )

    weighted = float(segment_lengths[0]) * 1.0
    for i, r in enumerate(ratios):
        weighted += float(segment_lengths[i + 1]) * float(r)
    avg = weighted / float(total_layers)

    return GlobalKeptResult(
        total_layers=int(total_layers),
        layers=list(layers),
        ratios=list(ratios),
        segment_lengths=segment_lengths,
        avg_kept_ratio=float(avg),
    )


def _format_percent(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="输入剪枝层层号 + (cumulative)保留率，计算“对 32 层平均”的全局保留率 avg_kept_ratio。",
    )
    parser.add_argument(
        "--layers",
        type=str,
        required=True,
        help="剪枝层 layer indices（0-based），逗号分隔。例如：1,13,25",
    )
    parser.add_argument(
        "--ratios",
        type=str,
        required=True,
        help="每个剪枝层对应的 cumulative kept ratio（0~1），逗号分隔。例如：0.70,0.45,0.30",
    )
    parser.add_argument(
        "--total-layers",
        type=int,
        default=32,
        help="总层数，默认 32（LLaVA/LLaMA-7B decoder layers）。",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="只输出一个数字（avg_kept_ratio，小数）。",
    )

    args = parser.parse_args()

    layers = _parse_csv_ints(args.layers)
    ratios = _parse_csv_floats(args.ratios)

    res = compute_global_avg_kept_ratio(
        pruning_layers=layers,
        cumulative_kept_ratios=ratios,
        total_layers=int(args.total_layers),
    )

    if args.quiet:
        print(f"{res.avg_kept_ratio:.8f}")
        return 0

    print("== Global Avg Kept Ratio ==")
    print(f"total_layers: {res.total_layers}")
    print(f"pruning_layers (sorted): {res.layers}")
    print(f"cumulative_kept_ratios:  {res.ratios}")
    if res.layers:
        print(f"segment_lengths:        {res.segment_lengths}  # sums to {sum(res.segment_lengths)}")
        # 解释一下每段含义，方便快速核对
        # seg0: pre-pruning
        print(f"segment[0] (layers 0..{res.layers[0]-1}): keep=1.0")
        for i, layer_idx in enumerate(res.layers):
            if i < len(res.layers) - 1:
                nxt = res.layers[i + 1]
            else:
                nxt = res.total_layers
            seg_len = res.segment_lengths[i + 1]
            print(f"segment[{i+1}] (layers {layer_idx}..{nxt-1}): len={seg_len}, keep={res.ratios[i]}")
    print(f"avg_kept_ratio: {res.avg_kept_ratio:.6f} ({_format_percent(res.avg_kept_ratio)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

