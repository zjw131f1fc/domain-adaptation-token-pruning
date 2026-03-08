#!/usr/bin/env python
"""Repair gain figure using diagonal-Gaussian 2-Wasserstein metric (W2^2).

This mirrors `scripts/plot_repair_gain_for_paper.py` but replaces the metric:
  OFF/ON alignment = w2_sq_off / w2_sq_on
and plots relative gain (%) computed from W2^2.
"""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def set_paper_style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 8.8,
            "axes.labelsize": 9.6,
            "axes.titlesize": 9.6,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.labelpad": 1.8,
            "axes.linewidth": 0.9,
            "xtick.major.size": 3.8,
            "ytick.major.size": 3.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "lines.linewidth": 2.2,
            "lines.markersize": 5.0,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        }
    )


def _require_cols(df: pd.DataFrame, cols: list[str], csv_path: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"CSV missing columns {missing}. Please re-generate `{csv_path}` with the updated "
            f"`scripts/simple_repair_analysis.py` (it now writes `std_mse_*` and `w2_sq_*`)."
        )


def plot_repair_gain_w2(csv_path: str, output_path: str, repair_layers: list[int]):
    df = pd.read_csv(csv_path)
    _require_cols(df, ["layer", "w2_sq_off", "w2_sq_on"], csv_path)

    # Keep the same filtering policy as the original figure.
    df = df[(df["layer"] != 31)]
    # Focus on layers with non-trivial changes (avoid pure zeros).
    df = df[(df["w2_sq_off"] > 1e-12) & (df["w2_sq_on"] > 1e-12)]
    if len(df) == 0:
        print("Warning: no valid layers found for W2^2 gain.")
        return

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    layers = df["layer"].values
    off = df["w2_sq_off"].values
    on = df["w2_sq_on"].values

    denom = np.maximum(off, 1e-12)
    rel_gain = (off - on) / denom * 100.0

    ax.plot(
        layers,
        rel_gain,
        marker="o",
        color="#ff7f0e",
        linewidth=2.2,
        markersize=4.6,
        markerfacecolor="#ff7f0e",
        markeredgewidth=0,
    )

    ax.axhline(0, color="0.2", linewidth=0.85, alpha=0.6)

    present_layers = set(int(x) for x in layers.tolist())
    for layer in repair_layers:
        if int(layer) in present_layers:
            ax.axvline(int(layer), color="tab:green", linestyle="--", alpha=0.45, linewidth=0.85, zorder=0)

    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Relative Gain (%)")

    ax.set_xlim(int(min(layers)) - 0.5, int(max(layers)) + 0.5)
    if len(layers) > 7:
        ax.set_xticks(list(range(int(min(layers)), int(max(layers)) + 1, 2)))
    else:
        ax.set_xticks(layers)

    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, alpha=0.22)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    import os

    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv_path",
        type=str,
        default="outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv",
        help="Path to repair_analysis.csv that contains w2_sq_* columns.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations/paper_figures_w2",
        help="Output directory for paper figures.",
    )
    p.add_argument(
        "--repair_layers",
        type=str,
        default="22,29",
        help="Comma-separated repair layer indices (e.g., 22,29).",
    )
    args = p.parse_args()

    csv_path = args.csv_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    repair_layers = [int(x.strip()) for x in args.repair_layers.split(",") if x.strip()]

    plot_repair_gain_w2(
        csv_path,
        f"{output_dir}/fig_repair_gain.png",
        repair_layers,
    )

    print("\nDone. W2^2 repair gain generated.")
