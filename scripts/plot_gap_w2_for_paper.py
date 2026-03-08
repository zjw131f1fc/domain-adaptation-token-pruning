#!/usr/bin/env python
"""Paper figures using diagonal-Gaussian 2-Wasserstein metric (W2^2).

This version avoids the training-time `var_weight` hyper-parameter by using the
diagonal-Gaussian closed form:
  W2^2 = ||mu_s - mu_t||^2 + ||sigma_s - sigma_t||^2

In our CSV this is stored as (per-dimension average):
  w2_sq = mean_mse + std_mse
"""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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


def plot_gap_introduction_w2(csv_path: str, output_path: str, pruning_layers: list[int]):
    df = pd.read_csv(csv_path)
    _require_cols(df, ["layer", "w2_sq_off"], csv_path)
    set_paper_style()

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    layer_start = int(pruning_layers[1])
    layer_end = int(pruning_layers[2])

    df = df[(df["layer"] >= layer_start) & (df["layer"] <= layer_end)]
    layers = df["layer"].values
    y = df["w2_sq_off"].values

    valid = y > 1e-12
    layers = layers[valid]
    y = y[valid]

    ax.plot(
        layers,
        y,
        marker="o",
        color="#d62728",
        linewidth=2.2,
        markersize=4.6,
        markerfacecolor="#d62728",
        markeredgewidth=0,
    )

    ax.axvline(layer_start, color="0.55", linestyle="--", alpha=0.7, linewidth=0.9, zorder=0)
    ax.axvline(layer_end, color="0.55", linestyle="--", alpha=0.7, linewidth=0.9, zorder=0)
    ax.text(
        layer_start + 0.15,
        0.98,
        f"L{layer_start}",
        transform=ax.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=8.6,
        color="0.25",
    )
    ax.text(
        layer_end - 0.15,
        0.98,
        f"L{layer_end}",
        transform=ax.get_xaxis_transform(),
        ha="right",
        va="top",
        fontsize=8.6,
        color="0.25",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Representation Drift (W2$^2$)")

    ax.set_xlim(layer_start - 1, layer_end + 1)
    ax.set_xticks(range(layer_start, layer_end + 1, 2))
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, alpha=0.22)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_gap_comparison_w2(csv_path: str, output_path: str, pruning_layers: list[int]):
    df = pd.read_csv(csv_path)
    _require_cols(df, ["layer", "w2_sq_off", "w2_sq_on"], csv_path)
    set_paper_style()

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    layers = df["layer"].values
    off = df["w2_sq_off"].values
    on = df["w2_sq_on"].values

    valid = (off > 1e-12) & (on > 1e-12)
    layers = layers[valid]
    off = off[valid]
    on = on[valid]

    ax.plot(
        layers,
        off,
        marker="o",
        color="#d62728",
        linewidth=2.2,
        markersize=4.6,
        label="w/o repair",
        markerfacecolor="#d62728",
        markeredgewidth=0,
        alpha=0.85,
    )
    ax.plot(
        layers,
        on,
        marker="s",
        color="#2ca02c",
        linewidth=2.2,
        markersize=4.6,
        label="w/ repair (ours)",
        markerfacecolor="#2ca02c",
        markeredgewidth=0,
        alpha=0.85,
    )

    for layer in pruning_layers:
        ax.axvline(int(layer), color="0.55", linestyle="--", alpha=0.55, linewidth=0.85, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Representation Drift (W2$^2$)")

    ax.set_xlim(-1, int(max(layers)) + 1)
    ax.set_xticks(range(0, int(max(layers)) + 1, 5))
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, alpha=0.22)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=False,
        handlelength=1.5,
        labelspacing=0.3,
        columnspacing=0.8,
        borderaxespad=0.0,
    )

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
        "--pruning_layers",
        type=str,
        default="4,14,24",
        help="Comma-separated pruning layer indices (e.g., 4,14,24).",
    )
    args = p.parse_args()

    csv_path = args.csv_path
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    pruning_layers = [int(x.strip()) for x in args.pruning_layers.split(",") if x.strip()]

    plot_gap_introduction_w2(
        csv_path,
        f"{output_dir}/fig_introduction_gap.png",
        pruning_layers,
    )

    plot_gap_comparison_w2(
        csv_path,
        f"{output_dir}/fig_repair_comparison.png",
        pruning_layers,
    )

    print("\nDone. W2^2 figures generated.")
