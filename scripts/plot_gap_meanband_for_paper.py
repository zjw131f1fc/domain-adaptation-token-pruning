#!/usr/bin/env python
"""Paper figures (mean + variance band).

Same figures as `outputs/visualizations/paper_figures/`, but we visualize:
  - line: mean alignment MSE
  - band: variance-mismatch term (i.e., total - mean)

This makes the mean/variance contribution explicit without adding extra subplots.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def set_paper_style():
    """ECCV-ish clean style (single-column / half-page)."""
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


def _safe_band(mean_y: np.ndarray, total_y: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower, upper) for a non-negative band on log-scale axes."""
    mean_y = np.asarray(mean_y, dtype=np.float64)
    total_y = np.asarray(total_y, dtype=np.float64)
    lower = np.maximum(mean_y, eps)
    upper = np.maximum(total_y, lower)
    return lower, upper


def plot_gap_introduction_meanband(csv_path: str, output_path: str, pruning_layers: list[int]):
    """Introduction gap figure: show layers between the 2nd and 3rd pruner."""
    df = pd.read_csv(csv_path)
    set_paper_style()

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    layer_start = int(pruning_layers[1])
    layer_end = int(pruning_layers[2])

    df_filtered = df[(df["layer"] >= layer_start) & (df["layer"] <= layer_end)]

    layers = df_filtered["layer"].values
    mean_off = df_filtered["mean_mse_off"].values
    total_off = df_filtered["total_off"].values

    valid = (total_off > 1e-10) & (mean_off > 1e-12)
    layers = layers[valid]
    mean_off = mean_off[valid]
    total_off = total_off[valid]

    y0, y1 = _safe_band(mean_off, total_off)

    # Band: variance term (total - mean)
    ax.fill_between(layers, y0, y1, color="#d62728", alpha=0.18, linewidth=0.0, zorder=1)

    # Line: mean term
    ax.plot(
        layers,
        y0,
        marker="o",
        color="#d62728",
        linewidth=2.2,
        markersize=4.6,
        markerfacecolor="#d62728",
        markeredgewidth=0,
        zorder=2,
    )

    # Mark pruning layers
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
    ax.set_ylabel("Distribution Alignment Loss")

    ax.set_xlim(layer_start - 1, layer_end + 1)
    ax.set_xticks(range(layer_start, layer_end + 1, 2))
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, alpha=0.22)

    fig.tight_layout(pad=0.6)
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_gap_comparison_meanband(csv_path: str, output_path: str, pruning_layers: list[int]):
    """Comparison figure: w/o repair vs w/ repair with mean + band."""
    df = pd.read_csv(csv_path)
    set_paper_style()

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    layers = df["layer"].values

    mean_off = df["mean_mse_off"].values
    total_off = df["total_off"].values
    mean_on = df["mean_mse_on"].values
    total_on = df["total_on"].values

    # Filter invalid for each curve separately (log-scale safe)
    valid_off = (total_off > 1e-10) & (mean_off > 1e-12)
    valid_on = (total_on > 1e-10) & (mean_on > 1e-12)

    x_off = layers[valid_off]
    y0_off, y1_off = _safe_band(mean_off[valid_off], total_off[valid_off])

    x_on = layers[valid_on]
    y0_on, y1_on = _safe_band(mean_on[valid_on], total_on[valid_on])

    # Bands first
    ax.fill_between(x_off, y0_off, y1_off, color="#d62728", alpha=0.15, linewidth=0.0, zorder=1)
    ax.fill_between(x_on, y0_on, y1_on, color="#2ca02c", alpha=0.15, linewidth=0.0, zorder=1)

    # Mean lines
    ax.plot(
        x_off,
        y0_off,
        marker="o",
        color="#d62728",
        linewidth=2.2,
        markersize=4.6,
        label="w/o repair",
        markerfacecolor="#d62728",
        markeredgewidth=0,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        x_on,
        y0_on,
        marker="s",
        color="#2ca02c",
        linewidth=2.2,
        markersize=4.6,
        label="w/ repair (ours)",
        markerfacecolor="#2ca02c",
        markeredgewidth=0,
        alpha=0.85,
        zorder=2,
    )

    for layer in pruning_layers:
        ax.axvline(int(layer), color="0.55", linestyle="--", alpha=0.55, linewidth=0.85, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Distribution Alignment Loss")

    ax.set_xlim(-1, max(layers) + 1)
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

    csv_path = "outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv"
    output_dir = "outputs/visualizations/paper_figures_meanband"
    os.makedirs(output_dir, exist_ok=True)

    pruning_layers = [4, 14, 24]

    plot_gap_introduction_meanband(
        csv_path,
        f"{output_dir}/fig_introduction_gap.png",
        pruning_layers,
    )

    plot_gap_comparison_meanband(
        csv_path,
        f"{output_dir}/fig_repair_comparison.png",
        pruning_layers,
    )

    print("\nDone. Mean+band figures generated.")
