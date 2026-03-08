#!/usr/bin/env python
"""Method comparison (mean + variance band).

Same as `scripts/plot_method_comparison.py`, but visualize:
  - line: mean alignment MSE
  - band: variance-mismatch term (total - mean)
"""

import sys
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


def _safe_band(mean_y: np.ndarray, total_y: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    mean_y = np.asarray(mean_y, dtype=np.float64)
    total_y = np.asarray(total_y, dtype=np.float64)
    lower = np.maximum(mean_y, eps)
    upper = np.maximum(total_y, lower)
    return lower, upper


def plot_method_comparison_meanband(csv_ours: str, csv_baseline: str, output_path: str):
    df_ours = pd.read_csv(csv_ours)
    df_baseline = pd.read_csv(csv_baseline)

    # Keep consistent with the original figure.
    df_ours = df_ours[(df_ours["layer"] != 31) & (df_ours["layer"] != 12)]
    df_baseline = df_baseline[(df_baseline["layer"] != 31) & (df_baseline["layer"] != 12)]

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    # Ours: ON (repair)
    x_ours = df_ours["layer"].values
    mean_ours = df_ours["mean_mse_on"].values
    total_ours = df_ours["total_on"].values

    # Baseline: OFF (no repair)
    x_base = df_baseline["layer"].values
    mean_base = df_baseline["mean_mse_off"].values
    total_base = df_baseline["total_off"].values

    valid_ours = (total_ours > 1e-10) & (mean_ours > 1e-12)
    valid_base = (total_base > 1e-10) & (mean_base > 1e-12)

    x_ours = x_ours[valid_ours]
    y0_ours, y1_ours = _safe_band(mean_ours[valid_ours], total_ours[valid_ours])

    x_base = x_base[valid_base]
    y0_base, y1_base = _safe_band(mean_base[valid_base], total_base[valid_base])

    # Bands
    ax.fill_between(x_base, y0_base, y1_base, color="#d62728", alpha=0.15, linewidth=0.0, zorder=1)
    ax.fill_between(x_ours, y0_ours, y1_ours, color="#2ca02c", alpha=0.15, linewidth=0.0, zorder=1)

    # Mean lines
    ax.plot(
        x_base,
        y0_base,
        marker="s",
        color="#d62728",
        linewidth=2.2,
        markersize=4.6,
        label="Baseline (w/o repair)",
        markerfacecolor="#d62728",
        markeredgewidth=0,
        alpha=0.85,
        zorder=2,
    )
    ax.plot(
        x_ours,
        y0_ours,
        marker="o",
        color="#2ca02c",
        linewidth=2.2,
        markersize=4.6,
        label="Ours (w/ repair)",
        markerfacecolor="#2ca02c",
        markeredgewidth=0,
        alpha=0.85,
        zorder=2,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Distribution Alignment Loss")

    all_layers = sorted(set(x_ours.tolist()) | set(x_base.tolist()))
    if all_layers:
        ax.set_xlim(min(all_layers) - 1, max(all_layers) + 1)
        ax.set_xticks(range(0, int(max(all_layers)) + 1, 5))
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

    csv_ours = "outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv"
    csv_baseline = "outputs/visualizations/simple_repair_20260304-1707_vqa-vqav2_llava157b_b7cd_skip1st/repair_analysis.csv"

    output_dir = "outputs/visualizations/paper_figures_meanband"
    os.makedirs(output_dir, exist_ok=True)

    plot_method_comparison_meanband(
        csv_ours,
        csv_baseline,
        f"{output_dir}/fig_method_comparison.png",
    )

    print("\nDone. Mean+band method comparison generated.")
