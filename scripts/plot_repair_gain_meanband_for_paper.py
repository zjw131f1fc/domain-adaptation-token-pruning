#!/usr/bin/env python
"""Repair gain figure (mean + variance band).

Original `fig_repair_gain` plots total relative gain:
  gain_total = (total_off - total_on) / total_off

Here we decompose it:
  gain_mean  = (mean_off - mean_on) / total_off
  band shows gain_var = gain_total - gain_mean

So the line directly shows the mean-term improvement, while the band thickness
visualizes how much of the gain comes from the variance-mismatch term.
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


def plot_repair_gain_meanband(csv_path: str, output_path: str, repair_layers: list[int]):
    df = pd.read_csv(csv_path)

    # Keep the same filtering policy as the original plot.
    df = df[(df["gain"].abs() > 1e-10) & (df["layer"] != 31)]
    if len(df) == 0:
        print("Warning: no non-zero gain layers found.")
        return

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    layers = df["layer"].values
    total_off = df["total_off"].values
    total_on = df["total_on"].values
    mean_off = df["mean_mse_off"].values
    mean_on = df["mean_mse_on"].values

    # Relative gains in percent
    denom = np.maximum(total_off, 1e-12)
    rel_total = (total_off - total_on) / denom * 100.0
    rel_mean = (mean_off - mean_on) / denom * 100.0

    # Band: contribution from var term (can be positive or negative)
    y0 = rel_mean
    y1 = rel_total

    ax.fill_between(layers, y0, y1, color="#ff7f0e", alpha=0.22, linewidth=0.0, zorder=1)
    ax.plot(
        layers,
        rel_mean,
        marker="o",
        color="#ff7f0e",
        linewidth=2.2,
        markersize=4.6,
        markerfacecolor="#ff7f0e",
        markeredgewidth=0,
        zorder=2,
    )

    ax.axhline(0, color="0.2", linewidth=0.85, alpha=0.6)

    for layer in repair_layers:
        if int(layer) in set(int(x) for x in layers.tolist()):
            ax.axvline(int(layer), color="tab:green", linestyle="--", alpha=0.45, linewidth=0.85, zorder=0)

    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Relative Gain (%)")

    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
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

    csv_path = "outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv"
    output_dir = "outputs/visualizations/paper_figures_meanband"
    os.makedirs(output_dir, exist_ok=True)

    repair_layers = [22, 29]  # skip1st setting

    plot_repair_gain_meanband(
        csv_path,
        f"{output_dir}/fig_repair_gain.png",
        repair_layers,
    )

    print("\nDone. Mean+band repair gain generated.")
