#!/usr/bin/env python
"""Method comparison figure using diagonal-Gaussian 2-Wasserstein metric (W2^2)."""

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


def plot_method_comparison_w2(csv_ours: str, csv_baseline: str, output_path: str):
    df_ours = pd.read_csv(csv_ours)
    df_baseline = pd.read_csv(csv_baseline)
    _require_cols(df_ours, ["layer", "w2_sq_on"], csv_ours)
    _require_cols(df_baseline, ["layer", "w2_sq_off"], csv_baseline)

    # Keep consistent with original filtering.
    df_ours = df_ours[(df_ours["layer"] != 31) & (df_ours["layer"] != 12)]
    df_baseline = df_baseline[(df_baseline["layer"] != 31) & (df_baseline["layer"] != 12)]

    set_paper_style()
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    x_ours = df_ours["layer"].values
    y_ours = df_ours["w2_sq_on"].values

    x_base = df_baseline["layer"].values
    y_base = df_baseline["w2_sq_off"].values

    valid_ours = y_ours > 1e-12
    valid_base = y_base > 1e-12

    x_ours = x_ours[valid_ours]
    y_ours = y_ours[valid_ours]
    x_base = x_base[valid_base]
    y_base = y_base[valid_base]

    ax.plot(
        x_base,
        y_base,
        marker="s",
        color="#d62728",
        linewidth=2.2,
        markersize=4.6,
        label="Baseline (w/o repair)",
        markerfacecolor="#d62728",
        markeredgewidth=0,
        alpha=0.85,
    )
    ax.plot(
        x_ours,
        y_ours,
        marker="o",
        color="#2ca02c",
        linewidth=2.2,
        markersize=4.6,
        label="Ours (w/ repair)",
        markerfacecolor="#2ca02c",
        markeredgewidth=0,
        alpha=0.85,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Representation Drift (W2$^2$)")

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

    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv_ours",
        type=str,
        default="outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv",
        help="Our repair_analysis.csv (expects w2_sq_on column).",
    )
    p.add_argument(
        "--csv_baseline",
        type=str,
        default="outputs/visualizations/simple_repair_20260304-1707_vqa-vqav2_llava157b_b7cd_skip1st/repair_analysis.csv",
        help="Baseline repair_analysis.csv (expects w2_sq_off column).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/visualizations/paper_figures_w2",
        help="Output directory for paper figures.",
    )
    args = p.parse_args()

    csv_ours = args.csv_ours
    csv_baseline = args.csv_baseline
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    plot_method_comparison_w2(
        csv_ours,
        csv_baseline,
        f"{output_dir}/fig_method_comparison.png",
    )

    print("\nDone. W2^2 method comparison generated.")
