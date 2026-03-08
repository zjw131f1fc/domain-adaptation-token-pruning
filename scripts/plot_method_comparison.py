#!/usr/bin/env python
"""对比图：展示我们的方法相比一般方法的修复作用"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def set_paper_style():
    """设置论文风格"""
    plt.rcParams.update({
        # 目标：顶会常见风格（干净、紧凑、半页/单栏放大不显得臃肿）
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
    })


def plot_method_comparison(csv_ours: str, csv_baseline: str, output_path: str):
    """对比我们的方法和一般方法"""
    # 读取数据
    df_ours = pd.read_csv(csv_ours)
    df_baseline = pd.read_csv(csv_baseline)

    # 排除 Layer 31 和 Layer 12
    df_ours = df_ours[(df_ours["layer"] != 31) & (df_ours["layer"] != 12)]
    df_baseline = df_baseline[(df_baseline["layer"] != 31) & (df_baseline["layer"] != 12)]

    # 设置样式
    set_paper_style()

    # 创建图表
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    layers_ours = df_ours["layer"].values
    layers_baseline = df_baseline["layer"].values

    # 我们的方法：使用 ON（有 repair）
    total_ours = df_ours["total_on"].values

    # 一般方法：使用 OFF（no repair）
    total_baseline = df_baseline["total_off"].values

    # 过滤 0 值（对数尺度）
    valid_mask_ours = total_ours > 1e-10
    valid_mask_baseline = total_baseline > 1e-10

    layers_ours_valid = layers_ours[valid_mask_ours]
    total_ours_valid = total_ours[valid_mask_ours]

    layers_baseline_valid = layers_baseline[valid_mask_baseline]
    total_baseline_valid = total_baseline[valid_mask_baseline]

    # 绘制两条曲线
    ax.plot(layers_baseline_valid, total_baseline_valid,
            marker="s", color="#d62728", linewidth=2.2,
            markersize=4.6, label="Baseline (w/o repair)",
            markerfacecolor="#d62728", markeredgewidth=0, alpha=0.8)

    ax.plot(layers_ours_valid, total_ours_valid,
            marker="o", color="#2ca02c", linewidth=2.2,
            markersize=4.6, label="Ours (w/ repair)",
            markerfacecolor="#2ca02c", markeredgewidth=0, alpha=0.8)

    # 设置对数尺度
    ax.set_yscale("log")

    # 设置标签
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Distribution Alignment Loss")
    # 不需要标题

    # 设置 x 轴范围和刻度
    all_layers = sorted(set(layers_ours_valid) | set(layers_baseline_valid))
    ax.set_xlim(min(all_layers) - 1, max(all_layers) + 1)
    ax.set_xticks(range(0, max(all_layers) + 1, 5))

    # 网格
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, alpha=0.22)

    # 图例：放到坐标轴上方，避免覆盖数据
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

    # 保存
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)

    print(f"已保存对比图: {output_path}")
    print(f"已保存 PDF: {output_path.replace('.png', '.pdf')}")


if __name__ == "__main__":
    import os

    # 我们的方法（full，跳过第一个 adapter）
    csv_ours = "outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv"

    # 一般方法（no adapter no repair）
    csv_baseline = "outputs/visualizations/simple_repair_20260304-1707_vqa-vqav2_llava157b_b7cd_skip1st/repair_analysis.csv"

    output_dir = "outputs/visualizations/paper_figures"
    os.makedirs(output_dir, exist_ok=True)

    plot_method_comparison(
        csv_ours,
        csv_baseline,
        f"{output_dir}/fig_method_comparison.png"
    )

    print("\n完成！对比图已生成。")
