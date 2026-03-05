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
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 13,  # 增大图例字体
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 3.5,  # 统一线条粗细
        "lines.markersize": 8.0,  # 统一标记点大小
        "axes.grid": True,
        "grid.alpha": 0.3,
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
    fig, ax = plt.subplots(figsize=(7, 4.5))

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
            marker="s", color="#d62728", linewidth=3.0,
            markersize=7, label="Baseline (no repair)",
            markerfacecolor="#d62728", markeredgewidth=0, alpha=0.8)

    ax.plot(layers_ours_valid, total_ours_valid,
            marker="o", color="#2ca02c", linewidth=3.0,
            markersize=7, label="Ours (with repair)",
            markerfacecolor="#2ca02c", markeredgewidth=0, alpha=0.8)

    # 设置对数尺度
    ax.set_yscale("log")

    # 设置标签
    ax.set_xlabel("Decoder Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distribution Alignment Loss", fontsize=12, fontweight="bold")
    # 不需要标题

    # 设置 x 轴范围和刻度
    all_layers = sorted(set(layers_ours_valid) | set(layers_baseline_valid))
    ax.set_xlim(min(all_layers) - 1, max(all_layers) + 1)
    ax.set_xticks(range(0, max(all_layers) + 1, 5))

    # 网格
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)

    # 图例
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

    # 保存
    plt.tight_layout()
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
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
