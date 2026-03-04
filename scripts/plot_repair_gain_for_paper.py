#!/usr/bin/env python
"""为论文生成 repair gain 图表 - 只展示有效果的层"""

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
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.5,
        "lines.markersize": 6.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    })


def plot_repair_gain(csv_path: str, output_path: str, repair_layers: list):
    """绘制论文用的 repair gain 图 - 只展示有效果的层"""
    # 读取数据
    df = pd.read_csv(csv_path)

    # 只保留有非零 gain 的层（排除 Layer 31）
    df_filtered = df[(df["gain"].abs() > 1e-10) & (df["layer"] != 31)]

    if len(df_filtered) == 0:
        print("警告: 没有非零 gain 的层")
        return

    # 设置样式
    set_paper_style()

    # 创建图表
    fig, ax = plt.subplots(figsize=(7, 4.5))

    layers = df_filtered["layer"].values
    total_off = df_filtered["total_off"].values
    gains = df_filtered["gain"].values

    # 计算相对 gain（百分比）
    relative_gains = (gains / total_off) * 100

    # 绘制曲线
    ax.plot(layers, relative_gains,
            marker="o", color="#ff7f0e", linewidth=2.5,
            markersize=6, markerfacecolor="#ff7f0e", markeredgewidth=0)

    # 添加 0 线
    ax.axhline(0, color="k", linewidth=1.0, alpha=0.4)

    # 标记 repair 层
    for layer in repair_layers:
        if layer in layers:
            ax.axvline(layer, color="tab:green", linestyle="--",
                      alpha=0.4, linewidth=1.5, zorder=0)

    # 设置标签
    ax.set_xlabel("Decoder Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Gain (%)", fontsize=12, fontweight="bold")
    # 不需要标题

    # 设置 x 轴范围和刻度
    ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)
    ax.set_xticks(layers)

    # 网格
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)

    # 保存
    plt.tight_layout()
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)

    print(f"已保存论文图表: {output_path}")
    print(f"已保存 PDF: {output_path.replace('.png', '.pdf')}")


if __name__ == "__main__":
    import os

    # 使用跳过第一个 adapter 的数据
    csv_path = "outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv"
    output_dir = "outputs/visualizations/paper_figures"

    os.makedirs(output_dir, exist_ok=True)

    repair_layers = [22, 29]  # 实际的 repair 层（跳过第一个后）

    plot_repair_gain(
        csv_path,
        f"{output_dir}/fig_repair_gain.png",
        repair_layers
    )

    print("\n完成！论文图表已生成。")
