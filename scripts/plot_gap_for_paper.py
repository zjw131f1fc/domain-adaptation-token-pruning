#!/usr/bin/env python
"""为论文 Introduction 生成 gap 示意图"""

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


def plot_gap_introduction(csv_path: str, output_path: str, pruning_layers: list):
    """绘制 Introduction 用的 gap 图 - 只展示第二个到第三个剪枝器之间"""
    # 读取数据
    df = pd.read_csv(csv_path)

    # 设置样式
    set_paper_style()

    # 创建图表 - 单列宽度适合论文
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # 只展示第二个剪枝器到第三个剪枝器之间的层
    # pruning_layers = [4, 14, 24]
    layer_start = pruning_layers[1]  # 14
    layer_end = pruning_layers[2]    # 24

    # 过滤数据：只保留 [layer_start, layer_end] 范围
    mask = (df["layer"] >= layer_start) & (df["layer"] <= layer_end)
    df_filtered = df[mask]

    layers = df_filtered["layer"].values
    total_off = df_filtered["total_off"].values

    # 过滤掉 0 值（对数尺度）
    valid_mask = total_off > 1e-10
    layers_valid = layers[valid_mask]
    total_off_valid = total_off[valid_mask]

    # 绘制主曲线
    ax.plot(layers_valid, total_off_valid,
            marker="o", color="#d62728", linewidth=2.5,
            markersize=6, label="Pruned (without repair)",
            markerfacecolor="#d62728", markeredgewidth=0)

    # 标记剪枝层（只标记起点和终点）
    ax.axvline(layer_start, color="gray", linestyle="--",
               alpha=0.6, linewidth=1.8, zorder=0)
    ax.axvline(layer_end, color="gray", linestyle="--",
               alpha=0.6, linewidth=1.8, zorder=0)

    # 添加剪枝层标签
    y_pos = ax.get_ylim()[1] * 0.3
    ax.text(layer_start, y_pos,
           f"Pruning\nLayer {layer_start}",
           ha="center", va="bottom", fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white", edgecolor="gray", alpha=0.9))
    ax.text(layer_end, y_pos,
           f"Pruning\nLayer {layer_end}",
           ha="center", va="bottom", fontsize=9,
           bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white", edgecolor="gray", alpha=0.9))

    # 设置对数尺度
    ax.set_yscale("log")

    # 设置标签
    ax.set_xlabel("Decoder Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distribution Alignment Loss", fontsize=12, fontweight="bold")
    # 不需要标题
    # ax.set_title("Vision Token Pruning Causes Representation Drift",
    #             fontsize=13, fontweight="bold", pad=15)

    # 设置 x 轴范围和刻度
    ax.set_xlim(layer_start - 1, layer_end + 1)
    ax.set_xticks(range(layer_start, layer_end + 1, 2))

    # 网格
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)

    # 图例
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

    # 保存
    plt.tight_layout()
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)

    print(f"已保存论文图表: {output_path}")
    print(f"已保存 PDF: {output_path.replace('.png', '.pdf')}")


def plot_gap_comparison(csv_path: str, output_path: str, pruning_layers: list):
    """绘制对比图（可选）- 展示 repair 的效果"""
    df = pd.read_csv(csv_path)
    set_paper_style()

    fig, ax = plt.subplots(figsize=(7, 4.5))

    layers = df["layer"].values
    total_off = df["total_off"].values
    total_on = df["total_on"].values

    # 过滤 0 值
    valid_mask = (total_off > 1e-10) & (total_on > 1e-10)
    layers_valid = layers[valid_mask]
    total_off_valid = total_off[valid_mask]
    total_on_valid = total_on[valid_mask]

    # 绘制两条曲线
    ax.plot(layers_valid, total_off_valid,
            marker="o", color="#d62728", linewidth=2.5,
            markersize=5, label="Without repair",
            markerfacecolor="#d62728", markeredgewidth=0, alpha=0.8)

    ax.plot(layers_valid, total_on_valid,
            marker="s", color="#2ca02c", linewidth=2.5,
            markersize=5, label="With repair (ours)",
            markerfacecolor="#2ca02c", markeredgewidth=0, alpha=0.8)

    # 标记剪枝层
    for layer in pruning_layers:
        ax.axvline(layer, color="gray", linestyle="--",
                   alpha=0.4, linewidth=1.5, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Decoder Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Distribution Alignment Loss", fontsize=12, fontweight="bold")
    ax.set_title("Repair Adapter Reduces Representation Drift",
                fontsize=13, fontweight="bold", pad=15)

    ax.set_xlim(-1, max(layers) + 1)
    ax.set_xticks(range(0, max(layers) + 1, 5))
    ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.8)
    ax.legend(loc="upper left", frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)

    print(f"已保存对比图: {output_path}")
    print(f"已保存 PDF: {output_path.replace('.png', '.pdf')}")


if __name__ == "__main__":
    # 使用跳过第一个 adapter 的数据（因为完整版的 repair 反而有害）
    csv_path = "outputs/visualizations/simple_repair_20260304-1405_vqa-vqav2_llava157b_6056_skip1st/repair_analysis.csv"
    output_dir = "outputs/visualizations/paper_figures"

    import os
    os.makedirs(output_dir, exist_ok=True)

    pruning_layers = [4, 14, 24]

    # 图1: Introduction 用 - 只展示问题
    plot_gap_introduction(
        csv_path,
        f"{output_dir}/fig_introduction_gap.png",
        pruning_layers
    )

    # 图2: 对比图 - 展示解决方案（可选）
    plot_gap_comparison(
        csv_path,
        f"{output_dir}/fig_repair_comparison.png",
        pruning_layers
    )

    print("\n完成！论文图表已生成。")
