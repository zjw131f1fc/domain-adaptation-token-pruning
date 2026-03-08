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
        # 不在 rcParams 里全局开 grid；每张图按需开 y-grid（更干净）
        "axes.grid": False,
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

    # 创建图表：按“单栏/半页宽度”设计，避免缩放带来的字体过小
    fig, ax = plt.subplots(figsize=(3.45, 2.35))

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
    ax.plot(
            layers_valid, total_off_valid,
            marker="o", color="#d62728", linewidth=2.2,
            markersize=4.6,
            # Single curve: omit legend for a cleaner paper figure (use caption to describe).
            markerfacecolor="#d62728", markeredgewidth=0)

    # 标记剪枝层（只标记起点和终点）
    ax.axvline(layer_start, color="0.55", linestyle="--", alpha=0.7, linewidth=0.9, zorder=0)
    ax.axvline(layer_end, color="0.55", linestyle="--", alpha=0.7, linewidth=0.9, zorder=0)

    # 添加剪枝层标签：放在坐标轴上方（避免遮挡曲线/legend）
    # 只标注 L14/L24（短文本避免挤出图边界）
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

    # 设置对数尺度
    ax.set_yscale("log")

    # 设置标签
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Distribution Alignment Loss")
    # 不需要标题
    # ax.set_title("Vision Token Pruning Causes Representation Drift",
    #             fontsize=13, fontweight="bold", pad=15)

    # 设置 x 轴范围和刻度
    ax.set_xlim(layer_start - 1, layer_end + 1)
    ax.set_xticks(range(layer_start, layer_end + 1, 2))

    # 网格
    # ECCV 风格：只开 y 方向的轻量网格（读 log 值更方便，且不显脏）
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.6, alpha=0.22)

    # No legend for the introduction gap figure (single series).

    # 保存
    fig.tight_layout(pad=0.6)
    fig.savefig(output_path.replace(".png", ".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.08, dpi=300)
    plt.close(fig)

    print(f"已保存论文图表: {output_path}")
    print(f"已保存 PDF: {output_path.replace('.png', '.pdf')}")


def plot_gap_comparison(csv_path: str, output_path: str, pruning_layers: list):
    """绘制对比图（可选）- 展示 repair 的效果"""
    df = pd.read_csv(csv_path)
    set_paper_style()

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

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
            marker="o", color="#d62728", linewidth=2.2,
            markersize=4.6, label="w/o repair",
            markerfacecolor="#d62728", markeredgewidth=0, alpha=0.8)

    ax.plot(layers_valid, total_on_valid,
            marker="s", color="#2ca02c", linewidth=2.2,
            markersize=4.6, label="w/ repair (ours)",
            markerfacecolor="#2ca02c", markeredgewidth=0, alpha=0.8)

    # 标记剪枝层
    for layer in pruning_layers:
        ax.axvline(layer, color="0.55", linestyle="--", alpha=0.55, linewidth=0.85, zorder=0)

    ax.set_yscale("log")
    ax.set_xlabel("Decoder Layer")
    ax.set_ylabel("Distribution Alignment Loss")

    ax.set_xlim(-1, max(layers) + 1)
    ax.set_xticks(range(0, max(layers) + 1, 5))
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
