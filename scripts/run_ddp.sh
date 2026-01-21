#!/bin/bash
# DDP 分布式训练启动脚本
#
# 使用方法:
#   ./scripts/run_ddp.sh [GPU数量] [配置文件]
#
# 示例:
#   ./scripts/run_ddp.sh 4                                      # 使用 4 GPU，默认配置
#   ./scripts/run_ddp.sh 2 configs/vision_token_pruning.yaml    # 使用 2 GPU，指定配置
#   ./scripts/run_ddp.sh                                        # 使用所有可用 GPU

set -e


# 参数解析
NUM_GPUS=${1:-$(nvidia-smi -L | wc -l)}  # 默认使用所有可用 GPU
CONFIG=${2:-"configs/vision_token_pruning.yaml"}

echo "=============================================="
echo "DDP Training Configuration"
echo "=============================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG"
echo "=============================================="

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 使用 torchrun 启动分布式训练
# --standalone: 单机多卡模式
# --nproc_per_node: 每个节点的进程数（GPU数）
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    main_acp_ddp.py \
    --config "$CONFIG"

echo "=============================================="
echo "Training completed!"
echo "=============================================="
