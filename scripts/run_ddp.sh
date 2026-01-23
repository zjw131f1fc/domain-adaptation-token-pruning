#!/bin/bash
# DDP 分布式训练启动脚本
#
# 使用方法:
#   ./scripts/run_ddp.sh [GPU列表] [配置文件]
#
# 示例:
#   ./scripts/run_ddp.sh 0,1,2,3                                # 使用 GPU 0,1,2,3，默认配置
#   ./scripts/run_ddp.sh 4,5 configs/vision_token_pruning.yaml  # 使用 GPU 4,5，指定配置
#   ./scripts/run_ddp.sh                                        # 使用所有可用 GPU

set -e


# 参数解析
GPU_IDS=${1:-""}  # GPU ID 列表，如 "0,1,2,3"
CONFIG=${2:-"configs/vision_token_pruning.yaml"}

# 计算 GPU 数量并设置 CUDA_VISIBLE_DEVICES
if [ -z "$GPU_IDS" ]; then
    # 未指定时使用所有可用 GPU
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    # 计算逗号分隔的 GPU 数量
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
fi

# 创建日志目录
LOG_DIR="logs/ddp_runs"
mkdir -p "$LOG_DIR"

# 生成时间戳日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "=============================================="
echo "DDP Training Configuration"
echo "=============================================="
echo "GPU IDs: ${GPU_IDS:-all}"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG"
echo "Log file: $LOG_FILE"
echo "=============================================="

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 使用 torchrun 启动分布式训练，输出重定向到日志文件
# --standalone: 单机多卡模式
# --nproc_per_node: 每个节点的进程数（GPU数）
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    main_acp_ddp.py \
    --config "$CONFIG" 2>&1 | tee "$LOG_FILE"

echo "=============================================="
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
echo "=============================================="

