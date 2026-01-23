#!/bin/bash
# DDP 分布式评估启动脚本
#
# 使用方法:
#   ./scripts/run_eval_ddp.sh [GPU列表] [其他参数...]
#
# 示例:
#   # 使用配置文件中的 checkpoint
#   ./scripts/run_eval_ddp.sh
#   ./scripts/run_eval_ddp.sh 0,1,2,3
#
#   # 命令行指定 checkpoint（覆盖配置）
#   ./scripts/run_eval_ddp.sh 4,5,6,7 --checkpoint outputs/checkpoints/checkpoint_final.pt
#   ./scripts/run_eval_ddp.sh 0,1 --checkpoint outputs/checkpoints/checkpoint_final.pt --mode origin hard
#   ./scripts/run_eval_ddp.sh 2,3 --max_samples 5000
#
# 网格搜索：在配置文件中设置 evaluation_settings.grid_search.enable: true
#
# 单卡评估（不使用 torchrun）:
#   python eval_acp_ddp.py
#   python eval_acp_ddp.py --checkpoint outputs/checkpoints/checkpoint_final.pt

set -e

# 检查第一个参数是否是 GPU 列表（包含数字和逗号）
if [[ $1 =~ ^[0-9,]+$ ]]; then
    GPU_IDS=$1
    shift
else
    GPU_IDS=""
fi

# 计算 GPU 数量并设置 CUDA_VISIBLE_DEVICES
if [ -z "$GPU_IDS" ]; then
    # 未指定时使用所有可用 GPU
    NUM_GPUS=$(nvidia-smi -L | wc -l)
else
    export CUDA_VISIBLE_DEVICES=$GPU_IDS
    # 计算逗号分隔的 GPU 数量
    NUM_GPUS=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)
fi

# 剩余参数传递给 Python 脚本
EXTRA_ARGS="$@"

# 配置文件（可通过 EXTRA_ARGS 中的 --config 覆盖）
CONFIG="configs/vision_token_pruning.yaml"

# 创建日志目录
LOG_DIR="logs/eval_runs"
mkdir -p "$LOG_DIR"

# 生成时间戳日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"

echo "=============================================="
echo "DDP Evaluation Configuration"
echo "=============================================="
echo "GPU IDs: ${GPU_IDS:-all}"
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG"
echo "Extra args: $EXTRA_ARGS"
echo "Log file: $LOG_FILE"
echo "=============================================="

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# 使用 torchrun 启动分布式评估
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    eval_acp_ddp.py \
    --config "$CONFIG" \
    $EXTRA_ARGS 2>&1 | tee "$LOG_FILE"

echo "=============================================="
echo "Evaluation completed!"
echo "Log saved to: $LOG_FILE"
echo "=============================================="
