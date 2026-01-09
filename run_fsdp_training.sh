#!/bin/bash

# Vision Token Pruning - FSDP 多卡训练启动脚本
#
# 使用方法:
#   1. 单卡测试:
#      bash run_fsdp_training.sh --gpus 1
#
#   2. 多卡训练 (推荐):
#      bash run_fsdp_training.sh --gpus 3
#
#   3. 指定特定GPU:
#      CUDA_VISIBLE_DEVICES=2,5,7 bash run_fsdp_training.sh --gpus 3

set -e

# 默认参数
NUM_GPUS=3
MASTER_PORT=29500

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Vision Token Pruning - FSDP 多卡训练"
echo "=========================================="
echo "GPU 数量: $NUM_GPUS"
echo "Master Port: $MASTER_PORT"
echo "=========================================="

# 激活 conda 环境
echo "激活 conda 环境: rl-pruning"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate rl-pruning

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_LAUNCH_BLOCKING=0  # 并行启动CUDA kernels

# 设置 PyTorch 分布式训练环境变量
export OMP_NUM_THREADS=4  # 减少CPU线程数，避免资源竞争

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "单卡训练模式"
    python main_hf_trainer.py
else
    echo "多卡训练模式 (FSDP)"

    # 使用 torchrun 启动分布式训练
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        main_hf_trainer.py
fi

echo "=========================================="
echo "训练完成!"
echo "=========================================="
