#!/bin/bash
# Vision Token Pruning 训练启动脚本

# ==================== 环境变量 ====================
export HF_HOME="/data/users/zjw/huggingface_cache"
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="0,1"  # 修改为你要使用的 GPU

# ==================== 启动训练 ====================
# 获取 GPU 数量
IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPU_ARRAY[@]}

cd "$(dirname "$0")/.."  # 切换到项目根目录

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "使用 $NUM_GPUS 个 GPU 进行 DDP 训练..."
    torchrun --nproc_per_node=$NUM_GPUS refactor/main.py
else
    echo "使用单 GPU 训练..."
    python refactor/main.py
fi
