#!/usr/bin/env python3
"""
GPU监控和占用脚本
持续监测GPU 5、6、7，只要有剩余显存就自动占用
"""

import time
import torch
import subprocess
import random
from typing import List, Dict
import signal
import sys

# 要监控的GPU ID
TARGET_GPUS = [5, 6, 7]
# 检查间隔（秒）
CHECK_INTERVAL = 5
# 为每个GPU预留的随机显存空间(MB)，范围1-1000
GPU_RESERVED_MEMORY = {gpu_id: random.randint(1, 1000) for gpu_id in TARGET_GPUS}

# 全局变量存储占用的tensor
occupied_tensors = {}


def get_gpu_memory(gpu_ids: List[int]) -> Dict[int, Dict]:
    """
    获取指定GPU的显存信息

    Returns:
        Dict[gpu_id, {'memory_used_mb': float, 'memory_total_mb': float, 'memory_free_mb': float}]
    """
    try:
        # 使用nvidia-smi获取GPU显存信息
        cmd = [
            'nvidia-smi',
            '--query-gpu=index,memory.used,memory.free,memory.total',
            '--format=csv,noheader,nounits'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        gpu_info = {}
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 4:
                gpu_id = int(parts[0])
                if gpu_id in gpu_ids:
                    gpu_info[gpu_id] = {
                        'memory_used_mb': float(parts[1]),
                        'memory_free_mb': float(parts[2]),
                        'memory_total_mb': float(parts[3])
                    }

        return gpu_info

    except Exception as e:
        print(f"获取GPU信息失败: {e}")
        return {}


def get_available_memory(gpu_id: int, free_mb: float) -> float:
    """
    计算可用于占用的显存大小

    Args:
        gpu_id: GPU ID
        free_mb: 当前空闲显存(MB)

    Returns:
        可占用的显存大小(MB)，如果没有足够空间返回0
    """
    reserved = GPU_RESERVED_MEMORY[gpu_id]
    available = free_mb - reserved
    return max(0, available)


def occupy_gpu(gpu_id: int, memory_mb: float):
    """
    在指定GPU上创建大tensor占用显存

    Args:
        gpu_id: GPU ID
        memory_mb: 要占用的显存大小(MB)
    """
    if memory_mb <= 0:
        return

    try:
        # 每个float32占4字节
        num_elements = int(memory_mb * 1024 * 1024 / 4)

        with torch.cuda.device(gpu_id):
            # 创建大tensor
            tensor = torch.randn(num_elements, dtype=torch.float32, device=f'cuda:{gpu_id}')

            # 如果之前已有tensor，将新的加入列表
            if gpu_id not in occupied_tensors:
                occupied_tensors[gpu_id] = []
            occupied_tensors[gpu_id].append(tensor)

        reserved = GPU_RESERVED_MEMORY[gpu_id]
        print(f"✓ 成功占用 GPU {gpu_id}，占用显存: {memory_mb:.0f} MB (预留: {reserved} MB)")

    except Exception as e:
        print(f"✗ 占用 GPU {gpu_id} 失败: {e}")


def release_gpu(gpu_id: int):
    """释放指定GPU的占用"""
    if gpu_id in occupied_tensors:
        del occupied_tensors[gpu_id]
        torch.cuda.empty_cache()
        print(f"释放 GPU {gpu_id}")


def signal_handler(sig, frame):
    """处理中断信号，清理资源"""
    print("\n\n收到中断信号，释放所有GPU...")
    for gpu_id in list(occupied_tensors.keys()):
        release_gpu(gpu_id)
    print("清理完成，退出程序")
    sys.exit(0)


def main():
    """主循环"""
    print("=" * 70)
    print("GPU 监控和占用脚本")
    print(f"目标GPU: {TARGET_GPUS}")
    print(f"检查间隔: {CHECK_INTERVAL}秒")
    print("预留显存空间:")
    for gpu_id in TARGET_GPUS:
        print(f"  GPU {gpu_id}: {GPU_RESERVED_MEMORY[gpu_id]} MB")
    print("按 Ctrl+C 停止监控并释放GPU")
    print("=" * 70)
    print()

    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("错误: CUDA不可用!")
        return

    print(f"检测到 {torch.cuda.device_count()} 张GPU\n")

    iteration = 0
    while True:
        iteration += 1
        print(f"[检查 #{iteration}] {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 获取GPU显存信息
        gpu_info = get_gpu_memory(TARGET_GPUS)

        if not gpu_info:
            print("无法获取GPU信息，等待下次检查...")
            time.sleep(CHECK_INTERVAL)
            continue

        # 检查每个目标GPU
        for gpu_id in TARGET_GPUS:
            if gpu_id not in gpu_info:
                print(f"  GPU {gpu_id}: 无法获取信息")
                continue

            info = gpu_info[gpu_id]
            is_occupied = gpu_id in occupied_tensors
            available_memory = get_available_memory(gpu_id, info['memory_free_mb'])

            occupy_status = "[已占用]" if is_occupied else "[未占用]"
            reserved = GPU_RESERVED_MEMORY[gpu_id]

            print(f"  GPU {gpu_id}: "
                  f"已用: {info['memory_used_mb']:.0f} MB | "
                  f"空闲: {info['memory_free_mb']:.0f} MB | "
                  f"总计: {info['memory_total_mb']:.0f} MB | "
                  f"可占用: {available_memory:.0f} MB | "
                  f"{occupy_status}")

            # 决策：如果有可占用的显存空间
            if available_memory > 100:  # 至少有100MB才值得占用
                if not is_occupied:
                    print(f"    → GPU {gpu_id} 有可用空间，开始占用...")
                    occupy_gpu(gpu_id, available_memory)
                else:
                    # 已占用但还有空间，追加占用
                    print(f"    → GPU {gpu_id} 还有剩余空间，追加占用...")
                    occupy_gpu(gpu_id, available_memory)
            elif available_memory < 0 and is_occupied:
                # 显存不足，说明有其他程序在使用，释放我们的占用
                print(f"    → GPU {gpu_id} 显存被其他程序使用，释放占用...")
                release_gpu(gpu_id)

        print()
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
