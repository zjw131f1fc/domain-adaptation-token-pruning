#!/usr/bin/env python3
"""
GPU显存占用脚本

用于在共享服务器上占住GPU显存，防止被其他程序抢占。
在另一个终端运行训练脚本时，此脚本会持续占用指定的显存。

用法:
    # 占用所有可见GPU的90%显存
    python hold_gpu.py

    # 占用指定GPU的80%显存
    CUDA_VISIBLE_DEVICES=0,1 python hold_gpu.py --ratio 0.8

    # 占用指定大小的显存 (单位: GB)
    python hold_gpu.py --size 20

按 Ctrl+C 释放显存并退出。
"""

import argparse
import signal
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="GPU显存占用脚本")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.9,
        help="占用显存的比例 (0-1)，默认0.9"
    )
    parser.add_argument(
        "--size",
        type=float,
        default=None,
        help="占用显存的大小 (GB)，如果指定则忽略ratio"
    )
    parser.add_argument(
        "--refresh",
        type=int,
        default=30,
        help="状态刷新间隔 (秒)，默认30"
    )
    return parser.parse_args()


def hold_gpu_memory(ratio=0.9, size_gb=None):
    """
    占用GPU显存

    参数:
        ratio: 占用比例 (0-1)
        size_gb: 占用大小 (GB)，如果指定则忽略ratio

    返回:
        reserved_tensors: 占用显存的tensor列表
    """
    reserved_tensors = []
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("未检测到可用GPU")
        return reserved_tensors

    print(f"检测到 {num_gpus} 个GPU")
    print("-" * 50)

    for device_id in range(num_gpus):
        try:
            props = torch.cuda.get_device_properties(device_id)
            total_memory = props.total_memory
            gpu_name = props.name

            # 计算要占用的大小
            if size_gb is not None:
                reserve_size = int(size_gb * 1024**3)
                reserve_size = min(reserve_size, int(total_memory * 0.95))  # 最多95%
            else:
                reserve_size = int(total_memory * ratio)

            # 分配tensor (int8 = 1 byte per element)
            num_elements = reserve_size
            dummy_tensor = torch.empty(num_elements, dtype=torch.int8, device=f'cuda:{device_id}')
            reserved_tensors.append(dummy_tensor)

            actual_gb = reserve_size / 1024**3
            total_gb = total_memory / 1024**3
            print(f"GPU {device_id} ({gpu_name}): 已占用 {actual_gb:.2f} GB / {total_gb:.2f} GB ({actual_gb/total_gb*100:.1f}%)")

        except torch.cuda.OutOfMemoryError:
            print(f"GPU {device_id}: 显存不足，可能已被其他程序占用")
        except Exception as e:
            print(f"GPU {device_id}: 占用失败 - {e}")

    return reserved_tensors


def print_status(reserved_tensors):
    """打印当前GPU状态"""
    print("\n" + "=" * 50)
    print(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)

    for device_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(device_id) / 1024**3
        reserved = torch.cuda.memory_reserved(device_id) / 1024**3
        total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        print(f"GPU {device_id}: 已分配 {allocated:.2f} GB, 已预留 {reserved:.2f} GB, 总计 {total:.2f} GB")

    print("=" * 50)
    print("按 Ctrl+C 释放显存并退出")


def main():
    args = parse_args()

    print("=" * 50)
    print("GPU显存占用脚本")
    print("=" * 50)

    # 占用显存
    reserved_tensors = hold_gpu_memory(ratio=args.ratio, size_gb=args.size)

    if not reserved_tensors:
        print("没有成功占用任何GPU显存")
        sys.exit(1)

    print("-" * 50)
    print("显存已占用，保持运行中...")
    print("按 Ctrl+C 释放显存并退出")
    print("-" * 50)

    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n\n正在释放显存...")
        reserved_tensors.clear()
        torch.cuda.empty_cache()
        print("显存已释放，退出")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 保持运行，定期打印状态
    try:
        while True:
            time.sleep(args.refresh)
            print_status(reserved_tensors)
    except KeyboardInterrupt:
        signal_handler(None, None)


if __name__ == "__main__":
    main()
