"""
GPU 锁定/解锁工具类
用于通过占用显存来锁定 GPU，防止其他进程使用
"""

import torch
from typing import Optional


class GPULock:
    """
    GPU 锁定工具类

    通过创建大型 tensor 占用显存来锁定 GPU，
    解锁时释放 tensor 并清理显存。

    注意：会锁定所有可见的 GPU（由 CUDA_VISIBLE_DEVICES 控制）

    Usage:
        lock = GPULock()
        lock.lock()      # 锁定所有可见 GPU
        # ... 其他进程无法使用被锁定的 GPU
        lock.unlock()    # 释放显存
    """

    def __init__(self, reserve_mb: float = 512):
        """
        Args:
            reserve_mb: 保留的显存大小(MB)，避免完全占满导致系统不稳定
        """
        self.reserve_mb = reserve_mb
        self._lock_tensors: dict[int, torch.Tensor] = {}

    def lock(self) -> dict[int, float]:
        """
        锁定所有可见的 GPU

        Returns:
            dict: {device_id: 占用的显存大小(MB)}
        """
        if not torch.cuda.is_available():
            print("CUDA 不可用，无法锁定 GPU")
            return {}

        result = {}
        device_count = torch.cuda.device_count()

        for device_id in range(device_count):
            if device_id in self._lock_tensors:
                print(f"GPU {device_id} 已被锁定，跳过")
                continue

            try:
                occupied_mb = self._lock_device(device_id)
                result[device_id] = occupied_mb
                print(f"GPU {device_id} 已锁定，占用 {occupied_mb:.1f} MB")
            except Exception as e:
                print(f"锁定 GPU {device_id} 失败: {e}")

        return result

    def _lock_device(self, device_id: int) -> float:
        """锁定单个 GPU"""
        torch.cuda.set_device(device_id)
        torch.cuda.empty_cache()

        # 获取可用显存
        free_memory, _ = torch.cuda.mem_get_info(device_id)
        free_mb = free_memory / (1024 ** 2)

        # 计算要占用的显存大小
        target_mb = free_mb - self.reserve_mb
        if target_mb <= 0:
            raise ValueError(f"可用显存不足: {free_mb:.1f} MB")

        # 创建 tensor 占用显存 (使用 float32，每个元素 4 bytes)
        num_elements = int(target_mb * 1024 * 1024 / 4)
        self._lock_tensors[device_id] = torch.empty(
            num_elements,
            dtype=torch.float32,
            device=f'cuda:{device_id}'
        )

        return target_mb

    def unlock(self):
        """解锁所有已锁定的 GPU"""
        for device_id in list(self._lock_tensors.keys()):
            del self._lock_tensors[device_id]
            torch.cuda.set_device(device_id)
            torch.cuda.empty_cache()
            print(f"GPU {device_id} 已解锁")

    def is_locked(self) -> bool:
        """检查是否有 GPU 被锁定"""
        return len(self._lock_tensors) > 0

    def __del__(self):
        """析构时自动解锁"""
        self.unlock()
