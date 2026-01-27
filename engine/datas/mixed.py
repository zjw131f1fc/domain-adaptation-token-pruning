"""多数据集混合训练支持

提供多数据集的混合与均衡采样功能：
  - MixedDataset: 混合多个数据集，通过索引映射委托访问原始数据集
  - BalancedMultiDatasetSampler: 均衡采样器，优先从采样次数最少的数据集采样

设计要点：
  - MixedDataset 不复制样本，保存原始数据集引用，确保 lazy 加载正常工作
  - 采样器支持 DDP 分布式训练
  - 每个 batch 尽量均匀覆盖各数据集
"""

from typing import Dict, List, Tuple, Any, Iterator, Optional
import random
import math
from torch.utils.data import Sampler
import torch.distributed as dist


class MixedDataset:
    """混合数据集包装器

    通过索引映射委托访问原始数据集，保持 lazy 加载能力。

    Attributes:
        datasets: 原始数据集字典 {dataset_name: dataset}
        dataset_names: 数据集名称列表
        index_map: 全局索引到 (数据集名, 局部索引) 的映射
    """

    def __init__(self, datasets: Dict[str, Any]):
        """
        Args:
            datasets: {dataset_name: dataset_instance} 字典
        """
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())

        # 构建索引映射: global_idx -> (dataset_name, local_idx)
        self.index_map: List[Tuple[str, int]] = []
        self.dataset_ranges: Dict[str, Tuple[int, int]] = {}  # {name: (start, end)}

        current_idx = 0
        for name in self.dataset_names:
            ds = datasets[name]
            ds_len = len(ds)
            start_idx = current_idx
            for local_idx in range(ds_len):
                self.index_map.append((name, local_idx))
            current_idx += ds_len
            self.dataset_ranges[name] = (start_idx, current_idx)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, global_idx: int) -> Dict[str, Any]:
        """通过全局索引访问样本，委托给原始数据集

        Args:
            global_idx: 全局索引

        Returns:
            样本字典，附加 '_dataset_name' 字段标记来源
        """
        dataset_name, local_idx = self.index_map[global_idx]
        # 委托给原始数据集，触发其 lazy 加载逻辑
        sample = self.datasets[dataset_name][local_idx]
        # 浅拷贝后附加数据集来源标记
        sample = sample.copy()
        sample['_dataset_name'] = dataset_name
        return sample

    def get_dataset_indices(self, dataset_name: str) -> List[int]:
        """获取指定数据集的所有全局索引

        Args:
            dataset_name: 数据集名称

        Returns:
            该数据集对应的全局索引列表
        """
        start, end = self.dataset_ranges[dataset_name]
        return list(range(start, end))

    def get_dataset_sizes(self) -> Dict[str, int]:
        """获取各数据集的大小

        Returns:
            {dataset_name: size} 字典
        """
        return {name: len(ds) for name, ds in self.datasets.items()}


class BalancedMultiDatasetSampler(Sampler[int]):
    """均衡多数据集采样器

    采样策略：
    1. 每个 batch 尽量从各数据集均匀采样
    2. 优先从采样次数最少的数据集采样
    3. 支持 DDP 分布式训练

    Attributes:
        mixed_dataset: MixedDataset 实例
        batch_size: 批次大小
        num_replicas: 分布式进程数
        rank: 当前进程编号
        shuffle: 是否打乱
        seed: 随机种子
    """

    def __init__(
        self,
        mixed_dataset: MixedDataset,
        batch_size: int,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ):
        """
        Args:
            mixed_dataset: MixedDataset 实例
            batch_size: 批次大小
            num_replicas: 分布式进程数（None 时自动检测）
            rank: 当前进程编号（None 时自动检测）
            shuffle: 是否打乱各数据集内部顺序
            seed: 随机种子
            drop_last: 是否丢弃最后不完整的批次
        """
        # 分布式设置
        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1
        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        self.mixed_dataset = mixed_dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # 按数据集分组全局索引
        self.dataset_indices: Dict[str, List[int]] = {}
        for name in mixed_dataset.dataset_names:
            self.dataset_indices[name] = mixed_dataset.get_dataset_indices(name)

        # 计算总样本数和每个进程的样本数
        self.total_size = len(mixed_dataset)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)

        if drop_last and self.total_size % self.num_replicas != 0:
            self.num_samples = math.floor(self.total_size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        """设置 epoch，用于分布式训练时确保不同 epoch 有不同的打乱顺序"""
        self.epoch = epoch

    def _generate_balanced_indices(self) -> List[int]:
        """生成均衡采样的索引序列

        策略：
        1. 各数据集内部打乱（如果 shuffle=True）
        2. 轮流从各数据集采样，优先从剩余样本最多的数据集采样
        3. 确保各数据集被均匀消耗

        Returns:
            全局索引列表
        """
        rng = random.Random(self.seed + self.epoch)

        # 复制并打乱各数据集的索引
        dataset_pools: Dict[str, List[int]] = {}
        for name, indices in self.dataset_indices.items():
            pool = indices.copy()
            if self.shuffle:
                rng.shuffle(pool)
            dataset_pools[name] = pool

        # 记录各数据集的采样次数
        sample_counts: Dict[str, int] = {name: 0 for name in dataset_pools}

        result: List[int] = []
        dataset_names = list(dataset_pools.keys())

        while any(len(pool) > 0 for pool in dataset_pools.values()):
            # 按采样次数排序，优先采样次数少的
            # 采样次数相同时，优先剩余样本多的
            active_datasets = [
                (name, sample_counts[name], len(dataset_pools[name]))
                for name in dataset_names
                if len(dataset_pools[name]) > 0
            ]

            if not active_datasets:
                break

            # 排序：先按采样次数升序，再按剩余样本数降序
            active_datasets.sort(key=lambda x: (x[1], -x[2]))

            # 从采样次数最少的数据集中取一个样本
            name = active_datasets[0][0]
            idx = dataset_pools[name].pop(0)
            result.append(idx)
            sample_counts[name] += 1

        return result

    def __iter__(self) -> Iterator[int]:
        """生成当前进程应处理的索引迭代器"""
        # 生成完整的均衡索引序列
        indices = self._generate_balanced_indices()

        # 补齐到能被 num_replicas 整除
        if len(indices) % self.num_replicas != 0:
            padding_size = self.num_replicas - (len(indices) % self.num_replicas)
            # 用前面的索引补齐
            indices += indices[:padding_size]

        assert len(indices) % self.num_replicas == 0

        # 分配给当前进程
        indices = indices[self.rank::self.num_replicas]

        return iter(indices)

    def __len__(self) -> int:
        """返回当前进程的样本数"""
        return self.num_samples
