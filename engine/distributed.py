"""分布式训练工具函数

提供 DDP 分布式训练所需的工具函数。
"""

import os
import torch
import torch.distributed as dist


def setup_distributed():
    """初始化分布式环境

    使用 torchrun 启动时，环境变量会自动设置
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return rank, world_size, local_rank, device


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """判断是否是主进程"""
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """在所有进程间平均 tensor"""
    if not dist.is_initialized():
        return tensor

    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor = tensor / dist.get_world_size()
    return tensor


def sync_gradients(model):
    """手动同步所有可训练参数的梯度

    由于我们的模型结构特殊（冻结主干 + 可训练小模块），
    使用手动梯度同步比 DDP 更灵活。

    这个函数会对所有可训练参数执行 all_reduce 平均。

    注意：不同 rank 可能因为数据差异/条件分支导致某些参数在该步没有梯度（grad=None）。
    如果只对“有梯度的参数”做 all_reduce，会造成各 rank 参与的 collective 不一致，从而死锁。
    因此这里对每个参数都执行一次 all_reduce：有梯度就同步真实梯度，没有梯度就同步一个同形状的零张量。
    """
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    # 固定顺序遍历参数，确保所有 rank 调用的 collective 完全一致
    params = []
    params.extend(list(model.get_pruner_parameters()))
    params.extend(list(model.get_adapter_parameters()))
    params.extend(list(model.get_discriminator_parameters()))

    for param in params:
        if param.grad is None:
            zero_grad = torch.zeros_like(param.data)
            dist.all_reduce(zero_grad, op=dist.ReduceOp.SUM)
            continue

        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data.div_(world_size)


def broadcast_model_params(model, src: int = 0):
    """从 src 进程广播模型参数到所有进程

    在训练开始前调用，确保所有进程的模型参数一致。
    """
    if not dist.is_initialized():
        return

    for param in model.get_pruner_parameters():
        dist.broadcast(param.data, src=src)
    for param in model.get_adapter_parameters():
        dist.broadcast(param.data, src=src)
    for param in model.get_discriminator_parameters():
        dist.broadcast(param.data, src=src)
