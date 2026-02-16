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

    这个函数会对所有有梯度的参数执行 all_reduce 平均。
    """
    if not dist.is_initialized():
        return

    world_size = dist.get_world_size()
    if world_size == 1:
        return

    # 收集所有需要同步的梯度
    grads = []
    for param in model.get_pruner_parameters():
        if param.grad is not None:
            grads.append(param.grad.data)
    for param in model.get_adapter_parameters():
        if param.grad is not None:
            grads.append(param.grad.data)
    for param in model.get_discriminator_parameters():
        if param.grad is not None:
            grads.append(param.grad.data)

    # 合并成一个大 tensor 以减少通信开销
    if grads:
        # 扁平化所有梯度
        flat_grads = torch.cat([g.flatten() for g in grads])

        # All-reduce（求和后平均）
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
        flat_grads.div_(world_size)

        # 写回原始梯度
        offset = 0
        for grad in grads:
            numel = grad.numel()
            grad.copy_(flat_grads[offset:offset + numel].view_as(grad))
            offset += numel


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
