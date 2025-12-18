import os
import sys
from typing import List, Optional, Dict, Any

def worker_launcher(
    gpu_ids: List[int],
    cmd_queue,
    status_queue,
    state_pool,
    config_file: Optional[str],
    config_overrides: Optional[Dict[str, Any]]
):
    """
    Launcher for SubTask worker that sets CUDA_VISIBLE_DEVICES before importing torch.

    关键：CUDA_VISIBLE_DEVICES必须在import torch之前设置，以实现子进程级别的GPU隔离。
    设置后，PyTorch会将GPU重新编号从0开始，因此所有CUDA代码都应使用相对编号。
    """
    # Set CUDA_VISIBLE_DEVICES for this subprocess
    if gpu_ids:
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        # 注意：设置后，PyTorch看到的GPU编号会从0开始重新映射
        # 例如：CUDA_VISIBLE_DEVICES="2,3,4" -> PyTorch看到[0,1,2]

    # Now import the actual worker
    # This ensures torch is imported AFTER env var is set
    from .worker import subtask_worker

    subtask_worker(cmd_queue, status_queue, state_pool, config_file, config_overrides)
