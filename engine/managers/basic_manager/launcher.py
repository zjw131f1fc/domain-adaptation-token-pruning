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
    """
    # Set CUDA_VISIBLE_DEVICES
    if gpu_ids:
        gpu_str = ",".join(map(str, gpu_ids))
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
        # print(f"[Launcher] Set CUDA_VISIBLE_DEVICES={gpu_str}", flush=True)
    
    # Now import the actual worker
    # This ensures torch is imported AFTER env var is set
    from .worker import subtask_worker
    
    subtask_worker(cmd_queue, status_queue, state_pool, config_file, config_overrides)
