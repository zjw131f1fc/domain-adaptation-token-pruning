"""Manager package.

Usage:
    from manager import load_manager
    from configs.loader import load_config
    
    config = load_config(...)
    manager = load_manager(config, preload_fn, run_fn, task_generator_fn, result_handler_fn)
    manager.start()
    manager.wait()
"""

from .loader import load_manager, register_manager
from .basic_manager import BasicManager

__all__ = ['load_manager', 'register_manager', 'BasicManager']
