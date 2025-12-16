"""Manager loader - 根据配置加载不同类型的 Manager

类似于 datas/loader.py, backbones/loader.py, trainers/loader.py 的设计模式
"""

from typing import Dict, Any, Callable, Optional


MANAGER_REGISTRY = {}


def register_manager(name: str):
    """注册 Manager 的装饰器"""
    def decorator(cls):
        MANAGER_REGISTRY[name] = cls
        return cls
    return decorator


def load_manager(
    config: Dict[str, Any],
    preload_fn: Callable[[Dict], Dict],
    run_fn: Callable[[Dict, Dict], Dict],
    task_generator_fn: Optional[Callable[[Dict], Optional[Dict]]] = None,
    result_handler_fn: Optional[Callable[[str, Dict, Dict], None]] = None
):
    """根据配置加载 Manager
    
    参数:
        config: 配置字典（必须包含 manager_settings）
        preload_fn: 子任务预加载函数
        run_fn: 子任务主函数
        task_generator_fn: 任务生成器函数（可选，为None时只运行当前配置的单个任务）
        result_handler_fn: 结果处理器函数（可选）
    
    返回:
        Manager 实例
    """
    manager_name = config["manager_settings"]["name"]
    
    if manager_name not in MANAGER_REGISTRY:
        raise ValueError(
            f"Manager name '{manager_name}' not registered. "
            f"Available: {list(MANAGER_REGISTRY.keys())}"
        )
    
    manager_cls = MANAGER_REGISTRY[manager_name]
    return manager_cls(
        config=config,
        preload_fn=preload_fn,
        run_fn=run_fn,
        task_generator_fn=task_generator_fn,
        result_handler_fn=result_handler_fn
    )


# 注册内置的 Manager
from .basic_manager.basic_manager import BasicManager
register_manager("basic")(BasicManager)
