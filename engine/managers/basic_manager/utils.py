"""Utility functions for SubTask."""

import pickle
from typing import Callable, Any


def serialize_function(fn: Callable) -> bytes:
    """序列化函数以便进程间传递"""
    if fn is None:
        return None
    try:
        return pickle.dumps(fn)
    except Exception as e:
        raise RuntimeError(f"无法序列化函数 {fn.__name__}: {e}. 请确保函数定义在模块顶层。")


def deserialize_function(data: bytes) -> Callable:
    """反序列化函数"""
    if data is None:
        return None
    return pickle.loads(data)


def generate_task_id() -> str:
    """生成唯一的任务ID"""
    import uuid
    return str(uuid.uuid4())[:8]
