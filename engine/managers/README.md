# Manager 模块结构

## 目录结构

```
manager/
├── __init__.py              # 包入口，导出 load_manager, register_manager
├── loader.py                # Manager 加载器（registry 模式）
└── basic_manager/           # BasicManager 实现
    ├── __init__.py          # BasicManager 包入口
    ├── basic_manager.py     # 核心 Manager 逻辑
    ├── subtask.py           # SubTask 实现
    ├── worker.py            # 子进程 worker 函数
    └── utils.py             # 工具函数
```

## 使用方式

```python
from manager import load_manager
from configs.loader import load_config

# 加载配置
config = load_config(override_dict={
    'manager_settings': {
        'type': 'basic',
        'num_subtasks': 4,
        'gpus_per_subtask': 1,
        'available_gpus': [4, 5, 6, 7]
    }
})

# 创建 Manager
manager = load_manager(
    config=config,
    preload_fn=your_preload_fn,
    run_fn=your_run_fn,
    task_generator_fn=your_task_generator_fn,
    result_handler_fn=your_result_handler_fn  # 可选
)

# 启动并等待
manager.start()
manager.wait()
```

## 测试文件

- `test_manager_quick.py` - 快速功能测试（推荐）
- `test_manager.py` - 完整集成测试
- `example_new_api.py` - 使用示例

运行测试：
```bash
python test_manager_quick.py
```

## 扩展新的 Manager 类型

1. 在 `manager/` 下创建新目录，如 `optuna_manager/`
2. 实现 Manager 类
3. 在 `loader.py` 中注册：

```python
from .optuna_manager import OptunaManager
register_manager("optuna")(OptunaManager)
```

4. 在配置中使用：
```python
'manager_settings': {'type': 'optuna', ...}
```
