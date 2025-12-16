"""SubTask worker process."""

from engine.configs.loader import load_config  # 自动设置sys.path和环境变量
import sys
import traceback
from typing import Dict, Any, Optional, Callable
from .utils import deserialize_function


def default_preload(config: Dict[str, Any]) -> Dict[str, Any]:
    """默认预加载函数"""
    return {}


def default_run(config: Dict[str, Any], cache: Dict[str, Any]) -> Dict[str, Any]:
    """默认运行函数"""
    return {
        "message": "Default run executed"
    }


def subtask_worker(
    cmd_queue,
    status_queue,
    state_pool,  # 共享状态池
    config_file: Optional[str],
    config_overrides: Optional[Dict[str, Any]]
):
    """SubTask 工作进程主函数
    
    参数:
        cmd_queue: 接收命令的队列
        status_queue: 发送状态的队列
        config_file: 配置文件路径
        config_overrides: 初始配置覆盖
    """
    import os
    
    config = None
    cache = {}
    
    # 初始化状态池
    state_pool['status'] = 'IDLE'
    state_pool['message'] = 'Worker started'
    
    try:
        # 初始状态
        status_queue.put({'type': 'STATUS', 'state': 'IDLE', 'message': 'Worker started'})
        
        # 打印进程信息
        pid = os.getpid()
        print(f"[SubTask Worker PID={pid}] 启动", flush=True)
        
        # 子任务启动时等待第一个 UPDATE_CONFIG 命令，不预先加载配置
        # 这样避免不必要的配置加载（会被立即覆盖）
        print(f"[SubTask Worker PID={pid}] 等待配置...", flush=True)
        
        # 命令循环
        while True:
            cmd = cmd_queue.get()  # 阻塞等待命令
            
            cmd_type = cmd.get('type')
            
            if cmd_type == 'UPDATE_CONFIG':
                # 更新配置
                try:
                    status_queue.put({'type': 'STATUS', 'state': 'UPDATING', 'message': 'Updating config...'})
                    
                    new_overrides = cmd.get('data')
                    # skip_auto_paths=True 表示使用 override_dict 中提供的路径，不自动生成
                    config = load_config(
                        override_dict=new_overrides,
                        override_file=config_file,
                        skip_auto_paths=True
                    )
                    
                    # load_config 会自动设置 CUDA_VISIBLE_DEVICES 环境变量
                    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')
                    print(f"[SubTask Worker PID={pid}] 配置更新完成, CUDA_VISIBLE_DEVICES={cuda_visible}", flush=True)
                    
                    status_queue.put({'type': 'CONFIG', 'config': config})
                    status_queue.put({'type': 'STATUS', 'state': 'READY', 'message': 'Config updated'})
                    
                except Exception as e:
                    error_msg = f"Config update failed: {str(e)}"
                    full_traceback = traceback.format_exc()
                    print(f"[Worker] Config更新失败:", file=sys.stderr, flush=True)
                    print(full_traceback, file=sys.stderr, flush=True)
                    status_queue.put({
                        'type': 'STATUS',
                        'state': 'FAILED',
                        'message': error_msg,
                        'error': full_traceback
                    })
            
            elif cmd_type == 'PRELOAD':
                # 执行预加载
                try:
                    print(f"[Worker] 收到PRELOAD命令", flush=True)
                    status_queue.put({'type': 'STATUS', 'state': 'PRELOADING', 'message': 'Preloading...'})
                    
                    if config is None:
                        raise RuntimeError("Config not loaded, call UPDATE_CONFIG first")
                    
                    # 获取预加载函数
                    preload_fn_data = cmd.get('data')
                    if preload_fn_data is not None:
                        preload_fn = deserialize_function(preload_fn_data)
                    else:
                        preload_fn = default_preload
                    
                    # 执行预加载
                    result = preload_fn(config)
                    
                    cache = result if result else {}
                    cache['config'] = config
                    
                    # 为Optuna模式将状态池引用放入cache和dataset_bundle
                    manager_mode = config.get('manager_settings', {}).get('mode')
                    if manager_mode == 'optuna':
                        cache['_state_pool'] = state_pool
                        cache['_cmd_queue'] = cmd_queue
                        # 同时注入到dataset_bundle中，供trainer读取
                        if 'dataset_bundle' in cache and isinstance(cache['dataset_bundle'], dict):
                            cache['dataset_bundle']['_state_pool'] = state_pool
                            cache['dataset_bundle']['_cmd_queue'] = cmd_queue
                    
                    status_queue.put({'type': 'STATUS', 'state': 'READY', 'message': 'Preload completed'})
                    
                except Exception as e:
                    error_msg = f"Preload failed: {str(e)}"
                    full_traceback = traceback.format_exc()
                    # 打印完整错误到标准错误，便于调试
                    print(f"[Worker] Preload失败:", file=sys.stderr, flush=True)
                    print(full_traceback, file=sys.stderr, flush=True)
                    status_queue.put({
                        'type': 'STATUS',
                        'state': 'FAILED',
                        'message': error_msg,
                        'error': full_traceback
                    })
            
            elif cmd_type == 'RUN':
                # 执行主流程
                try:
                    status_queue.put({'type': 'STATUS', 'state': 'RUNNING', 'message': 'Running...'})
                    
                    # 检查cache是否为空（config字段是preload时自动添加的）
                    if not cache or 'config' not in cache:
                        raise RuntimeError("Cache is empty, call PRELOAD first")
                    
                    # 获取运行函数
                    run_fn_data = cmd.get('data')
                    if run_fn_data is not None:
                        run_fn = deserialize_function(run_fn_data)
                    else:
                        run_fn = default_run
                    
                    # 执行运行
                    result = run_fn(config, cache)
                    
                    status_queue.put({'type': 'RESULT', 'result': result})
                    status_queue.put({'type': 'STATUS', 'state': 'COMPLETED', 'message': 'Run completed'})
                    
                except Exception as e:
                    error_msg = f"Run failed: {str(e)}"
                    full_traceback = traceback.format_exc()
                    # 打印完整错误到标准错误，便于调试
                    print(f"[Worker] 任务执行失败:", file=sys.stderr)
                    print(full_traceback, file=sys.stderr)
                    status_queue.put({
                        'type': 'STATUS',
                        'state': 'FAILED',
                        'message': error_msg,
                        'error': full_traceback
                    })
            
            elif cmd_type == 'TERMINATE':
                # 终止进程
                status_queue.put({'type': 'STATUS', 'state': 'TERMINATED', 'message': 'Worker terminated'})
                break
            
            else:
                status_queue.put({
                    'type': 'STATUS',
                    'state': 'FAILED',
                    'message': f'Unknown command type: {cmd_type}'
                })
    
    except Exception as e:
        # 顶层异常捕获
        error_msg = f"Worker crashed: {str(e)}"
        status_queue.put({
            'type': 'STATUS',
            'state': 'FAILED',
            'message': error_msg,
            'error': traceback.format_exc()
        })
