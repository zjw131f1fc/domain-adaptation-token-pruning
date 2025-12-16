"""SubTask implementation."""

import multiprocessing as mp
import time
from typing import Dict, Any, Optional, Callable
from .utils import serialize_function, generate_task_id
from .worker import subtask_worker
from .launcher import worker_launcher


class SubTask:
    """独立进程中运行的任务单元"""
    
    def __init__(self, tag: Optional[str] = None, config_file: Optional[str] = None, assigned_gpus = None):
        """创建任务
        
        参数:
            tag: 任务标识（可选，默认自动生成）
            config_file: 配置文件路径（可选）
            assigned_gpus: 分配给该任务的GPU ID列表
        """
        self.tag = tag or generate_task_id()
        self.config_file = config_file
        self.assigned_gpus = assigned_gpus or []
        
        # 进程间通信
        self.cmd_queue = mp.Queue()
        self.status_queue = mp.Queue()
        
        # 共享状态池（跨进程共享）
        manager = mp.Manager()
        self.state_pool = manager.dict({
            'status': 'IDLE',
            'message': 'Worker not started',
            'intermediate_result': None
        })
        
        # 进程对象
        self.process: Optional[mp.Process] = None
        
        # 状态跟踪
        self._state = 'IDLE'
        self._message = 'Created'
        self._config = None
        self._result = None
        self._error = None
        
        # 配置变化跟踪
        self.current_config_hash = None  # 存储当前配置的哈希值，用于检测配置变化
        
        # 启动进程
        self._start_process()
    
    def _start_process(self, config_overrides: Optional[Dict[str, Any]] = None):
        """启动工作进程"""
        self.process = mp.Process(
            target=worker_launcher,
            args=(
                self.assigned_gpus,
                self.cmd_queue,
                self.status_queue,
                self.state_pool,  # 传递共享状态池
                self.config_file,
                config_overrides
            )
        )
        self.process.start()
        
        # 等待进程启动
        time.sleep(0.1)
        self._poll_status()
    
    def _poll_status(self):
        """轮询状态更新"""
        try:
            while not self.status_queue.empty():
                msg = self.status_queue.get_nowait()
                
                msg_type = msg.get('type')
                
                if msg_type == 'STATUS':
                    self._state = msg.get('state', self._state)
                    self._message = msg.get('message', '')
                    self._error = msg.get('error')
                    
                elif msg_type == 'CONFIG':
                    self._config = msg.get('config')
                    
                elif msg_type == 'RESULT':
                    self._result = msg.get('result')
                    
        except Exception:
            pass
    
    def add(self, command: str, data: Any = None):
        """向任务添加命令
        
        参数:
            command: 命令类型 ('update_config', 'preload', 'run')
            data: 命令数据
                - 'update_config': 配置字典
                - 'preload': 预加载函数或 None（使用默认）
                - 'run': 运行函数或 None（使用默认）
        """
        command_map = {
            'update_config': 'UPDATE_CONFIG',
            'preload': 'PRELOAD',
            'run': 'RUN'
        }
        
        cmd_type = command_map.get(command)
        if cmd_type is None:
            raise ValueError(f"Unknown command: {command}")
        
        # 特殊处理：序列化函数
        if command in ['preload', 'run'] and callable(data):
            data = serialize_function(data)
        
        # 发送命令
        self.cmd_queue.put({'type': cmd_type, 'data': data})
        
        # 等待一小段时间让进程处理
        time.sleep(0.1)
        self._poll_status()
    
    def get_intermediate_result(self) -> Optional[Dict[str, Any]]:
        """从状态池读取中间结果
        
        返回:
            中间结果字典，包含 step 和 value
        """
        if self.state_pool and 'intermediate_result' in self.state_pool:
            return self.state_pool['intermediate_result']
        return None
    
    def send_prune_decision(self, should_prune: bool):
        """发送剪枝决策给子进程
        
        参数:
            should_prune: True表示剪枝，False表示继续
        """
        self.cmd_queue.put({'type': 'PRUNE_DECISION', 'data': should_prune})
    
    def get_status(self) -> str:
        """返回当前状态"""
        self._poll_status()
        return self._state
    
    def get_result(self) -> Optional[Dict[str, Any]]:
        """获取执行结果（COMPLETED 状态时可用）"""
        self._poll_status()
        return self._result
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """获取当前配置"""
        self._poll_status()
        return self._config
    
    def get_message(self) -> str:
        """获取状态消息"""
        self._poll_status()
        return self._message
    
    def get_error(self) -> Optional[str]:
        """获取错误信息（FAILED 状态时可用）"""
        self._poll_status()
        return self._error
    
    def is_alive(self) -> bool:
        """检查进程是否存活"""
        return self.process is not None and self.process.is_alive()
    
    def wait(self, target_state: str = 'COMPLETED', timeout: Optional[float] = None, poll_interval: float = 0.5):
        """等待任务达到目标状态
        
        参数:
            target_state: 目标状态
            timeout: 超时时间（秒），None 表示无限等待
            poll_interval: 轮询间隔（秒）
        
        返回:
            是否达到目标状态
        """
        start_time = time.time()
        
        while True:
            current_state = self.get_status()
            
            if current_state == target_state:
                return True
            
            if current_state == 'FAILED':
                return False
            
            if timeout is not None and (time.time() - start_time) > timeout:
                return False
            
            time.sleep(poll_interval)
    
    def terminate(self):
        """终止任务进程"""
        if self.is_alive():
            self.cmd_queue.put({'type': 'TERMINATE'})
            time.sleep(0.2)
            
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
                
            if self.process.is_alive():
                self.process.kill()
                self.process.join()
        
        self._state = 'TERMINATED'
    
    def __repr__(self):
        return f"<SubTask tag={self.tag} state={self._state}>"
