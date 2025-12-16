"""BasicManager implementation - 自动管理 SubTask 的生命周期"""

import os
import time
import json
import threading
import signal
import sys
import multiprocessing as mp
from typing import Dict, Any, Optional, Callable, List
from copy import deepcopy
from .subtask import SubTask

# 设置多进程启动方式为spawn（CUDA要求）
mp.set_start_method('spawn', force=True)


class BasicManager:
    """基础管理器 - 自动调度和管理多个 SubTask"""
    
    def __init__(
        self,
        config: Dict[str, Any],
        preload_fn: Callable[[Dict], Dict],
        run_fn: Callable[[Dict, Dict], Dict],
        task_generator_fn: Optional[Callable[[Dict], Optional[Dict]]] = None,
        result_handler_fn: Optional[Callable[[str, Dict, Dict], None]] = None
    ):
        """初始化管理器
        
        参数:
            config: 主配置（用于创建子任务）
            preload_fn: 子任务预加载函数 (config) -> cache
            run_fn: 子任务主函数 (config, cache) -> result
            task_generator_fn: 任务生成器 (state) -> config_overrides or None（可选，为None时只运行单个任务）
            result_handler_fn: 结果处理器 (task_id, result, state) -> None (可选)
        """
        self.base_config = config
        self.preload_fn = preload_fn
        self.run_fn = run_fn
        
        # 获取manager模式
        manager_mode = config.get("manager_settings", {}).get("mode")
        
        # 检测direct模式
        self._direct_mode = (manager_mode == "direct")
        
        # 根据模式设置task_generator和result_handler
        if manager_mode == "optuna":
            self._use_optuna = True
            self.task_generator_fn = self._optuna_task_generator
            self.result_handler_fn = self._optuna_result_handler
        elif manager_mode == "batch_configs":
            self._use_optuna = False
            self.task_generator_fn = self._batch_configs_task_generator
            self.result_handler_fn = result_handler_fn or self._default_result_handler
        else:
            # None 或其他（包括direct）：使用用户提供的
            self._use_optuna = False
            self.task_generator_fn = task_generator_fn
            self.result_handler_fn = result_handler_fn
        
        # 预加载监控的配置键（从config中读取）
        self.preload_watch_keys = config.get('manager_settings', {}).get('preload_watch_keys', None)
        
        # 获取配置
        manager_cfg = config["manager_settings"]
        self.gpus_per_subtask = manager_cfg["gpus_per_subtask"]
        self.available_gpus = manager_cfg["available_gpus"]
        self.poll_interval = manager_cfg.get("poll_interval", 1.0)  # 调度循环轮询间隔（秒）
        
        # 日志和输出
        self.task_dir = config["global_settings"]["task_dir"]
        self.logger = config.get("logger")  # 直接使用 load_config 提供的 logger
        
        # 管理器状态
        self.state: Dict[str, Any] = {
            "task_count": 0,  # 已生成任务数
            "completed_count": 0,  # 已完成任务数
            "failed_count": 0,  # 失败任务数
            "running_tasks": {},  # {subtask_id: task_info}
            "all_results": {},  # {task_id: result}
            "all_configs": {},  # {task_id: config}
        }
        
        # Direct模式：在父进程中直接运行，不创建子进程
        if self._direct_mode:
            self.logger.info("="*60)
            self.logger.info("使用Direct模式（调试模式）")
            self.logger.info("="*60)
            
            # 立即设置GPU（在preload_fn之前）
            assigned_gpus = self.available_gpus[:self.gpus_per_subtask]
            gpu_str = ",".join(map(str, assigned_gpus))
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
            self.logger.info(f"分配GPU: {assigned_gpus}")
            self.logger.info(f"CUDA_VISIBLE_DEVICES={gpu_str}")
            self.logger.info("="*60)
            
            self.num_subtasks = 0
            self.subtasks: List[SubTask] = []
            self.free_subtasks: List[SubTask] = []
            self._running = False
            self._thread = None
            self.optuna_study = None
            self.best_trial_info = None
            # Direct模式强制task_generator为None（只运行一次）
            self.task_generator_fn = None
            return  # 提前返回，跳过子任务和Optuna初始化
        
        # 正常模式：初始化子任务
        # 根据可用GPU和每个子任务GPU数量自动计算子任务数量
        self.num_subtasks = len(self.available_gpus) // self.gpus_per_subtask
        
        if self.num_subtasks == 0:
            raise ValueError(
                f"可用GPU数量({len(self.available_gpus)})不足以创建子任务 "
                f"(每个子任务需要{self.gpus_per_subtask}个GPU)"
            )
        
        # 子任务池
        self.subtasks: List[SubTask] = []
        self.free_subtasks: List[SubTask] = []
        
        # 运行控制
        self._running = False
        self._thread = None
        
        # Optuna相关
        self.optuna_study = None
        self.best_trial_info = None
        
        # 注册信号处理器（用于Ctrl+C时保存最优结果）
        if self._use_optuna:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._init_subtasks()
        
        # 如果启用Optuna，初始化study
        if self._use_optuna:
            self._init_optuna()
    
    def _init_subtasks(self):
        """初始化子任务池"""
        self.logger.info(f"初始化 {self.num_subtasks} 个子任务...")
        
        # 如果配置了 preload_watch_keys，打印日志
        if self.preload_watch_keys:
            self.logger.info(f"预加载监控配置项: {self.preload_watch_keys}")
        
        for i in range(self.num_subtasks):
            # 分配 GPU
            start_gpu = i * self.gpus_per_subtask
            assigned_gpus = self.available_gpus[start_gpu:start_gpu + self.gpus_per_subtask]
            
            # 创建子任务配置
            subtask_config = self._create_subtask_config(f"subtask_{i}", assigned_gpus)
            
            # 创建子任务
            task = SubTask(tag=f"subtask_{i}", assigned_gpus=assigned_gpus)
            # task.assigned_gpus = assigned_gpus  # 已在__init__中设置
            task.current_config_hash = None  # 用于配置比对
            task.has_preloaded = False  # 标记是否已执行过preload
            
            # 配置子任务
            task.add('update_config', subtask_config)
            time.sleep(0.1)
            
            self.subtasks.append(task)
            self.free_subtasks.append(task)
            
            self.logger.info(f"  [subtask_{i}] GPU: {assigned_gpus}")
        
        # 等待所有子任务进程启动并记录它们的 PID
        time.sleep(0.5)  # 确保进程已启动
        self.logger.info("子任务进程 PID 列表:")
        for task in self.subtasks:
            pid = task.process.pid if task.process else "N/A"
            self.logger.info(f"  [{task.tag}] PID: {pid}, GPU: {task.assigned_gpus}")
        
        self.logger.info(f"子任务初始化完成")
    

    
    def _create_subtask_config(self, subtask_name: str, gpus: List[int]) -> Dict[str, Any]:
        """为子任务创建配置 - 返回 override_dict"""
        # 设置子任务的输出目录（在管理器任务目录下）
        subtask_dir = os.path.join(self.task_dir, "subtasks", subtask_name)
        os.makedirs(subtask_dir, exist_ok=True)
        
        # 设置 GPU
        gpu_str = ",".join(map(str, gpus))
        
        # 返回 override_dict，这些配置会覆盖 load_config 生成的路径
        # 重要：传递完整的base_config，确保所有设置（包括dataset_settings等）都被正确传递
        from copy import deepcopy
        override_dict = deepcopy(self.base_config)
        
        # 覆盖子任务特定的设置
        override_dict["global_settings"]["save_dir"] = os.path.join(subtask_dir, "checkpoints")
        override_dict["global_settings"]["log_dir"] = os.path.join(subtask_dir, "logs")
        override_dict["global_settings"]["task_dir"] = subtask_dir
        override_dict["global_settings"]["study_name"] = subtask_name
        override_dict["global_settings"]["cuda_visible_devices"] = gpu_str
        override_dict["config_settings"]["log_config_on_load"] = False
        
        return override_dict
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """根据点分隔路径获取嵌套值
        
        参数:
            config: 配置字典
            path: 点分隔的路径，如 'trainer_settings.dl_settings.batch_size'
        
        返回:
            路径对应的值，如果路径不存在返回None
        """
        keys = path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
    
    def _config_hash(self, config: Dict[str, Any], watch_keys: Optional[List[str]] = None) -> str:
        """计算配置的哈希值（用于比对）
        
        参数:
            config: 配置字典
            watch_keys: 需要监控的配置键列表，支持两种格式：
                       - 顶层键: 'dataset_settings', 'backbone_settings'
                       - 嵌套路径: 'trainer_settings.dl_settings.batch_size'
                       如果为None，使用默认的关键字段
        """
        if watch_keys is None:
            # 默认只监控数据集和模型配置（Optuna搜索时不需要重新preload）
            watch_keys = ["dataset_settings", "backbone_settings"]
        
        # 收集指定路径的配置值
        key_fields = {}
        for key in watch_keys:
            value = self._get_nested_value(config, key)
            if value is not None:
                key_fields[key] = value
        
        return json.dumps(key_fields, sort_keys=True)
    
    def _check_and_update_config(self, task: SubTask, new_config: Dict[str, Any]) -> bool:
        """检查并更新子任务配置（如果需要）
        
        返回:
            是否需要重新预加载
        """
        # 使用预加载函数指定的监控路径，如果没有则使用默认
        new_hash = self._config_hash(new_config, self.preload_watch_keys)
        
        # 检查是否首次preload或监控的配置字段是否变化
        need_preload = (not task.has_preloaded) or (task.current_config_hash != new_hash)
        
        # 无论是否需要预加载，都要更新配置（因为可能有其他字段变化）
        task.add('update_config', new_config)
        
        # 注意：哈希值在preload完成后才更新（在_assign_task中）
        
        return need_preload
    
    def _schedule_loop(self):
        """调度循环（在后台线程中运行）"""
        self.logger.info("调度循环启动")
        
        while self._running:
            try:
                # 1. 检查完成的任务和中间结果（用于实时剪枝）
                for subtask in self.subtasks:
                    in_free = subtask in self.free_subtasks
                    if not in_free:
                        status = subtask.get_status()
                        
                        if status == 'COMPLETED':
                            # 任务完成
                            task_id = self.state["running_tasks"].get(subtask.tag)
                            if task_id:
                                result = subtask.get_result()
                                self._handle_completion(task_id, result, subtask)
                        
                        elif status == 'FAILED':
                            # 任务失败
                            task_id = self.state["running_tasks"].get(subtask.tag)
                            if task_id:
                                error = subtask.get_error()
                                self._handle_failure(task_id, error, subtask)
                        
                        elif status == 'RUNNING' and self._use_optuna:
                            # Optuna模式：检查中间结果并发送剪枝决策
                            inter_result = subtask.get_intermediate_result()
                            if inter_result:
                                self.logger.info(f"[Manager] 收到INTERMEDIATE: {inter_result}")
                                should_prune = self._check_optuna_pruning(subtask, inter_result)
                                subtask.send_prune_decision(should_prune)
                                self.logger.info(f"[Manager] 已发送PRUNE_DECISION: {should_prune}")
                
                # 2. 为空闲子任务分配新任务
                if self.free_subtasks:
                    # 如果没有任务生成器，只运行一次当前配置
                    if self.task_generator_fn is None:
                        if self.state['task_count'] == 0:
                            # 为所有空闲子任务分配相同的任务（使用基础配置）
                            subtask = self.free_subtasks.pop(0)
                            self._assign_task(subtask, {})
                        else:
                            # 已分配过任务，检查是否全部完成
                            finished_count = self.state['completed_count'] + self.state['failed_count']
                            if finished_count >= self.state['task_count']:
                                self.logger.info("单任务已完成，调度循环结束")
                                self._running = False
                                break
                    else:
                        # 调用任务生成器
                        task_config = self.task_generator_fn(self.state)
                        
                        if task_config is not None:
                            # 有新任务，分配给空闲子任务
                            subtask = self.free_subtasks.pop(0)
                            self._assign_task(subtask, task_config)
                        else:
                            # 没有新任务了
                            if len(self.free_subtasks) == self.num_subtasks:
                                # 所有子任务都空闲，且没有新任务，结束
                                self.logger.info("所有任务已完成，调度循环结束")
                                self._running = False
                                break
                
                # 休眠
                time.sleep(self.poll_interval)
                
            except Exception as e:
                self.logger.error(f"调度循环错误: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # 发生错误后退出循环
                self._running = False
                break
        
        self.logger.info("调度循环已停止")
    
    def _assign_task(self, subtask: SubTask, task_config_overrides: Dict[str, Any]):
        """为子任务分配任务"""
        # 生成任务 ID
        task_id = f"task_{self.state['task_count']}"
        self.state['task_count'] += 1
        
        self.logger.info(f"分配任务: {task_id} -> {subtask.tag} (GPU: {subtask.assigned_gpus})")
        
        # 提取trial：优先从state获取（Optuna模式），否则从config_overrides获取
        trial = self.state.get("current_trial") or task_config_overrides.pop("_optuna_trial", None)
        
        # 合并配置
        subtask_base = self._create_subtask_config(subtask.tag, subtask.assigned_gpus)
        
        # 深度合并 task_config_overrides
        def merge_dict(base, override):
            for k, v in override.items():
                if isinstance(v, dict) and isinstance(base.get(k), dict):
                    merge_dict(base[k], v)
                else:
                    base[k] = v
        
        merge_dict(subtask_base, task_config_overrides)
        
        # 如果有trial，添加到配置中（通过特殊字段传递）
        if trial is not None:
            subtask_base["_optuna_trial"] = trial
        
        # 保存配置
        self.state["all_configs"][task_id] = subtask_base
        
        # 检查是否需要重新预加载
        need_preload = self._check_and_update_config(subtask, subtask_base)
        
        if need_preload:
            if not subtask.has_preloaded:
                self.logger.info(f"  [{subtask.tag}] 首次运行，需要预加载")
            else:
                self.logger.info(f"  [{subtask.tag}] 配置变化，需要重新预加载")
            
            subtask.add('preload', self.preload_fn)
            # 等待preload完成
            self.logger.info(f"  [{subtask.tag}] 等待PRELOAD完成...")
            while subtask.get_status() not in ['READY', 'FAILED']:
                time.sleep(0.05)
            
            final_status = subtask.get_status()
            self.logger.info(f"  [{subtask.tag}] PRELOAD完成，状态: {final_status}")
            
            if final_status == 'FAILED':
                # Preload失败，记录错误并标记任务失败
                error = subtask.get_error()
                self.logger.error(f"  [{subtask.tag}] PRELOAD失败！")
                if error:
                    self.logger.error(f"错误信息:\n{error}")
                # 直接调用失败处理
                self._handle_failure(task_id, error or "Preload failed", subtask)
                return
            
            # preload成功后才更新哈希值和标记
            subtask.current_config_hash = self._config_hash(subtask_base, self.preload_watch_keys)
            subtask.has_preloaded = True
        
        # 添加运行命令
        self.logger.info(f"  [{subtask.tag}] 发送RUN命令")
        subtask.add('run', self.run_fn)
        
        # 记录运行状态
        self.state["running_tasks"][subtask.tag] = task_id
    
    def _handle_completion(self, task_id: str, result: Dict, subtask: SubTask):
        """处理任务完成"""
        self.logger.info(f"任务完成: {task_id} (执行者: {subtask.tag})")
        
        # 保存结果
        self.state["all_results"][task_id] = result
        self.state["completed_count"] += 1
        
        # 调用用户的结果处理器
        if self.result_handler_fn:
            try:
                self.result_handler_fn(task_id, result, self.state)
            except Exception as e:
                self.logger.error(f"[Manager] 结果处理器错误: {e}")
        
        # 释放子任务
        del self.state["running_tasks"][subtask.tag]
        self.free_subtasks.append(subtask)
        
        # 保存状态
        self._save_state()
    
    def _handle_failure(self, task_id: str, error: str, subtask: SubTask):
        """处理任务失败"""
        self.logger.error(f"任务失败: {task_id} (执行者: {subtask.tag})")
        self.logger.error(f"错误信息: {error[:200] if error else 'Unknown'}")
        
        # 记录失败
        self.state["failed_count"] += 1
        self.state["all_results"][task_id] = {"status": "failed", "error": error}
        
        # 释放子任务
        del self.state["running_tasks"][subtask.tag]
        self.free_subtasks.append(subtask)
        
        # 保存状态
        self._save_state()

    def _check_optuna_pruning(self, subtask: SubTask, inter_result: Dict):
        """检查是否应该剪枝（只report和检查，不tell）
        
        Args:
            subtask: 当前SubTask
            inter_result: 中间结果，格式 {"step": int, "value": float}
        
        Returns:
            bool: 是否应该剪枝
        """
        task_id = self.state["running_tasks"].get(subtask.tag)
        if not task_id:
            return False
        
        task_config = self.state["all_configs"].get(task_id)
        trial = task_config.get("_optuna_trial") if task_config else None
        
        if trial is None:
            return False
        
        step = inter_result["step"]
        value = inter_result["value"]
        
        trial.report(value, step)
        should_prune = trial.should_prune()
        
        # 记录剪枝决策详情
        trial_number = trial.number
        if should_prune:
            self.logger.info(f"[Optuna] Trial {trial_number} 在 step={step} 被剪枝 (score={value:.4f})")
        else:
            self.logger.info(f"[Optuna] Trial {trial_number} 在 step={step} 继续训练 (score={value:.4f})")
        
        return should_prune
    
    def _save_state(self):
        """保存管理器状态到文件"""
        state_file = os.path.join(self.task_dir, "manager_state.json")
        try:
            # 只保存可序列化的部分，移除顶层的 logger 等不可序列化对象
            def clean_config(cfg):
                """清理配置字典，移除 logger 等不可序列化对象"""
                if not isinstance(cfg, dict):
                    return cfg
                cleaned = {}
                for k, v in cfg.items():
                    # 跳过不可序列化的字段
                    if k in ('logger', 'timestamp', '_optuna_trial'):
                        continue
                    cleaned[k] = v
                return cleaned
            
            save_data = {
                "task_count": self.state["task_count"],
                "completed_count": self.state["completed_count"],
                "failed_count": self.state["failed_count"],
                "all_configs": {k: clean_config(v) for k, v in self.state["all_configs"].items()},
                "all_results": self.state["all_results"]
            }
            
            with open(state_file, 'w') as f:
                json.dump(save_data, f, indent=2)
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def _run_direct(self):
        """Direct模式：在父进程中直接运行任务"""
        self.logger.info("=" * 60)
        self.logger.info("Direct模式：开始执行任务")
        self.logger.info("=" * 60)
        
        # GPU已在__init__中设置
        
        # 1. 准备配置（使用base_config）
        config = deepcopy(self.base_config)
        
        # 2. 更新state
        task_id = "direct_task"
        self.state["task_count"] = 1
        self.state["all_configs"][task_id] = config
        
        # 3. 执行preload
        try:
            self.logger.info("执行预加载...")
            cache = self.preload_fn(config)
            self.logger.info("预加载完成")
        except Exception as e:
            self.logger.error(f"预加载失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.state["failed_count"] = 1
            self.state["all_results"][task_id] = {"status": "failed", "error": str(e)}
            return
        
        # 4. 执行run
        try:
            self.logger.info("执行任务...")
            result = self.run_fn(config, cache)
            self.logger.info(f"任务完成")
            self.logger.info(f"结果: {result}")
            
            # 保存结果
            self.state["all_results"][task_id] = result
            self.state["completed_count"] = 1
            
            # 5. 可选：调用result_handler（如果有）
            if self.result_handler_fn:
                try:
                    self.result_handler_fn(task_id, result, self.state)
                except Exception as e:
                    self.logger.error(f"结果处理器错误: {e}")
            
        except Exception as e:
            self.logger.error(f"任务执行失败: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            self.state["failed_count"] = 1
            self.state["all_results"][task_id] = {"status": "failed", "error": str(e)}
        
        self.logger.info("=" * 60)
        self.logger.info("Direct模式：执行完毕")
        self.logger.info("=" * 60)
    
    def start(self):
        """启动管理器（异步）"""
        # Direct模式：直接在当前线程执行
        if self._direct_mode:
            self._run_direct()
            return
        
        # 正常模式：启动调度线程
        if self._running:
            self.logger.warning("[Manager] 已经在运行中")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self._thread.start()
        self.logger.info("已启动（异步模式）")
    
    def wait(self, timeout: Optional[float] = None):
        """等待所有任务完成
        
        参数:
            timeout: 超时时间（秒），None 表示无限等待
        """
        # Direct模式：任务已在start中完成，立即返回
        if self._direct_mode:
            return
        
        # 正常模式：等待调度线程
        if self._thread:
            self._thread.join(timeout=timeout)
        
        # 等待完成后自动清理子进程
        for task in self.subtasks:
            task.terminate()
    
    def stop(self):
        """停止管理器"""
        # Direct模式：无需停止
        if self._direct_mode:
            return
        
        # 正常模式：停止调度线程和子任务
        self.logger.info("正在停止...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5)
        
        # 终止所有子任务
        for task in self.subtasks:
            task.terminate()
        
        self.logger.info("已停止")
    
    def get_summary(self) -> Dict[str, Any]:
        """获取管理器状态摘要"""
        return {
            "total_tasks": self.state["task_count"],
            "completed": self.state["completed_count"],
            "failed": self.state["failed_count"],
            "running": len(self.state["running_tasks"]),
            "free_subtasks": len(self.free_subtasks)
        }
    
    def __repr__(self):
        summary = self.get_summary()
        return f"<BasicManager tasks={summary['total_tasks']} completed={summary['completed']} running={summary['running']}>"
    
    # ==================== 预定义模式方法 ====================
    
    def _default_result_handler(self, task_id: str, result: Dict, state: Dict[str, Any]):
        """默认结果处理器"""
        self.logger.info(f"[DefaultHandler] {task_id} 完成: {result}")
    
    def _init_optuna(self):
        """初始化Optuna study"""
        try:
            import optuna
        except ImportError:
            self.logger.error("Optuna未安装，请运行: pip install optuna")
            self._use_optuna = False
            return
        
        search_settings = self.base_config["search_settings"]
        study_name = search_settings.get("study_name", "optuna_study")
        
        # 根据study_name自动生成storage路径（放在outputs/optuna_studies下）
        study_dir = os.path.join("./outputs/optuna_studies")
        os.makedirs(study_dir, exist_ok=True)
        storage = f"sqlite:///{os.path.join(study_dir, f'{study_name}.db')}"
        
        self.logger.info(f"Optuna storage: {storage}")
        
        # 创建pruner
        pruner_cfg = search_settings.get("pruner", {})
        pruner_type = pruner_cfg.get("type", "successive_halving")
        if pruner_type == "successive_halving":
            pruner = optuna.pruners.SuccessiveHalvingPruner(
                min_resource=pruner_cfg.get("min_resource", 1),
                reduction_factor=pruner_cfg.get("reduction_factor", 3),
                min_early_stopping_rate=pruner_cfg.get("min_early_stopping_rate", 0)
            )
        else:
            pruner = optuna.pruners.MedianPruner()
        
        # 创建sampler
        sampler_cfg = search_settings.get("sampler", {})
        sampler_type = sampler_cfg.get("type", "tpe")
        if sampler_type == "tpe":
            from datetime import datetime
            timestamp_seed = int(datetime.now().timestamp() * 1000) % (2**31)
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=sampler_cfg.get("n_startup_trials", 10),
                multivariate=sampler_cfg.get("multivariate", True),
                seed=timestamp_seed
            )
        else:
            sampler = optuna.samplers.RandomSampler()
        
        # 创建study
        self.optuna_study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            load_if_exists=True
        )
        
        # 统计已有trial
        existing_trials = len(self.optuna_study.trials)
        completed_trials = len([t for t in self.optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_trials = len([t for t in self.optuna_study.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        self.logger.info("=" * 60)
        self.logger.info("Optuna超参数搜索已启用")
        self.logger.info(f"Study name: {study_name}")
        self.logger.info(f"Storage: {storage}")
        self.logger.info("=" * 60)
        
        if existing_trials > 0:
            self.logger.info(f"已加载 {existing_trials} 个历史trial:")
            self.logger.info(f"  - 完成: {completed_trials} 个")
            self.logger.info(f"  - 剪枝: {pruned_trials} 个")
            if completed_trials > 0:
                self.logger.info(f"  - 历史最佳得分: {self.optuna_study.best_value:.4f} (Trial {self.optuna_study.best_trial.number})")
        else:
            self.logger.info("这是一个新的study，尚无历史trial")
        
        self.logger.info("提示: 按 Ctrl+C 可中断并保存当前最优结果")
        self.logger.info("=" * 60)
    
    def _optuna_task_generator(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Optuna任务生成器：使用trial生成超参数配置"""
        import optuna
        
        search_settings = self.base_config["search_settings"]
        n_trials = search_settings.get("n_trials", 100)
        
        # 检查是否已完成所有trial
        if state["task_count"] >= n_trials:
            return None
        
        # 创建新trial
        trial = self.optuna_study.ask()
        
        # 从配置中读取参数定义（参数名直接是配置路径）
        params = search_settings.get("params", {})
        
        # 生成超参数
        suggested_params = {}
        for config_path, param_config in params.items():
            param_type = param_config["type"]
            if param_type == "float":
                value = trial.suggest_float(
                    config_path,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False)
                )
            elif param_type == "int":
                value = trial.suggest_int(
                    config_path,
                    param_config["low"],
                    param_config["high"]
                )
            else:
                value = trial.suggest_categorical(config_path, param_config["choices"])
            
            suggested_params[config_path] = value
        
        # 记录trial信息到state
        state["current_trial"] = trial
        state["current_trial_params"] = suggested_params
        
        # 日志输出
        self.logger.info("=" * 60)
        self.logger.info(f"Optuna Trial {trial.number}")
        self.logger.info("=" * 60)
        self.logger.info("超参数:")
        for k, v in suggested_params.items():
            self.logger.info(f"  {k}: {v}")
        
        # 将配置路径转换为嵌套配置字典
        config_override = {}
        for config_path, value in suggested_params.items():
            # 解析路径并设置值
            parts = config_path.split(".")
            current = config_override
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        
        # 确保启用Optuna模式
        if "search_settings" not in config_override:
            config_override["search_settings"] = {}
        config_override["search_settings"]["enable"] = True
        config_override["search_settings"]["type"] = "optuna"
        
        # 传递trial_number和params（不传trial对象避免序列化问题）
        config_override["_optuna_trial_number"] = trial.number
        config_override["_optuna_params"] = suggested_params
        
        return config_override
    
    def _optuna_result_handler(self, task_id: str, result: Dict, state: Dict[str, Any]):
        """Optuna结果处理器：根据返回结果做tell并更新最佳结果"""
        import optuna
        
        trial = state.get("current_trial")
        if trial is None:
            return
        
        # 检查trial是否已完成（避免重复tell）
        trial_obj = self.optuna_study.trials[trial.number]
        if trial_obj.state.is_finished():
            return
        
        # 根据返回状态做tell
        if result.get("pruned"):
            # Worker返回pruned=True，告知study被剪枝
            self.optuna_study.tell(trial, state=optuna.trial.TrialState.PRUNED)
            step = result.get("intermediate_results", [{}])[-1].get("step", 0)
            score = result.get("score", 0)
            self.logger.info(f"Trial {trial.number} 在step {step}被剪枝 (score={score:.4f})")
        else:
            # Worker正常完成，告知study最终score
            final_score = result.get("score", float('-inf'))
            self.logger.info(f"Trial {trial.number} 完成，得分: {final_score:.4f}")
            self.optuna_study.tell(trial, final_score)
        
        # 更新最佳结果
        try:
            if self.optuna_study.best_trial is not None:
                self.best_trial_info = {
                    "trial_number": self.optuna_study.best_trial.number,
                    "best_score": self.optuna_study.best_value,
                    "best_params": self.optuna_study.best_params,
                    "completed_trials": len([t for t in self.optuna_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                }
                
                self.logger.info("=" * 60)
                self.logger.info("当前最优结果:")
                self.logger.info(f"  Trial: {self.best_trial_info['trial_number']}")
                self.logger.info(f"  Score: {self.best_trial_info['best_score']:.4f}")
                self.logger.info("  参数:")
                for k, v in self.best_trial_info['best_params'].items():
                    self.logger.info(f"    {k}: {v}")
                self.logger.info("=" * 60)
        except Exception as e:
            self.logger.warning(f"更新最优结果时出错: {e}")
    
    def _signal_handler(self, signum, frame):
        """信号处理器：Ctrl+C时保存最优结果"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("接收到中断信号，保存当前最优结果...")
        self.logger.info("=" * 60)
        
        if self.best_trial_info:
            self.logger.info("最优结果:")
            self.logger.info(f"  Trial: {self.best_trial_info['trial_number']}")
            self.logger.info(f"  Score: {self.best_trial_info['best_score']:.4f}")
            self.logger.info("  参数:")
            for k, v in self.best_trial_info['best_params'].items():
                self.logger.info(f"    {k}: {v}")
            
            # 保存到文件
            best_params_file = os.path.join(self.task_dir, "best_params.json")
            with open(best_params_file, 'w') as f:
                json.dump(self.best_trial_info, f, indent=2)
            self.logger.info(f"\n最优参数已保存到: {best_params_file}")
        else:
            self.logger.info("尚未完成任何trial")
        
        self.logger.info("程序退出")
        sys.exit(0)

    def _batch_configs_task_generator(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """批量配置任务生成器：从目录读取多个YAML文件
        
        Args:
            state: Manager状态字典
        
        Returns:
            下一个任务的配置覆盖字典，如果没有更多任务则返回None
        """
        import glob
        import yaml
        
        manager_settings = self.base_config["manager_settings"]
        batch_configs_dir = manager_settings.get("batch_configs_dir")
        
        if not batch_configs_dir:
            self.logger.error("batch_configs_dir未配置")
            return None
        
        # 第一次调用时，读取所有YAML文件
        if "batch_config_files" not in state:
            pattern = os.path.join(batch_configs_dir, "*.yaml")
            config_files = sorted(glob.glob(pattern))
            
            if not config_files:
                self.logger.warning(f"未找到配置文件: {pattern}")
                return None
            
            self.logger.info(f"找到 {len(config_files)} 个配置文件: {config_files}")
            state["batch_config_files"] = config_files
            state["batch_config_index"] = 0
        
        # 检查是否还有未处理的配置
        if state["batch_config_index"] >= len(state["batch_config_files"]):
            return None
        
        # 读取下一个配置文件
        config_file = state["batch_config_files"][state["batch_config_index"]]
        state["batch_config_index"] += 1
        
        self.logger.info(f"加载配置文件 [{state['batch_config_index']}/{len(state['batch_config_files'])}]: {config_file}")
        
        with open(config_file, 'r') as f:
            config_override = yaml.safe_load(f)
        
        return config_override
