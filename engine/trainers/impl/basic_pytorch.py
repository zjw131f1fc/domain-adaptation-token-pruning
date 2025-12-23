"""Basic PyTorch Trainer.

- 支持多优化器参数挂载
- 接收数据集并内部批处理
- 提供基本训练与评估循环

此训练器保持通用，不绑定具体模型；用户可通过
`register_train_step(name, fn)` 和 `register_eval_step(fn)`
注册具体的 step 计算逻辑，以实现灵活的训练流程。
"""

from typing import Any, Dict, List, Callable, Optional
from dataclasses import dataclass
import time
import math
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


@dataclass
class _ParamGroupSpec:
    name: str
    params: List[torch.nn.Parameter]
    opt_type: str
    lr: float
    extra: Dict[str, Any]


class BasicPytorchTrainer:
    def __init__(self, config: Any, dataset_bundle: Dict[str, Any]):
        # 支持字典型 config 或具名属性访问
        self.config = config
        self.logger = getattr(config, "logger", None) or config.get("logger")

        # 训练相关设置（来自 configs/loader.py 的 DEFAULT_CONFIG.trainer_settings.dl_settings）
        if isinstance(config, dict):
            ts = config["trainer_settings"]["dl_settings"]
        else:
            ts = config.trainer_settings["dl_settings"] if isinstance(config.trainer_settings, dict) else config.trainer_settings.dl_settings
        self.epochs = int(ts["epochs"])
        self.batch_size = int(ts["batch_size"])
        # 配置项单位改为 batch
        self.print_loss_every_batches = int(ts["print_loss_every_batches"])  # 每隔多少个batch打印该batch的loss
        self.eval_every_batches = int(ts["eval_every_batches"])              # 每隔多少个batch评估一次（评估整个测试集）
        self.save_every_batches = int(ts["save_every_batches"])              # 每隔多少个batch保存一次checkpoint
        self.save_every_epochs = int(ts["save_every_epochs"])                # 每隔多少个epoch保存一次checkpoint
        self.eval_max_samples = int(ts["eval_max_samples"])                  # 每次评估抽样的最大样本数；<=0 表示全量
        
        # 梯度裁剪设置
        self.grad_clip_max_norm = ts.get("grad_clip_max_norm")  # None 表示不裁剪，float 表示裁剪阈值

        # 优化器设定
        self.optim_cfg = ts["optimizers"]
        self.param_groups: Dict[str, _ParamGroupSpec] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, LambdaLR] = {}  # 学习率调度器

        # 学习率调度器配置
        self.lr_scheduler_cfg = ts.get("lr_scheduler", {"type": "none"})

        # 数据集
        self.dataset_bundle = dataset_bundle
        self.splits = dataset_bundle["splits"]
        self.meta = dataset_bundle["meta"]

        # 训练回调（单一）：返回字典，键为优化器组名，值为该组的 loss（或包含 'loss' 键的字典）
        # 约定: train_fn(batch, device, info) -> Dict[str, float | Dict[str, float]]
        self.train_fn: Optional[Callable[[List[Any], torch.device, Dict[str, int]], Dict[str, float]]] = None
        # - 评估: 单一评估函数，返回 Dict 指标
        #   约定: eval_fn(batch, device, info) -> Dict[str, float]
        self.eval_fn: Optional[Callable[[List[Any], torch.device, Dict[str, int]], Dict[str, float]]] = None

        # 模型注册表（用于在info中传递模型）
        self.models: Dict[str, torch.nn.Module] = {}

        # 设备
        if isinstance(config, dict):
            device_str = config["global_settings"]["device"]
        else:
            device_str = config.global_settings["device"] if isinstance(config.global_settings, dict) else getattr(config.global_settings, "device")
        self.device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

        # 训练状态跟踪（用于 train_batches 方法）
        self._current_epoch = 0
        self._current_epoch_batch_idx = 0
        self._global_batch_count = 0
        self._dataloader_iterator = None
        self._current_loader = None

        # 剪枝回调（用于Optuna实时剪枝）
        self.prune_callback = None  # 函数签名: (step, score) -> should_prune (bool)
        
        # 持久化存储（跨评估调用保持状态，如EMA score等）
        # 每个新的trial会重新创建trainer，所以这个字典会被重置
        self.persistent_state: Dict[str, Any] = {}
        
        # 从cache获取Optuna状态池（如果有）
        if isinstance(dataset_bundle, dict):
            self._state_pool = dataset_bundle.get('_state_pool')
            self._cmd_queue = dataset_bundle.get('_cmd_queue')
        else:
            self._state_pool = None
            self._cmd_queue = None

        if self.logger:
            self.logger.info("BasicPytorchTrainer 初始化完成")
            self.logger.info(f"  - Device: {self.device}")
            self.logger.info(f"  - Epochs: {self.epochs}, Batch size: {self.batch_size}")

    # ---------------------- 参数与优化器管理 ----------------------
    def add_param_group(self, name: str, params: List[torch.nn.Parameter],
                        opt_type: Optional[str] = None, lr: Optional[float] = None, **extra):
        """注册一个参数组，稍后由 `setup_optimizers` 创建优化器。

        如果未指定 opt_type/lr，则从 config.trainer_settings.dl_settings.optimizers[name]
        中读取。支持 'adam' 与 'sgd'。
        """
        opt_cfg = self.optim_cfg.get(name)
        final_opt_type = (opt_type or opt_cfg["type"]).lower()
        final_lr = float(lr or opt_cfg["lr"])
        self.param_groups[name] = _ParamGroupSpec(name=name, params=list(params), opt_type=final_opt_type, lr=final_lr, extra=extra)
        if self.logger:
            self.logger.info(f"注册参数组: {name} | opt={final_opt_type} | lr={final_lr}")

    def setup_optimizers(self):
        """根据已注册的参数组创建优化器。"""
        for name, spec in self.param_groups.items():
            if spec.opt_type == "adam":
                opt = torch.optim.Adam(spec.params, lr=spec.lr)
            elif spec.opt_type == "sgd":
                momentum_val = spec.extra["momentum"] if "momentum" in spec.extra else 0.0
                momentum = float(momentum_val)
                opt = torch.optim.SGD(spec.params, lr=spec.lr, momentum=momentum)
            else:
                raise ValueError(f"不支持的优化器类型: {spec.opt_type}")
            self.optimizers[name] = opt
            if self.logger:
                self.logger.info(f"优化器就绪: {name} -> {spec.opt_type}")

    def setup_schedulers(self, total_steps: int):
        """根据配置创建学习率调度器。

        参数:
            total_steps: 总训练步数
        """
        scheduler_type = self.lr_scheduler_cfg.get("type", "none").lower()
        if scheduler_type == "none":
            if self.logger:
                self.logger.info("学习率调度器: 未启用")
            return

        warmup_ratio = self.lr_scheduler_cfg.get("warmup_ratio", 0.1)
        min_lr_ratio = self.lr_scheduler_cfg.get("min_lr_ratio", 0.01)
        warmup_steps = int(total_steps * warmup_ratio)

        if self.logger:
            self.logger.info(f"学习率调度器配置: type={scheduler_type}, warmup_steps={warmup_steps}, min_lr_ratio={min_lr_ratio}")

        for name, opt in self.optimizers.items():
            if scheduler_type == "cosine":
                # 带warmup的余弦退火
                def lr_lambda(current_step, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio):
                    if current_step < warmup_steps:
                        # warmup阶段: 线性增长
                        return float(current_step) / float(max(1, warmup_steps))
                    else:
                        # 余弦退火阶段
                        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

                scheduler = LambdaLR(opt, lr_lambda)
            elif scheduler_type == "linear":
                # 带warmup的线性衰减
                def lr_lambda(current_step, warmup_steps=warmup_steps, total_steps=total_steps, min_lr_ratio=min_lr_ratio):
                    if current_step < warmup_steps:
                        return float(current_step) / float(max(1, warmup_steps))
                    else:
                        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                        return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)

                scheduler = LambdaLR(opt, lr_lambda)
            else:
                raise ValueError(f"不支持的调度器类型: {scheduler_type}")

            self.schedulers[name] = scheduler
            if self.logger:
                self.logger.info(f"学习率调度器就绪: {name} -> {scheduler_type}")

    # ---------------------- 模型注册 ----------------------
    def register_model(self, name: str, model):
        """注册模型（会在info中传递给train_step和eval_step）。
        
        参数:
            name: 模型名称
            model: PyTorch模型实例或其他对象
        """
        self.models[name] = model
        if self.logger:
            # 仅对torch.nn.Module打印参数量
            if isinstance(model, torch.nn.Module):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f"注册模型: {name} | 总参数量: {total_params:,} | 可训练参数量: {trainable_params:,}")
            else:
                self.logger.info(f"注册模型: {name}")
    
    # ---------------------- 训练/评估回调注册 ----------------------
    def register_train_step(self, fn: Callable[[List[Any], torch.device, Dict[str, Any]], Dict[str, float]]):
        """注册训练回调（单一）。
        返回字典：{ group_name: { 'loss_name1': value1, 'loss_name2': value2, ... } }
        trainer会自动将每个组的所有loss求和后反向传播
        回调参数包含 info: { 'config': config, 'epoch': int, 'batch': int, 'models': dict, ... }
        """
        self.train_fn = fn
        if self.logger:
            self.logger.info("注册训练回调完成")

    def register_eval_step(self, fn: Callable[[List[Any], torch.device, Dict[str, Any]], Dict[str, float]]):
        """注册评估 step 函数。
        fn 接口: fn(batch, device, info) -> Dict[str, float]
        info 包含 config、训练状态信息和models字典
        """
        self.eval_fn = fn
        if self.logger:
            self.logger.info("注册评估回调完成")

    # ---------------------- 内部 DataLoader ----------------------
    @staticmethod
    def _collate_fn(batch):
        """Collate函数：保持list格式不变，支持多进程pickle"""
        return batch

    def _make_loader(self, dataset, shuffle: bool = True) -> DataLoader:
        # 启用多进程数据加载以加速预处理
        # num_workers=4: 使用4个子进程预加载数据
        # pin_memory=True: 使用页锁定内存加速CPU到GPU传输
        # prefetch_factor=2: 每个worker预取2个batch
        # persistent_workers=True: 保持worker进程存活，避免重复创建开销
        num_workers = 4 if self.device.type == 'cuda' else 0  # CPU模式下不使用多进程
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,  # 使用静态方法，可以被pickle
            num_workers=num_workers,
            pin_memory=(self.device.type == 'cuda'),
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=(num_workers > 0)
        )

    # ---------------------- 兼容路由层 ----------------------
    def run(self):
        """兼容路由层：根据配置决定使用train_full还是train_partial
        
        模式:
            - 普通模式：直接调用train_full()完整训练
            - Optuna模式：分批训练+定期评估+剪枝
        
        返回:
            - 普通模式：None
            - Optuna模式：Dict包含score、intermediate_results等
        """
        # 检测Optuna模式：优先检查manager_settings.mode
        if isinstance(self.config, dict):
            manager_settings = self.config.get("manager_settings", {})
            manager_mode = manager_settings.get("mode")
        else:
            manager_settings = getattr(self.config, "manager_settings", {})
            manager_mode = manager_settings.get("mode") if isinstance(manager_settings, dict) else None
        
        # 判断训练模式
        is_optuna_mode = (manager_mode == "optuna")
        
        if is_optuna_mode:
            if self.logger:
                self.logger.info("路由到Optuna超参数搜索模式")
            return self._run_optuna()
        else:
            if self.logger:
                self.logger.info("路由到完整训练模式")
            self.train_full()
            # 训练完成后做全测试
            eval_result = self.evaluate()
            return {"score": eval_result.get("score", 0), "eval_result": eval_result}
    
    def _run_optuna(self):
        """Optuna超参数搜索模式：分批训练+定期评估+实时剪枝"""
        if self.logger:
            self.logger.info("使用Optuna超参数搜索模式")
        
        # 获取配置参数
        if isinstance(self.config, dict):
            dl_settings = self.config["trainer_settings"]["dl_settings"]
        else:
            dl_settings = self.config.trainer_settings["dl_settings"]
        
        eval_interval = dl_settings.get("optuna_eval_interval_batches", 50)
        eval_max_samples = dl_settings.get("eval_max_samples", 50)
        
        if self.logger:
            self.logger.info(f"  - 评估间隔: 每 {eval_interval} 个batch")
            self.logger.info(f"  - 评估样本数: {eval_max_samples}")
        
        best_score = float('-inf')
        eval_step = 0
        intermediate_results = []
        
        while True:
            # 训练指定数量的batch
            train_result = self.train_partial(eval_interval)
            
            current_epoch = train_result.get("current_epoch", 0)
            total_batches = train_result.get("total_batches_trained", 0)
            training_completed = train_result.get("training_completed", False)
            
            # 执行评估
            max_samples = eval_max_samples if eval_max_samples > 0 else None
            eval_result = self.evaluate(max_samples=max_samples)
            
            # 检查评估结果是否包含score
            if "score" not in eval_result:
                available_keys = ", ".join(eval_result.keys()) if eval_result else "无"
                raise ValueError(
                    f"Optuna模式下，评估结果必须包含'score'字段！\n"
                    f"当前评估结果包含的字段: {available_keys}\n"
                    f"请在评估函数中添加'score'字段，例如：\n"
                    f"  return {{'accuracy': acc, 'score': acc}}"
                )
            
            score = eval_result["score"]
            
            # 更新最佳分数
            if score > best_score:
                best_score = score
            
            eval_step += 1
            
            # 记录中间结果
            intermediate_results.append({
                'step': total_batches,
                'score': score,
                'epoch': current_epoch
            })
            
            # 日志输出
            if self.logger:
                self.logger.info(f"[Optuna] Epoch {current_epoch}, Step {total_batches}, Eval #{eval_step}: score={score:.4f}, best={best_score:.4f}")
            
            # 实时剪枝：通过状态池上报中间结果
            if hasattr(self, '_state_pool') and self._state_pool and hasattr(self, '_cmd_queue'):
                # 更新状态池
                if self.logger:
                    self.logger.info(f"[Trainer] 上报INTERMEDIATE到状态池: step={total_batches}, score={score:.4f}")
                self._state_pool['intermediate_result'] = {
                    'step': total_batches,
                    'value': score
                }
                # 等待Manager的剪枝决策
                if self.logger:
                    self.logger.info(f"[Trainer] 等待PRUNE_DECISION...")
                while True:
                    decision_cmd = self._cmd_queue.get()
                    if decision_cmd.get('type') == 'PRUNE_DECISION':
                        should_prune = decision_cmd.get('data', False)
                        # 清空intermediate_result
                        self._state_pool['intermediate_result'] = None
                        if self.logger:
                            self.logger.info(f"[Trainer] 收到PRUNE_DECISION: {should_prune}")
                        if should_prune:
                            if self.logger:
                                self.logger.info(f"[Optuna] 在step {total_batches}被实时剪枝")
                            return {
                                'score': best_score,
                                'intermediate_results': intermediate_results,
                                'pruned': True
                            }
                        break
            elif self.prune_callback is not None:
                # 兼容旧的callback方式
                should_prune = self.prune_callback(total_batches, score)
                if should_prune:
                    if self.logger:
                        self.logger.info(f"[Optuna] 在step {total_batches}被实时剪枝")
                    return {
                        'score': best_score,
                        'intermediate_results': intermediate_results,
                        'pruned': True
                    }
            
            # 检查是否训练完成
            if training_completed:
                if self.logger:
                    self.logger.info(f"[Optuna] 训练完成，最终score={best_score:.4f}")
                # 清空状态池，避免manager重复处理
                if hasattr(self, '_state_pool') and self._state_pool:
                    self._state_pool['intermediate_result'] = None
                break
        
        # 返回结果时包含中间结果列表
        return {
            'score': best_score,
            'intermediate_results': intermediate_results,
            'pruned': False
        }
    
    # ---------------------- 训练循环 ----------------------
    def train_full(self):
        train_ds = self.splits["train"]
        if train_ds is None:
            raise ValueError("训练集不存在 (splits['train'] is None)")
        self.setup_optimizers()

        loader = self._make_loader(train_ds, shuffle=True)
        batch_count = 0
        total_planned_batches = self.epochs * len(loader)

        # 初始化学习率调度器
        self.setup_schedulers(total_planned_batches)

        # 时间跟踪：用于预估剩余时间
        start_time = time.time()
        batch_times = []  # 记录最近若干batch的耗时

        if self.logger:
            self.logger.info(f"开始训练: epochs={self.epochs}, steps/epoch={len(loader)}")

        for epoch in range(self.epochs):
            if self.logger:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}")
            for epoch_batch_idx, batch in enumerate(tqdm(loader, desc=f"Train E{epoch+1}")):
                batch_start_time = time.time()
                
                if self.train_fn is None:
                    raise RuntimeError("未注册训练回调")

                # 单次前向，返回各组的指标/损失
                info = {
                    "config": self.config,
                    "epoch": epoch + 1,
                    "batch": batch_count + 1,
                    "epoch_batch_index": epoch_batch_idx + 1,
                    "global_batch_index": batch_count + 1,
                    "total_planned_batches": total_planned_batches,
                    "models": self.models,
                    "persistent_state": self.persistent_state,
                }
                outputs = self.train_fn(batch, self.device, info)

                # 对每个优化器组进行反向与步进
                # 先清空所有优化器的梯度
                for opt in self.optimizers.values():
                    opt.zero_grad(set_to_none=True)

                # 然后分别 backward 各组的损失
                group_names = list(self.optimizers.keys())
                for idx, group_name in enumerate(group_names):
                    group_out = outputs[group_name]
                    # 如果是字典，将所有loss求和
                    if isinstance(group_out, dict):
                        loss = sum(group_out.values())
                    else:
                        loss = group_out

                    if not torch.is_tensor(loss):
                        loss = torch.tensor(float(loss), dtype=torch.float32, device=self.device)
                    # 如果不是最后一个组，保留计算图
                    retain_graph = (idx < len(group_names) - 1)
                    loss.backward(retain_graph=retain_graph)

                # 梯度裁剪（在 backward 之后，optimizer.step 之前）
                if self.grad_clip_max_norm is not None:
                    for group_name, spec in self.param_groups.items():
                        torch.nn.utils.clip_grad_norm_(spec.params, self.grad_clip_max_norm)

                # === 收集要打印的所有信息（梯度 + Loss + Metrics） ===
                if self.print_loss_every_batches and batch_count % self.print_loss_every_batches == 0:
                    # 0. 收集学习率信息
                    lr_stats = []
                    for name, opt in self.optimizers.items():
                        current_lr = opt.param_groups[0]['lr']
                        lr_stats.append(f"lr_{name}={current_lr:.2e}")

                    # 1. 收集梯度信息
                    grad_stats = []
                    for group_name, spec in self.param_groups.items():
                        grad_norms = [p.grad.norm().item() for p in spec.params if p.grad is not None]
                        if grad_norms:
                            total_norm = (sum(g**2 for g in grad_norms))**0.5
                            grad_stats.append(f"grad_{group_name}={total_norm:.4f}")

                    # 2. 收集Loss信息
                    loss_parts = []
                    metrics_info = outputs.get("metrics", {})  # 提取 metrics

                    for group_name in self.optimizers.keys():
                        group_out = outputs[group_name]
                        if isinstance(group_out, dict):
                            loss_strs = [f"{k}={float(v.detach() if isinstance(v, torch.Tensor) else v):.4f}"
                                       for k, v in group_out.items()]
                            loss_parts.append(f"{group_name}[{', '.join(loss_strs)}]")
                        else:
                            loss_val = group_out.detach() if isinstance(group_out, torch.Tensor) else group_out
                            loss_parts.append(f"{group_name}={float(loss_val):.4f}")

                    # 3. 添加关键 metrics（保留率、权重、判别器胜率等）
                    metric_strs = []

                    # 打印所有metrics（排除特定的内部指标）
                    exclude_keys = {'current_task_weight', 'current_adv_weight', 'current_sparsity_weight'}

                    # 自定义排序：将L开头的layer指标按数字排序，其他按字母排序
                    def metric_sort_key(k):
                        if k.startswith('L') and '_' in k:
                            # 提取层号，例如 "L6_kept" -> 6
                            try:
                                layer_num = int(k[1:k.index('_')])
                                # 返回(0, layer_num, metric_name) 确保layer指标在前且按数字排序
                                metric_name = k[k.index('_')+1:]
                                return (0, layer_num, metric_name)
                            except (ValueError, IndexError):
                                return (1, k)  # 解析失败，按字母排序
                        else:
                            return (1, k)  # 非layer指标，按字母排序

                    # 按自定义规则排序
                    for k in sorted(metrics_info.keys(), key=metric_sort_key):
                        if k not in exclude_keys:
                            v = metrics_info[k]
                            metric_strs.append(f"{k}={v:.3f}")

                    # 4. 组装完整消息
                    all_parts = grad_stats + loss_parts + metric_strs

                    # 5. 计算ETA
                    remaining_batches = total_planned_batches - batch_count
                    if len(batch_times) > 0 and remaining_batches > 0:
                        avg_batch_time = sum(batch_times) / len(batch_times)
                        eta_seconds = avg_batch_time * remaining_batches
                        eta_str = self._format_time(eta_seconds)
                        all_parts.append(f"ETA: {eta_str}")

                    # 6. 统一打印（多行格式，清晰易读）
                    if self.logger:
                        # 打印批次信息
                        self.logger.info(f"\n{'='*80}")
                        self.logger.info(f"[Batch {batch_count}/{total_planned_batches}]")

                        # 打印学习率信息
                        if lr_stats:
                            self.logger.info("  Learning rates: " + " | ".join(lr_stats))

                        # 打印梯度信息
                        if grad_stats:
                            self.logger.info("  Gradients: " + " | ".join(grad_stats))

                        # 打印损失信息
                        if loss_parts:
                            self.logger.info("  Losses:")
                            for loss_str in loss_parts:
                                self.logger.info(f"    - {loss_str}")

                        # 打印metrics信息
                        if metric_strs:
                            self.logger.info("  Metrics:")
                            # 每行打印3个metric
                            for i in range(0, len(metric_strs), 3):
                                batch_metrics = metric_strs[i:i+3]
                                self.logger.info(f"    {' | '.join(batch_metrics)}")

                        # 打印ETA
                        if len(batch_times) > 0 and remaining_batches > 0:
                            avg_batch_time = sum(batch_times) / len(batch_times)
                            eta_seconds = avg_batch_time * remaining_batches
                            eta_str = self._format_time(eta_seconds)
                            self.logger.info(f"  ETA: {eta_str}")

                        self.logger.info(f"{'='*80}")

                # 最后统一执行优化步骤
                for opt in self.optimizers.values():
                    opt.step()

                # 更新学习率调度器
                for scheduler in self.schedulers.values():
                    scheduler.step()

                batch_count += 1

                # 记录本batch耗时
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                # 保留最近50个batch的时间用于平滑预估
                if len(batch_times) > 50:
                    batch_times.pop(0)

                # 每隔 N 个 batch 评估一次（评估整个测试集）
                if self.eval_every_batches and self.eval_fn and batch_count % self.eval_every_batches == 0:
                    # eval_max_samples <= 0 则评估全量测试集
                    max_samples = self.eval_max_samples if self.eval_max_samples > 0 else None
                    self.evaluate(max_samples=max_samples, trigger_info=info)

                # 每隔 N 个 batch 保存检查点
                if self.save_every_batches and batch_count % self.save_every_batches == 0:
                    self._save_checkpoint(epoch=epoch, batch=batch_count)

            # 每隔 N 个 epoch 保存检查点
            if self.save_every_epochs and (epoch + 1) % self.save_every_epochs == 0:
                self._save_checkpoint(epoch=epoch, batch=None)

    # ---------------------- 评估循环 ----------------------
    def evaluate(self, max_samples: Optional[int] = None, trigger_info: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        if self.eval_fn is None:
            if self.logger:
                self.logger.info("未注册评估回调，跳过评估")
            return {}

        test_ds = self.splits["test"]
        if test_ds is None or len(test_ds) == 0:
            if self.logger:
                self.logger.info("测试集不存在或为空，跳过评估")
            return {}

        if max_samples is not None and max_samples > 0:
            # 随机抽样
            idx = torch.randperm(len(test_ds))[:min(len(test_ds), max_samples)].tolist()
            sub_ds = [test_ds[i] for i in idx]
        else:
            sub_ds = test_ds

        loader = self._make_loader(sub_ds, shuffle=False)
        agg: Dict[str, float] = {}
        n = 0
        for i, batch in enumerate(tqdm(loader, desc="Evaluate")):
            info = {
                "config": self.config,
                "epoch": (trigger_info["epoch"] if trigger_info and "epoch" in trigger_info else 0),
                "batch": (trigger_info.get("batch", 0) if trigger_info else 0),
                "epoch_batch_index": (trigger_info["epoch_batch_index"] if trigger_info and "epoch_batch_index" in trigger_info else 0),
                "global_batch_index": (trigger_info["global_batch_index"] if trigger_info and "global_batch_index" in trigger_info else 0),
                "total_planned_batches": (trigger_info["total_planned_batches"] if trigger_info and "total_planned_batches" in trigger_info else len(loader)),
                "eval_batch_index": i + 1,
                "eval_total_batches": len(loader),
                "models": self.models,
                "persistent_state": self.persistent_state,
            }
            metrics = self.eval_fn(batch, self.device, info)
            # 简单加和求平均
            for k, v in metrics.items():
                if k in agg:
                    agg[k] = agg[k] + float(v)
                else:
                    agg[k] = float(v)
            n += 1
        if n > 0:
            for k in list(agg.keys()):
                agg[k] /= n
        if self.logger:
            self.logger.info(f"评估完成: {agg}")
        return agg

    # ---------------------- Checkpoint 保存 ----------------------
    def _save_checkpoint(self, epoch: int, batch: Optional[int]):
        # 基于自动生成的 experiment_tag 创建文件夹
        # 根目录来自 config.global_settings.save_dir
        if isinstance(self.config, dict):
            gs = self.config["global_settings"]
        else:
            gs = self.config.global_settings
        save_root = gs["save_dir"]
        tag = gs["experiment_tag"]

        import os
        out_dir = os.path.join(save_root, tag)
        os.makedirs(out_dir, exist_ok=True)

        # 文件名包含 epoch 与 batch（如有）
        if batch is not None:
            fname = f"epoch{epoch+1}_batch{batch}.pt"
        else:
            fname = f"epoch{epoch+1}.pt"
        path = os.path.join(out_dir, fname)

        # 保存优化器与参数组的状态（通用）
        state = {
            "epoch": epoch + 1,
            "batch": batch,
            "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            "param_groups": {name: [p.detach().cpu() for p in spec.params] for name, spec in self.param_groups.items()},
        }
        torch.save(state, path)
        if self.logger:
            self.logger.info(f"Checkpoint 保存: {path}")
    
    # ---------------------- 训练指定batch数 ----------------------
    def train_partial(self, num_batches: int) -> Dict[str, Any]:
        """训练指定数量的batch，支持连续调用继续训练。
        
        该方法在训练过程中不会触发评估，只进行纯粹的训练。
        可以多次调用此方法，训练会从上次停止的地方继续。
        
        参数:
            num_batches: 要训练的batch数量
        
        返回:
            训练统计信息字典，包含:
            - 各优化器组的平均损失
            - batches_trained: 本次训练的batch数
            - total_batches_trained: 累计训练的总batch数
            - current_epoch: 当前所处的epoch
            - training_completed: 是否完成所有训练（达到epochs）
        """
        train_ds = self.splits["train"]
        if train_ds is None:
            raise ValueError("训练集不存在 (splits['train'] is None)")
        
        # 首次调用时初始化
        if self._current_loader is None:
            self.setup_optimizers()
            self._current_loader = self._make_loader(train_ds, shuffle=True)
            self._dataloader_iterator = iter(self._current_loader)
            # 估算总步数用于学习率调度（epochs * steps_per_epoch）
            total_steps = self.epochs * len(self._current_loader)
            self.setup_schedulers(total_steps)
            if self.logger:
                self.logger.info(f"初始化训练: batch_size={self.batch_size}, steps/epoch={len(self._current_loader)}")
        
        if self.train_fn is None:
            raise RuntimeError("未注册训练回调")
        
        # 统计信息
        batch_losses = {name: [] for name in self.optimizers.keys()}
        batches_trained = 0
        
        if self.logger:
            self.logger.info(f"开始训练 {num_batches} 个batch (当前epoch={self._current_epoch+1}, 已训练batch={self._global_batch_count})")
        
        for _ in range(num_batches):
            try:
                if self._dataloader_iterator is None:
                    raise RuntimeError("DataLoader iterator 未初始化")
                batch = next(self._dataloader_iterator)
            except StopIteration:
                # 当前epoch结束，开始新的epoch
                self._current_epoch += 1
                self._current_epoch_batch_idx = 0
                
                # 每隔 N 个 epoch 保存检查点
                if self.save_every_epochs and self._current_epoch % self.save_every_epochs == 0:
                    self._save_checkpoint(epoch=self._current_epoch-1, batch=None)
                
                # 重新创建dataloader和iterator
                self._current_loader = self._make_loader(train_ds, shuffle=True)
                self._dataloader_iterator = iter(self._current_loader)
                
                if self.logger:
                    self.logger.info(f"Epoch {self._current_epoch} 完成，进入 Epoch {self._current_epoch+1}")
                
                batch = next(self._dataloader_iterator)
            
            # 执行训练步骤
            info = {
                "config": self.config,
                "epoch": self._current_epoch + 1,
                "batch": self._global_batch_count + 1,
                "epoch_batch_index": self._current_epoch_batch_idx + 1,
                "global_batch_index": self._global_batch_count + 1,
                "total_planned_batches": -1,  # 未知总数
                "models": self.models,
                "persistent_state": self.persistent_state,
            }
            outputs = self.train_fn(batch, self.device, info)
            
            # 梯度清零
            for opt in self.optimizers.values():
                opt.zero_grad(set_to_none=True)
            
            # 反向传播
            group_names = list(self.optimizers.keys())
            for idx, group_name in enumerate(group_names):
                group_out = outputs[group_name]
                # 如果是字典，将所有loss求和
                if isinstance(group_out, dict):
                    loss = sum(group_out.values())
                else:
                    loss = group_out
                
                if not torch.is_tensor(loss):
                    loss = torch.tensor(float(loss), dtype=torch.float32, device=self.device)
                
                # 记录损失（先detach避免警告）
                batch_losses[group_name].append(loss.detach().item())
                
                # 保留计算图（除了最后一个组）
                retain_graph = (idx < len(group_names) - 1)
                loss.backward(retain_graph=retain_graph)
            
            # 优化步骤
            for opt in self.optimizers.values():
                opt.step()

            # 更新学习率调度器
            for scheduler in self.schedulers.values():
                scheduler.step()

            # 更新计数器
            self._current_epoch_batch_idx += 1
            self._global_batch_count += 1
            batches_trained += 1
            
            # 每隔 N 个 batch 保存检查点
            if self.save_every_batches and self._global_batch_count % self.save_every_batches == 0:
                self._save_checkpoint(epoch=self._current_epoch, batch=self._global_batch_count)
        
        # 计算平均损失
        avg_losses = {}
        for group_name, losses in batch_losses.items():
            if losses:
                avg_losses[f"{group_name}_loss"] = sum(losses) / len(losses)
        
        avg_losses["batches_trained"] = batches_trained
        avg_losses["total_batches_trained"] = self._global_batch_count
        avg_losses["current_epoch"] = self._current_epoch + 1
        # 检查是否完成所有训练（达到配置的epochs）
        avg_losses["training_completed"] = self._current_epoch >= self.epochs
        
        if self.logger:
            msg_parts = [f"{k}={v:.4f}" for k, v in avg_losses.items() if k.endswith("_loss")]
            self.logger.info(f"训练完成 {batches_trained} 个batch: " + ", ".join(msg_parts))
        
        return avg_losses
    
    # ---------------------- 辅助方法 ----------------------
    def _format_time(self, seconds: float) -> str:
        """格式化秒数为可读的时间字符串"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}min"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"
