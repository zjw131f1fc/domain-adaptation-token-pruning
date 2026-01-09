"""HuggingFace Trainer 包装器 - 增强版

支持:
1. 自定义optimizer (parameter groups with different learning rates)
2. FSDP多卡训练
3. 自定义训练逻辑(通过model.forward())
4. 进度跟踪和temperature annealing
5. 与VisionTokenPruningModel无缝集成

使用方式:
    在配置文件中设置:
    trainer_settings:
      type: "dl"
      name: "hf-trainer"
      dl_settings:
        batch_size: 4
        epochs: 3
        optimizers:  # 会被用于创建parameter groups
          token_merger:
            lr: 1.0e-5
          layer_pruners:
            lr: 1.0e-4
          discriminator:
            lr: 6.0e-4
      hf_settings:  # 直接映射到 TrainingArguments
        gradient_accumulation_steps: 4
        bf16: true
        warmup_ratio: 0.1
        # FSDP配置
        fsdp: "full_shard"
        fsdp_config:
          fsdp_transformer_layer_cls_to_wrap: ["LlamaDecoderLayer"]
          fsdp_backward_prefetch: "backward_pre"
"""

from typing import Any, Dict, Optional, List
import os
import torch
import torch.nn as nn
from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from torch.utils.data import Dataset


class ProgressTrackingCallback(TrainerCallback):
    """回调：跟踪训练进度并更新模型的temperature等"""

    def __init__(self, model, logger=None):
        self.model = model
        self.logger = logger

    def on_step_end(self, args, state, control, **kwargs):
        """每个训练step结束时更新模型状态"""
        if hasattr(self.model, 'set_training_progress'):
            total_steps = state.max_steps
            current_step = state.global_step
            self.model.set_training_progress(current_step, total_steps)


class CustomLoggerCallback(TrainerCallback):
    """自定义回调：将 HF Trainer 的日志输出到自定义 logger"""

    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        """当 HF Trainer 记录日志时触发"""
        if logs is None or not self.logger:
            return

        # 过滤出重要的指标
        important_keys = [
            'loss', 'learning_rate', 'epoch',
            'avg_tokens', 'disc_real_acc', 'disc_fake_acc'
        ]

        filtered_logs = {k: v for k, v in logs.items() if any(key in k for key in important_keys)}

        if filtered_logs:
            log_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                   for k, v in filtered_logs.items()])
            self.logger.info(f"[Step {state.global_step}] {log_str}")


class CustomDataset(Dataset):
    """适配器：将我们的数据格式转换为 HuggingFace Dataset 格式"""

    def __init__(self, samples):
        """
        Args:
            samples: 可以是 DistillDataset、QADataset 或普通列表
        """
        if hasattr(samples, 'samples'):
            self.samples = samples.samples
        elif isinstance(samples, list):
            self.samples = samples
        else:
            self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class CustomCollator:
    """自定义data collator - 保持原始batch格式"""

    def __call__(self, batch):
        """
        Args:
            batch: List of samples

        Returns:
            List of samples (不做任何处理，直接返回)
        """
        return batch  # VisionTokenPruningModel.forward()期望收到List[Dict]


class VTPTrainer(Trainer):
    """自定义Trainer，支持VisionTokenPruningModel

    主要修改:
    1. create_optimizer: 支持parameter groups
    2. compute_loss: 适配VisionTokenPruningModel的输出格式
    """

    def create_optimizer(self):
        """创建optimizer，支持parameter groups

        如果model有create_optimizer_groups方法，使用它创建parameter groups。
        否则使用默认方式。
        """
        if self.optimizer is None:
            opt_model = self.model

            # 检查model是否提供optimizer groups
            if hasattr(opt_model, 'create_optimizer_groups'):
                param_groups = opt_model.create_optimizer_groups()

                if self.args.optim == "adamw_torch":
                    from torch.optim import AdamW
                    self.optimizer = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
                elif self.args.optim == "adam":
                    from torch.optim import Adam
                    self.optimizer = Adam(param_groups, betas=(0.9, 0.999), eps=1e-8)
                else:
                    # 使用HF的默认optimizer
                    super().create_optimizer()

                print(f"✓ Created optimizer with {len(param_groups)} parameter groups")
            else:
                # 使用默认方式
                super().create_optimizer()

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        计算loss

        VisionTokenPruningModel.forward(batch) 返回:
            {
                'loss': tensor,
                'avg_tokens': float,
                'disc_real_acc': float,
                ...
            }
        """
        # inputs 是 batch (List[Dict])
        outputs = model(inputs)

        loss = outputs['loss']

        # 记录额外的metrics到trainer的log
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            # 将metrics添加到log_history
            for key, value in outputs.items():
                if key != 'loss' and isinstance(value, (int, float)):
                    self.log({key: value})

        return (loss, outputs) if return_outputs else loss


class HFTrainerWrapper:
    """HuggingFace Trainer 包装器 - 增强版

    支持自定义optimizer、FSDP、进度跟踪等功能。
    """

    def __init__(self, config: Any, dataset_bundle: Dict[str, Any]):
        """
        Args:
            config: 配置对象
            dataset_bundle: 数据集 bundle
        """
        self.config = config
        self.logger = getattr(config, "logger", None) or config.get("logger")
        self.dataset_bundle = dataset_bundle
        self.splits = dataset_bundle["splits"]
        self.meta = dataset_bundle["meta"]

        # HuggingFace Trainer 实例（延迟初始化）
        self.hf_trainer: Optional[VTPTrainer] = None
        self.model = None

        # 从配置中提取设置
        if isinstance(config, dict):
            ts = config["trainer_settings"]["dl_settings"]
            self.hf_settings = config["trainer_settings"].get("hf_settings", {})
            gs = config["global_settings"]
        else:
            ts = (
                config.trainer_settings["dl_settings"]
                if isinstance(config.trainer_settings, dict)
                else config.trainer_settings.dl_settings
            )
            self.hf_settings = (
                config.trainer_settings.get("hf_settings", {})
                if isinstance(config.trainer_settings, dict)
                else getattr(config.trainer_settings, "hf_settings", {})
            )
            gs = config.global_settings

        self.batch_size = int(ts["batch_size"])
        self.epochs = int(ts["epochs"])
        self.save_dir = gs["save_dir"] if isinstance(gs, dict) else gs.save_dir
        self.experiment_tag = gs.get("experiment_tag", "default") if isinstance(gs, dict) else gs.experiment_tag

        if self.logger:
            self.logger.info("HFTrainerWrapper 初始化完成")
            self.logger.info(f"  - Batch size: {self.batch_size}, Epochs: {self.epochs}")
            if 'fsdp' in self.hf_settings:
                self.logger.info(f"  - FSDP: {self.hf_settings['fsdp']}")

    def build_trainer(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        trainer_class: type = None,
        trainer_kwargs: Dict[str, Any] = None,
    ):
        """构建 HuggingFace Trainer

        Args:
            model: 要训练的模型(通常是VisionTokenPruningModel)
            tokenizer: 分词器（可选）
            trainer_class: 自定义 Trainer 类（可选，默认使用VTPTrainer）
            trainer_kwargs: 传递给 Trainer 的额外参数（可选）

        Returns:
            self (支持链式调用)
        """
        self.model = model

        # 准备数据集
        train_dataset = CustomDataset(self.splits["train"])
        eval_dataset = CustomDataset(self.splits["test"]) if "test" in self.splits else None

        # 构建输出目录
        output_dir = os.path.join(self.save_dir, self.experiment_tag)
        os.makedirs(output_dir, exist_ok=True)

        # 默认 TrainingArguments 参数
        default_args = {
            "output_dir": output_dir,
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "num_train_epochs": self.epochs,
            "logging_dir": os.path.join(output_dir, "logs"),
            "logging_steps": 10,
            "save_strategy": "steps",
            "save_steps": 500,
            "eval_strategy": "steps" if eval_dataset else "no",
            "eval_steps": 500 if eval_dataset else None,
            "save_total_limit": 3,
            "load_best_model_at_end": False,  # GAN训练不适合用best model
            "report_to": "none",
            "remove_unused_columns": False,  # 重要！保持原始数据格式
            "dataloader_num_workers": 0,
            "optim": "adamw_torch",  # 使用AdamW (支持parameter groups)
        }

        # 合并用户自定义的 hf_settings
        training_args_dict = {**default_args, **self.hf_settings}

        # 创建TrainingArguments
        training_args = TrainingArguments(**training_args_dict)

        # 准备 Trainer 参数
        base_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "data_collator": CustomCollator(),  # 保持batch原始格式
        }

        # 合并额外参数
        if trainer_kwargs:
            base_kwargs.update(trainer_kwargs)

        # 添加自定义callbacks
        callbacks = base_kwargs.get("callbacks", [])

        # 进度跟踪callback（更新temperature等）
        callbacks.append(ProgressTrackingCallback(model, self.logger))

        # 日志callback
        if self.logger:
            callbacks.append(CustomLoggerCallback(self.logger))

        base_kwargs["callbacks"] = callbacks

        # 创建 Trainer
        trainer_cls = trainer_class or VTPTrainer
        self.hf_trainer = trainer_cls(**base_kwargs)

        if self.logger:
            self.logger.info(f"✓ HuggingFace Trainer 已构建")
            self.logger.info(f"  - Trainer 类: {trainer_cls.__name__}")
            self.logger.info(f"  - 输出目录: {output_dir}")
            self.logger.info(f"  - 训练集大小: {len(train_dataset)}")
            if eval_dataset:
                self.logger.info(f"  - 评估集大小: {len(eval_dataset)}")
            if training_args.fsdp:
                self.logger.info(f"  - FSDP策略: {training_args.fsdp}")

        return self  # 支持链式调用

    def train(self):
        """启动训练

        Returns:
            train_result: HF Trainer 的训练结果
        """
        if self.hf_trainer is None:
            raise RuntimeError("Trainer 未构建！请先调用 build_trainer()")

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("开始训练（使用 HuggingFace Trainer + FSDP）")
            self.logger.info("=" * 60)

        # 训练
        train_result = self.hf_trainer.train()

        # 保存模型
        self.hf_trainer.save_model()

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("训练完成！")
            self.logger.info("=" * 60)

        return train_result

    def evaluate(self):
        """评估模型

        Returns:
            eval_result: HF Trainer 的评估结果
        """
        if self.hf_trainer is None:
            raise RuntimeError("Trainer 未构建！请先调用 build_trainer()")

        if "test" not in self.splits:
            if self.logger:
                self.logger.info("测试集不存在，跳过评估")
            return {}

        eval_result = self.hf_trainer.evaluate()

        if self.logger:
            self.logger.info(f"评估结果: {eval_result}")

        return eval_result

    def run(self):
        """训练 + 评估（兼容旧接口）

        Returns:
            dict: 包含 train_result 和 eval_result
        """
        if self.hf_trainer is None:
            raise RuntimeError("Trainer 未构建！请先调用 build_trainer()")

        train_result = self.train()
        eval_result = self.evaluate() if "test" in self.splits else {}

        return {
            "score": eval_result.get("eval_loss", 0),
            "train_result": train_result,
            "eval_result": eval_result,
        }

    def get_trainer(self) -> Trainer:
        """获取底层的 HF Trainer 实例

        Returns:
            HF Trainer 实例
        """
        if self.hf_trainer is None:
            raise RuntimeError("Trainer 未构建！请先调用 build_trainer()")
        return self.hf_trainer

    def get_dataset_bundle(self) -> Dict[str, Any]:
        """获取数据集 bundle

        Returns:
            dataset_bundle
        """
        return self.dataset_bundle
