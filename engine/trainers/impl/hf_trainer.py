"""HuggingFace Trainer 包装器。

提供简洁、直观的接口，专为 HuggingFace Trainer 设计。

使用方式:
    在配置文件中设置:
    trainer_settings:
      type: "dl"
      name: "hf-trainer"
      dl_settings:
        batch_size: 4
        epochs: 3
      hf_settings:  # 直接映射到 TrainingArguments
        gradient_accumulation_steps: 4
        bf16: true
        learning_rate: 5.0e-5
        warmup_ratio: 0.1
        ...
"""

from typing import Any, Dict, Optional
import os
import torch
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.utils.data import Dataset


class CustomLoggerCallback(TrainerCallback):
    """自定义回调：将 HF Trainer 的日志输出到自定义 logger"""

    def __init__(self, logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        """当 HF Trainer 记录日志时触发"""
        if logs is None:
            return

        # 过滤出评估相关的指标
        eval_metrics = {k: v for k, v in logs.items() if k.startswith("eval_")}

        if eval_metrics and self.logger:
            # 输出到自定义 logger
            self.logger.info("=" * 60)
            self.logger.info("评估结果:")
            for key, value in eval_metrics.items():
                if isinstance(value, float):
                    self.logger.info(f"  {key}: {value:.4f}")
                else:
                    self.logger.info(f"  {key}: {value}")
            self.logger.info("=" * 60)


class CustomDataset(Dataset):
    """适配器：将我们的数据格式转换为 HuggingFace Dataset 格式"""

    def __init__(self, samples):
        """
        Args:
            samples: 可以是 DistillDataset、QADataset 或普通列表
        """
        if hasattr(samples, 'samples'):
            # DistillDataset / QADataset
            self.samples = samples.samples
        elif isinstance(samples, list):
            self.samples = samples
        else:
            # 假设是可迭代对象
            self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class HFTrainerWrapper:
    """HuggingFace Trainer 包装器

    提供简洁的接口，专为 HF Trainer 设计，不强制兼容旧接口。
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
        self.hf_trainer: Optional[Trainer] = None

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
        self.experiment_tag = gs["experiment_tag"] if isinstance(gs, dict) else gs.experiment_tag

        if self.logger:
            self.logger.info("HFTrainerWrapper 初始化完成")
            self.logger.info(f"  - Batch size: {self.batch_size}, Epochs: {self.epochs}")

    # ==================== 核心方法：配置和构建 Trainer ====================

    def build_trainer(
        self,
        model: torch.nn.Module,
        tokenizer: Any = None,
        trainer_class: type = None,
        trainer_kwargs: Dict[str, Any] = None,
    ):
        """构建 HuggingFace Trainer

        Args:
            model: 要训练的模型
            tokenizer: 分词器（可选）
            trainer_class: 自定义 Trainer 类（可选，默认使用标准 Trainer）
            trainer_kwargs: 传递给 Trainer 的额外参数（可选）

        Returns:
            self (支持链式调用)
        """
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
            "logging_steps": 100,
            "save_strategy": "epoch",
            "eval_strategy": "epoch" if eval_dataset else "no",  # 注意：eval_strategy 不是 evaluation_strategy
            "save_total_limit": 3,
            "load_best_model_at_end": True if eval_dataset else False,
            "report_to": "none",
            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
        }

        # 合并用户自定义的 hf_settings
        training_args_dict = {**default_args, **self.hf_settings}
        training_args = TrainingArguments(**training_args_dict)

        # 自定义 data collator
        def custom_collate_fn(batch):
            """保持原始数据格式"""
            return batch

        # 准备 Trainer 参数
        base_kwargs = {
            "model": model,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "data_collator": custom_collate_fn,
        }

        # 合并额外参数
        if trainer_kwargs:
            base_kwargs.update(trainer_kwargs)

        # 添加自定义日志回调
        callbacks = base_kwargs.get("callbacks", [])
        if self.logger:
            callbacks.append(CustomLoggerCallback(self.logger))
        base_kwargs["callbacks"] = callbacks

        # 创建 Trainer
        trainer_cls = trainer_class or Trainer
        self.hf_trainer = trainer_cls(**base_kwargs)

        if self.logger:
            self.logger.info(f"HuggingFace Trainer 已构建")
            self.logger.info(f"  - Trainer 类: {trainer_cls.__name__}")
            self.logger.info(f"  - 输出目录: {output_dir}")
            self.logger.info(f"  - 训练集大小: {len(train_dataset)}")
            if eval_dataset:
                self.logger.info(f"  - 评估集大小: {len(eval_dataset)}")

        return self  # 支持链式调用

    # ==================== 训练和评估 ====================

    def train(self):
        """启动训练

        Returns:
            train_result: HF Trainer 的训练结果
        """
        if self.hf_trainer is None:
            raise RuntimeError("Trainer 未构建！请先调用 build_trainer()")

        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("开始训练（使用 HuggingFace Trainer）")
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
            "score": eval_result.get("eval_accuracy", eval_result.get("eval_loss", 0)),
            "train_result": train_result,
            "eval_result": eval_result,
        }

    # ==================== 辅助方法 ====================

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
