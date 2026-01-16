"""自定义 HuggingFace Trainer - 支持 GAN 训练和并行计算

继承 HuggingFace Trainer，利用其：
- 分布式训练（DDP）
- 混合精度训练（fp16/bf16）
- 梯度累积
- 自动 checkpoint
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput
from typing import Dict, Any, Optional, List, Union, Tuple
import os

from .training_step import train_step
from .utils import remove_hooks


class PruningTrainer(Trainer):
    """支持 GAN 训练的 HuggingFace Trainer

    特点:
    - 继承 HuggingFace Trainer，支持 FSDP 并行
    - 多优化器支持（layer_pruners 和 discriminator）
    - 使用原有的 train_step 逻辑
    """

    def __init__(
        self,
        model: nn.Module,  # PruningModelWrapper，包含 backbone、layer_pruners、discriminator
        args: TrainingArguments,
        config: Dict[str, Any],
        train_dataset=None,
        eval_dataset=None,
        data_collator=None,
        judge_fn=None,
        **kwargs
    ):
        # 保存额外组件
        self.full_config = config
        self.judge_fn = judge_fn

        # 调用父类初始化
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            **kwargs
        )

        # 创建判别器优化器
        self._create_discriminator_optimizer()

        # 训练状态
        self._global_step = 0

    @property
    def backbone(self):
        """获取 backbone（兼容 FSDP wrapped model）"""
        if hasattr(self.model, "module"):
            return self.model.module.backbone
        return self.model.backbone

    @property
    def layer_pruners(self):
        """获取 layer_pruners（兼容 FSDP wrapped model）"""
        if hasattr(self.model, "module"):
            return self.model.module.layer_pruners
        return self.model.layer_pruners

    @property
    def discriminator(self):
        """获取 discriminator（兼容 FSDP wrapped model）"""
        if hasattr(self.model, "module"):
            return self.model.module.discriminator
        return self.model.discriminator

    @property
    def models(self):
        """模型字典（用于 train_step）"""
        return {
            "backbone": self.backbone,
            "layer_pruners": self.layer_pruners,
            "discriminator": self.discriminator,
        }

    def _create_discriminator_optimizer(self):
        """创建判别器的独立优化器"""
        trainer_cfg = self.full_config.get("trainer_settings", {}).get("dl_settings", {})
        opt_cfg = trainer_cfg.get("optimizers", {}).get("discriminator", {})

        lr = opt_cfg.get("lr", 1.5e-4)

        self.disc_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

    def create_optimizer(self):
        """创建剪枝器优化器 (覆盖父类方法)"""
        trainer_cfg = self.full_config.get("trainer_settings", {}).get("dl_settings", {})
        opt_cfg = trainer_cfg.get("optimizers", {}).get("layer_pruners", {})

        lr = opt_cfg.get("lr", 1e-4)

        # 只优化 layer_pruners 参数
        self.optimizer = torch.optim.AdamW(
            self.layer_pruners.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        return self.optimizer

    def get_train_dataloader(self) -> DataLoader:
        """创建训练 DataLoader (覆盖父类方法)

        支持 DDP 分布式训练，使用 DistributedSampler
        """
        def collate_fn(batch):
            return batch  # 保持 list，不强制张量化

        # 检查是否是分布式训练
        if self.args.world_size > 1:
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                shuffle=True,
            )
            shuffle = False  # 使用 sampler 时不能 shuffle
        else:
            sampler = None
            shuffle = True

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """创建评估 DataLoader (覆盖父类方法)"""
        def collate_fn(batch):
            return batch

        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("No eval dataset provided")

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def training_step(
        self,
        model: nn.Module,
        inputs: List[Dict[str, Any]],
        num_items_in_batch: Optional[int] = None
    ) -> torch.Tensor:
        """执行单步训练 (覆盖父类方法)

        使用原有的 train_step 逻辑，但适配 HuggingFace Trainer 的接口
        """
        self._global_step += 1

        # 计算总步数
        total_steps = self.args.max_steps if self.args.max_steps > 0 else (
            len(self.get_train_dataloader()) * self.args.num_train_epochs
        )

        # 获取正确的设备（兼容 DDP）
        device = next(self.model.parameters()).device

        # 构建 info 字典
        info = {
            "config": self.full_config,
            "epoch": int(self.state.epoch) if self.state.epoch else 1,
            "batch": self._global_step,
            "epoch_batch_index": self._global_step,
            "global_batch_index": self._global_step,
            "total_planned_batches": total_steps,
            "models": self.models,
            "persistent_state": {},
        }

        # 调用原有的 train_step
        outputs = train_step(inputs, device, info)

        # === 处理 Discriminator 的 backward 和 step ===
        # Discriminator 使用独立优化器，不走 Trainer 的自动 backward
        disc_losses = outputs.get("discriminator", {})
        if disc_losses:
            disc_loss = sum(v for v in disc_losses.values() if torch.is_tensor(v))
            if torch.is_tensor(disc_loss):
                self.disc_optimizer.zero_grad()
                disc_loss.backward(retain_graph=True)
                # 梯度裁剪（使用原始参数，兼容 DDP）
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self._get_discriminator_params(),
                        self.args.max_grad_norm
                    )
                self.disc_optimizer.step()

        # === 返回 Layer Pruners 的 loss ===
        # Trainer 会自动处理这个 loss 的 backward 和 optimizer.step
        pruner_losses = outputs.get("layer_pruners", {})
        if pruner_losses:
            total_loss = sum(v for v in pruner_losses.values() if torch.is_tensor(v))
        else:
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 记录 metrics
        metrics = outputs.get("metrics", {})
        if metrics:
            self.log(metrics)

        return total_loss

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """计算损失 (覆盖父类方法)

        注意：由于我们重写了 training_step，这个方法在训练时不会被调用
        但在评估时会被调用，所以需要实现
        """
        # 评估时调用 train_step 但不更新参数
        self.model.eval()
        self.discriminator.eval()

        # 获取正确的设备
        device = next(self.model.parameters()).device

        # 构建 info 字典
        total_steps = self.args.max_steps if self.args.max_steps > 0 else 1000
        info = {
            "config": self.full_config,
            "epoch": 1,
            "batch": 0,
            "epoch_batch_index": 0,
            "global_batch_index": 0,
            "total_planned_batches": total_steps,
            "models": self.models,
            "persistent_state": {},
        }

        with torch.no_grad():
            outputs = train_step(inputs, device, info)

        # 计算总 loss
        pruner_losses = outputs.get("layer_pruners", {})
        if pruner_losses:
            total_loss = sum(v for v in pruner_losses.values() if torch.is_tensor(v))
        else:
            total_loss = torch.tensor(0.0, device=device)

        self.model.train()
        self.discriminator.train()

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval"
    ) -> EvalLoopOutput:
        """评估循环 (覆盖父类方法)

        使用 judge_fn 计算准确率指标
        """
        self.model.eval()
        self.discriminator.eval()

        # 如果没有 judge_fn，使用简化评估（只计算 loss）
        if self.judge_fn is None:
            output = super().evaluation_loop(
                dataloader=dataloader,
                description=description,
                prediction_loss_only=True,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix
            )
            self.model.train()
            self.discriminator.train()
            return output

        # 有 judge_fn，进行完整评估
        from .utils import register_multi_layer_hooks_batch, remove_hooks

        all_preds = []
        all_refs = []
        total_loss = 0.0
        num_batches = 0

        method_cfg = self.full_config.get("method_settings", {})
        use_attn_residual = method_cfg.get("use_attn_residual", False)

        for batch in dataloader:
            with torch.no_grad():
                # 预处理
                images = [s["image"] for s in batch]
                questions = [s["question"] for s in batch]
                # 获取参考答案（用于评估）
                refs = []
                for s in batch:
                    if "answers" in s and isinstance(s["answers"], list):
                        refs.append(s["answers"])
                    else:
                        refs.append(s["answer"])

                # 获取 embeddings
                emb_info = self.backbone.preprocess_batch(images, questions, None)
                embeddings = emb_info['embeddings']
                attention_mask = emb_info['attention_mask']
                vision_pos = emb_info['vision_token_positions']

                # 提取 question embeddings
                v_end = vision_pos[1]
                question_embeddings = embeddings[:, v_end+1:, :]

                # 注册 hooks
                pruning_masks = []
                handles = register_multi_layer_hooks_batch(
                    self.backbone,
                    self.model,
                    vision_pos,
                    question_embeddings,
                    mask_collector=pruning_masks,
                    use_attn_residual=use_attn_residual
                )

                try:
                    # 生成预测
                    preds = []
                    for i, (img, q) in enumerate(zip(images, questions)):
                        # 单样本生成（因为 generate 不支持 batch）
                        pred = self.backbone.generate(img, q, max_new_tokens=20)
                        preds.append(pred)
                finally:
                    remove_hooks(handles)

                all_preds.extend(preds)
                all_refs.extend(refs)
                num_batches += 1

        # 使用 judge_fn 计算指标
        judge_result = self.judge_fn(all_preds, all_refs)

        metrics = {
            f"{metric_key_prefix}_accuracy": judge_result["accuracy"],
            f"{metric_key_prefix}_correct": judge_result["correct"],
            f"{metric_key_prefix}_total": judge_result["total"],
        }

        self.log(metrics)
        self.model.train()
        self.discriminator.train()

        # 返回 EvalLoopOutput
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=len(all_preds)
        )

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """保存模型 (覆盖父类方法)"""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 保存剪枝器
        pruner_path = os.path.join(output_dir, "layer_pruners.pt")
        torch.save(self.layer_pruners.state_dict(), pruner_path)

        # 保存判别器
        disc_path = os.path.join(output_dir, "discriminator.pt")
        torch.save(self.discriminator.state_dict(), disc_path)

        # 保存配置
        config_path = os.path.join(output_dir, "config.json")
        import json
        serializable_config = {
            k: v for k, v in self.full_config.items()
            if k not in ["logger"]
        }
        with open(config_path, "w") as f:
            json.dump(serializable_config, f, indent=2, default=str)

        self.log({"save": output_dir})

    def _get_discriminator_params(self):
        """获取 discriminator 的参数"""
        return self.discriminator.parameters()
