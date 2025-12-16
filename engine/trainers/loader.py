"""Trainers loader.

根据配置加载并实例化训练器。
"""

from typing import Any, Dict


def load_trainer(config: Any, dataset_bundle: Dict[str, Any] = None):
	"""根据 `config.trainer_settings` 加载训练器。

	目前支持:
	- name: "basic-pytorch" -> `BasicPytorchTrainer` (需要dataset_bundle)
	- name: "basic-tianshou" -> `BasicTianshouTrainer` (不需要dataset_bundle)

	参数:
		config: 配置对象，需包含 `trainer_settings`
		dataset_bundle: 数据集 bundle（可选），监督学习用
	返回:
		训练器实例
	"""
	trainer_settings = getattr(config, "trainer_settings", None)
	if trainer_settings is None:
		# 兼容字典形式（在 configs.loader.load_config 返回字典时）
		trainer_settings = config.get("trainer_settings")

	name = None
	if isinstance(trainer_settings, dict):
		name = trainer_settings.get("name")
	else:
		name = getattr(trainer_settings, "name", None)

	if name is None:
		raise ValueError("trainer_settings.name 未设置，无法加载训练器")

	name = name.lower()

	if name == "basic-pytorch":
		from .impl.basic_pytorch import BasicPytorchTrainer
		if dataset_bundle is None:
			raise ValueError("BasicPytorchTrainer 需要 dataset_bundle 参数")
		return BasicPytorchTrainer(config=config, dataset_bundle=dataset_bundle)

	elif name == "basic-tianshou":
		from .impl.basic_tianshou import BasicTianshouTrainer
		return BasicTianshouTrainer(config=config)

	raise ValueError(f"不支持的训练器名称: {name}")

