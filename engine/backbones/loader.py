"""Backbone 加载入口 (registry 机制，支持按类型分层)。

配置关键字段 (Config.backbone_settings):
    type: str
        Backbone 类型，例如 'mllm'。
    name: str
        注册表中的 backbone 键，例如 'qwen-2.5-3b'。

当前支持类型:
    - 'mllm': 多模态大语言模型

扩展一个新 Backbone 步骤:
    1. 在 `backbones/base/` 下创建基类文件 (例如 base/mllm.py) 定义该类型的基类。
    2. 在 `backbones/impl/类型名/` 下创建实现目录 (例如 impl/mllm/)。
    3. 在实现目录下创建具体模型文件 (例如 impl/mllm/qwen.py) 继承基类。
    4. 在这里导入实现中的类。
    5. 将类型与名称加入 BACKBONE_REGISTRY（按类型分层）。
    6. 在 Config.backbone_settings 中设置 type 与 name。

注意: 这里不做 try/except 包装，错误将直接抛出以便快速发现配置问题。
"""

from typing import Dict, Type, Any, List, Optional

# 导入各类型的实现
# MLLM 类型
from .impl.mllm.qwen import QwenMLLMBackbone
from .impl.mllm.llava import LLaVAMLLMBackbone
from .impl.mllm.llava_next import LLaVANextMLLMBackbone

# 注册表结构: {type: {name: Class}}
BACKBONE_REGISTRY: Dict[str, Dict[str, Type[Any]]] = {
    "mllm": {
        "qwen-2.5-3b": QwenMLLMBackbone,
        "llava-1.5-7b": LLaVAMLLMBackbone,
        "llava-1.6-vicuna-7b": LLaVANextMLLMBackbone,
    },
}


def load_backbone(config: Optional[dict] = None) -> Any:
    """根据类型和名称加载并实例化 Backbone。

    参数:
        config: Config 实例，必须包含 backbone_settings['type'] 和 ['name']

    返回:
        Backbone 实例

    示例:
        backbone = load_backbone(config)
    """
    backbone_cfg = config.backbone_settings
    b_type = backbone_cfg['type']
    name = backbone_cfg['name']

    if b_type not in BACKBONE_REGISTRY:
        raise KeyError(
            f"Backbone type '{b_type}' is not registered. "
            f"已注册类型: {list_backbone_types()}"
        )

    type_registry = BACKBONE_REGISTRY[b_type]
    if name not in type_registry:
        raise KeyError(
            f"Backbone name '{name}' (type='{b_type}') is not registered. "
            f"已注册: {list_backbones(b_type)}"
        )

    backbone_cls = type_registry[name]
    return backbone_cls(config=config)


def list_backbone_types() -> List[str]:
    """返回当前已注册的 Backbone 类型列表。"""
    return list(BACKBONE_REGISTRY.keys())


def list_backbones(b_type: Optional[str] = None) -> List[str]:
    """返回指定类型下已注册的 Backbone 名称列表。
    
    参数:
        b_type: Backbone 类型，如果为 None 则返回所有类型的名称
    """
    if b_type is None:
        # 返回所有类型的所有名称
        all_names = []
        for type_registry in BACKBONE_REGISTRY.values():
            all_names.extend(type_registry.keys())
        return all_names
    
    if b_type not in BACKBONE_REGISTRY:
        return []
    
    return list(BACKBONE_REGISTRY[b_type].keys())

