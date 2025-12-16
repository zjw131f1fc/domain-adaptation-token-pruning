"""MLLM (多模态大语言模型) Backbone 基类。

定义 MLLM 类型 Backbone 的通用接口和基础功能。
具体实现应放在 impl/mllm.py 中。
"""

from typing import Dict, Any, Optional


class BaseMLLMBackbone:
    """MLLM Backbone 基类。

    子类需要实现具体模型加载和推理方法。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 Backbone。

        参数:
            config: Config 实例，包含 backbone_settings 等配置
        """
        self.config = config
        if config is not None:
            self.backbone_cfg = config.backbone_settings # type: ignore
            self.model_name = self.backbone_cfg['name']
        else:
            self.backbone_cfg = {}
            self.model_name = 'unknown'
        
        # TODO: 实现模型加载逻辑
        self.model = None
        self.processor = None

    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.model_name}')>"

