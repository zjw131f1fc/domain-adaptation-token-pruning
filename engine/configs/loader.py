import os
import sys
# --- Environment Setup ---
# 自动检测工作目录：从当前文件位置向上找到项目根目录
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.abspath(os.path.join(CURRENT_FILE_DIR, '../..'))
sys.path.insert(0, WORKSPACE_DIR)
import sys
import logging
import json
import re
import random
from datetime import datetime
from typing import Optional, Dict, Any
from copy import deepcopy
import yaml

# Lazy import torch and numpy to avoid early CUDA initialization
# import numpy as np
# import torch

# ==================== AttrDict: 支持属性访问的字典 ====================
class AttrDict(dict):
    """支持属性访问的字典类
    
    特性:
    - 支持 obj.key 和 obj["key"] 两种访问方式
    - 支持嵌套字典递归转换
    - 保持字典的所有方法
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 递归转换嵌套字典
        for key, value in self.items():
            if isinstance(value, dict) and not isinstance(value, AttrDict):
                self[key] = AttrDict(value)
            elif isinstance(value, list):
                self[key] = [AttrDict(item) if isinstance(item, dict) else item for item in value]
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'AttrDict' object has no attribute '{key}'")

# ==================== 全局默认配置 ====================
# 最小化默认配置，具体参数由 YAML 配置文件覆盖
DEFAULT_CONFIG = {
    "global_settings": {
        "seed": 42,
        "device": "cuda",
        "dtype": "float16",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
        "dataset_cache_dir": "/root/autodl-tmp/dataset_cache",
        "save_dir": "./outputs/checkpoints",
        "log_dir": "./outputs/logs",
        "study_name": ""
    },
    "trainer_settings": {
        "dl_settings": {
            "epochs": 1,
            "batch_size": 4,
            "optimizers": {
                "layer_pruners": {"type": "adam", "lr": 1e-4},
                "discriminator": {"type": "adam", "lr": 1.5e-4}
            },
            "print_loss_every_batches": 50,
            "eval_every_batches": 1000,
            "eval_max_samples": 500,
            "save_every_batches": 3000,
            "grad_clip_max_norm": 1.0,
        }
    },
    "dataset_settings": {
        "name": "vqa-vqav2",
        "split": {"train": 1000, "test": 500},
        "fast_load_no_random": False
    },
    "config_settings": {
        "enable_dict_overrides": True,
        "enable_yaml_overrides": True,
        "log_config_on_load": True
    },
    "backbone_settings": {
        "name": "llava-1.5-7b",
    },
    "method_settings": {
        # 剪枝层配置
        "pruning_layers": [4, 14, 24],
        # Pruner 配置 (CrossAttentionPruner)
        "pruner_d_internal": 128,
        "pruner_n_heads": 4,
        "pruner_dropout": 0.1,
        # Discriminator 配置
        "disc_d_d": 256,
        "disc_dropout": 0.1,
        "disc_use_spectral_norm": False,
        # 剪枝目标
        "target_token_num": 144,
        # Temperature 配置
        "temperature": 1.0,
        "temperature_min": 0.5,
        "temperature_anneal_rate": 0.4,
        # 损失权重
        "task_loss_weight": 1.0,
        "adv_loss_weight": 0.5,
        "sparsity_weight": 0.2,
        # 动态权重调度
        "loss_weight_warmup_ratio": 0.0,
        "task_loss_weight_start": 1.0,
        "adv_loss_weight_start": 0.5,
        # Sparsity warmup
        "sparsity_warmup_enable": False,
        "sparsity_warmup_ratio": 0.2,
        "sparsity_weight_max": 0.5,
    },
    "evaluation_settings": {
        "eval_mode": ["origin", "hard"],
    }
}

# ==================== 自动命名使用的字段列表 ====================
AUTO_NAME_FIELDS = [
    ("backbone_settings", "name"),
    ("dataset_settings", "name"),
    ("dataset_settings", "split"),
]

# ==================== 模型维度映射 ====================
# 根据 backbone 名称自动设置 hidden_dim 和 vision_dim
BACKBONE_DIM_MAP = {
    "qwen2-vl-2b": {"hidden_dim": 1536, "vision_dim": 1536, "model_type": "qwen2_vl"},
    "qwen2-vl-7b": {"hidden_dim": 3584, "vision_dim": 3584, "model_type": "qwen2_vl"},
    "llava-1.5-7b": {"hidden_dim": 4096, "vision_dim": 1024, "model_type": "llava"},
    "llava-1.5-13b": {"hidden_dim": 5120, "vision_dim": 1024, "model_type": "llava"},
}


# ==================== 辅助函数 ====================

def _merge_dict(base: Dict[str, Any], patch: Dict[str, Any]):
    """递归合并字典"""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge_dict(base[k], v)
        else:
            base[k] = v


def _normalize_value(value: Any, original_value: Any) -> Any:
    """根据原始值类型自动转换新值的类型"""
    # 如果原始值是None，保持新值不变
    if original_value is None:
        return value
    
    # 如果新值已经是正确类型，直接返回
    if type(value) == type(original_value):
        return value
    
    # 字符串到布尔值
    if isinstance(original_value, bool) and isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    
    # 字符串到整数
    if isinstance(original_value, int) and isinstance(value, str):
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        # 尝试先转float再转int
        try:
            return int(float(value.strip()))
        except ValueError:
            return value
    
    # 字符串到浮点数
    if isinstance(original_value, float) and isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return value
    
    # 字符串到列表（逗号分隔）
    if isinstance(original_value, list) and isinstance(value, str):
        if ',' in value:
            return [v.strip() for v in value.split(',')]
        else:
            return [value]
    
    # 处理字典中的值（如split）
    if isinstance(original_value, dict) and isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k in original_value:
                result[k] = _normalize_value(v, original_value[k])
            else:
                # 新字段，尝试自动识别类型
                if isinstance(v, str):
                    if v == '-1':
                        result[k] = -1
                    elif v == 'all':
                        result[k] = 'all'
                    elif v.isdigit():
                        result[k] = int(v)
                    else:
                        try:
                            result[k] = float(v)
                        except ValueError:
                            result[k] = v
                else:
                    result[k] = v
        return result
    
    return value


def _auto_normalize_types(config: Dict[str, Any], default_config: Dict[str, Any]):
    """自动识别并规范化配置中所有字段的类型"""
    for setting_name in config:
        if setting_name not in default_config:
            continue
        
        setting_dict = config[setting_name]
        default_setting_dict = default_config[setting_name]
        
        if not isinstance(setting_dict, dict):
            continue
        
        for field_name, value in list(setting_dict.items()):
            if field_name in default_setting_dict:
                original_value = default_setting_dict[field_name]
                setting_dict[field_name] = _normalize_value(value, original_value)


def _generate_experiment_tag(config: Dict[str, Any]) -> str:
    """生成实验标签，格式：[study_name_]timestamp_dataset_backbone_random
    
    示例: 20251126-1941_vqa-mme_qwen253b_a7f2
    或: baseline_20251126-1941_vqa-mme_qwen253b_a7f2
    """
    # 1. 时间戳（年月日-时分）- 放在最前方便排序
    timestamp = datetime.now().strftime('%Y%m%d-%H%M')
    
    # 2. 数据集名称（简化）
    dataset_name = config["dataset_settings"]["name"].lower().replace(" ", "-")
    
    # 3. Backbone名称（简化，移除点和连字符）
    backbone_name = config["backbone_settings"]["name"].lower()
    backbone_short = backbone_name.replace(".", "").replace("-", "")
    
    # 4. 4位随机十六进制数（保证不重复）
    random_suffix = format(random.randint(0, 0xFFFF), '04x')
    
    # 组合基础标签
    base_tag = f"{timestamp}_{dataset_name}_{backbone_short}_{random_suffix}"
    
    # 5. 如果指定了study_name，添加到前面
    study_name = config["global_settings"].get("study_name", "").strip()
    if study_name:
        tag = f"{study_name}_{base_tag}"
    else:
        tag = base_tag
    
    return tag



def _setup_logger(config: Dict[str, Any]) -> logging.Logger:
    """初始化logger"""
    experiment_tag = config["global_settings"]["experiment_tag"]
    log_dir = config["global_settings"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{experiment_tag}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # File handler - 默认启用文件日志
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def _log_config(logger: logging.Logger, config: Dict[str, Any], timestamp: str):
    """以易读的格式打印配置信息"""
    lines = []
    lines.append("="*80)
    lines.append("Configuration Loaded")
    lines.append(f"Timestamp: {timestamp}")
    lines.append("="*80)
    
    # 自动识别所有配置分组并格式化
    for section_key in config.keys():
        # 跳过非配置项（如logger、timestamp）
        if section_key in ("logger", "timestamp"):
            continue
        
        # 将下划线命名转为标题格式
        section_title = " ".join(word.capitalize() for word in section_key.split("_"))
        
        lines.append("")
        lines.append(f"[{section_title}]")
        lines.append("-" * 80)
        
        section_data = config[section_key]
        _format_section(section_data, lines, indent=2)
    
    lines.append("="*80)
    logger.info("\n" + "\n".join(lines))


def _format_section(data: Any, lines: list, indent: int = 0):
    """递归格式化配置项"""
    indent_str = " " * indent
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                _format_section(value, lines, indent + 2)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], dict):
                    lines.append(f"{indent_str}{key}:")
                    for i, item in enumerate(value):
                        lines.append(f"{indent_str}  [{i}]:")
                        _format_section(item, lines, indent + 4)
                else:
                    lines.append(f"{indent_str}{key}: {value}")
            else:
                lines.append(f"{indent_str}{key}: {value}")
    else:
        lines.append(f"{indent_str}{data}")


def set_random_seed(seed: int):
    """设置所有随机数生成器的种子以确保实验可重复性"""
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 关闭 deterministic 模式以加速训练（结果可能不完全可复现）
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_config(override_dict: Optional[Dict[str, Any]] = None, 
                override_file: Optional[str] = None,
                skip_auto_paths: bool = False) -> Dict[str, Any]:
    """加载配置
    
    参数:
        override_dict: 参数覆盖字典（可选）
        override_file: 参数覆盖文件路径（可选），支持yaml格式
        skip_auto_paths: 是否跳过自动生成路径（用于子任务，由管理器提供路径）
    
    返回:
        config: 配置字典，包含logger属性
    """
    # 1. 创建默认配置的深拷贝
    config = deepcopy(DEFAULT_CONFIG)
    
    # 2. 应用 YAML 文件覆盖（如果启用）- 先应用文件
    if override_file:
        # 先检查 override_dict 中的 config_settings（如果有）
        if override_dict and "config_settings" in override_dict:
            _merge_dict(config["config_settings"], override_dict["config_settings"])
        
        if not config["config_settings"]["enable_yaml_overrides"]:
            print("[Warning] YAML文件覆盖已禁用，忽略 override_file 参数")
        elif not os.path.isfile(override_file):
            print(f"[Warning] 配置文件未找到: {override_file}")
        else:
            try:
                with open(override_file, 'r', encoding='utf-8') as f:
                    yaml_data = yaml.safe_load(f) or {}
                if isinstance(yaml_data, dict):
                    _merge_dict(config, yaml_data)
                    _auto_normalize_types(config, DEFAULT_CONFIG)
                    print(f"[Info] 成功加载配置文件: {override_file}")
                else:
                    print(f"[Warning] 配置文件格式错误: {override_file}")
            except Exception as e:
                print(f"[Error] 读取配置文件失败: {override_file}, 错误: {e}")
    
    # 3. 应用字典覆盖（如果启用）- 后应用字典，优先级更高
    if override_dict:
        # 先应用 config_settings 的覆盖（如果有）
        if "config_settings" in override_dict:
            _merge_dict(config["config_settings"], override_dict["config_settings"])
        
        # 然后根据设置决定是否应用其他覆盖
        if not config["config_settings"]["enable_dict_overrides"]:
            print("[Warning] 字典覆盖已禁用，忽略 override_dict 中的非 config_settings 参数")
        else:
            _merge_dict(config, override_dict)
    
    # 4. 自动识别并规范化类型
    _auto_normalize_types(config, DEFAULT_CONFIG)
    
    # 5. 根据 backbone 名称自动设置 hidden_dim 和 vision_dim
    backbone_name = config["backbone_settings"]["name"]
    if backbone_name not in BACKBONE_DIM_MAP:
        raise ValueError(f"未知的 backbone 名称: {backbone_name}，支持的模型: {list(BACKBONE_DIM_MAP.keys())}")
    dim_config = BACKBONE_DIM_MAP[backbone_name]
    # 将维度信息和模型类型直接存到 backbone_settings
    config["backbone_settings"]["hidden_dim"] = dim_config["hidden_dim"]
    config["backbone_settings"]["vision_dim"] = dim_config["vision_dim"]
    config["backbone_settings"]["model_type"] = dim_config.get("model_type", "llava")
    
    # 6. 生成 experiment_tag 和路径（如果不跳过）
    if not skip_auto_paths:
        experiment_tag = _generate_experiment_tag(config)
        config["global_settings"]["experiment_tag"] = experiment_tag
        
        # 6.1. 创建独立的任务目录（所有任务都使用独立目录）
        task_dir = os.path.join("./outputs/tasks", experiment_tag)
        os.makedirs(task_dir, exist_ok=True)
        
        # 更新 save_dir 和 log_dir 指向任务目录
        config["global_settings"]["save_dir"] = os.path.join(task_dir, "checkpoints")
        config["global_settings"]["log_dir"] = os.path.join(task_dir, "logs")
        config["global_settings"]["task_dir"] = task_dir
    else:
        # 跳过自动路径生成，使用 override_dict 中提供的路径
        # 确保 experiment_tag 存在（用于日志文件名）
        if "experiment_tag" not in config["global_settings"] or not config["global_settings"]["experiment_tag"]:
            # 如果没有提供，使用 study_name 作为 experiment_tag
            config["global_settings"]["experiment_tag"] = config["global_settings"].get("study_name", "subtask")
        
        # 确保必要的目录存在
        if "save_dir" in config["global_settings"]:
            os.makedirs(config["global_settings"]["save_dir"], exist_ok=True)
        if "log_dir" in config["global_settings"]:
            os.makedirs(config["global_settings"]["log_dir"], exist_ok=True)
    
    # 7. 设置环境变量
    # 设置CUDA_VISIBLE_DEVICES（如果Manager提供了）
    if "cuda_visible_devices" in config["global_settings"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = config["global_settings"]["cuda_visible_devices"]
    
    # 提前设置 CUDA_VISIBLE_DEVICES 以确保在任何 CUDA 初始化之前生效
    if "manager_settings" in config:
        ms = config["manager_settings"]
        available_gpus = ms.get("available_gpus", [])
        mode = ms.get("mode")
        
        # 确保 available_gpus 是列表
        if isinstance(available_gpus, str):
             available_gpus = [int(x.strip()) for x in available_gpus.split(",") if x.strip()]
        
        if available_gpus:
            gpu_str = ""
            if mode == "direct":
                # Direct模式：只设置分配给当前任务的GPU
                gpus_per_subtask = ms.get("gpus_per_subtask", 1)
                assigned_gpus = available_gpus[:gpus_per_subtask]
                gpu_str = ",".join(map(str, assigned_gpus))
            else:
                # 其他模式：设置所有可用GPU，防止主进程占用未分配的GPU
                gpu_str = ",".join(map(str, available_gpus))
            
            if gpu_str:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
                print(f"[ConfigLoader] Set CUDA_VISIBLE_DEVICES={gpu_str} (mode={mode})")

    # 设置PYTORCH_CUDA_ALLOC_CONF
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = config["global_settings"]["pytorch_cuda_alloc_conf"]
    
    # 8. 初始化logger
    logger = _setup_logger(config)
    config["logger"] = logger
    config["timestamp"] = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 9. 设置随机种子
    seed = config["global_settings"]["seed"]
    logger.info(f"Setting random seed to: {seed}")
    set_random_seed(seed)
    
    # 10. 打印配置（如果启用）
    if config["config_settings"]["log_config_on_load"]:
        _log_config(logger, config, config["timestamp"])
    
    # 11. 转换为AttrDict支持属性访问
    config = AttrDict(config)
    
    return config
