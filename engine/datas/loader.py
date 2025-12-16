"""数据集加载入口 (registry 机制)。

当前支持: 1) MME (键: 'vqa-mme')

配置关键字段 (Config.dataset_settings):
    name: str
        注册表中的数据集键，例如 'vqa-mme'。
    split: Dict[str, float|int]
        目标 split 及其大小。float 表示总样本比例 (0<val<=1)，int 表示绝对数量 (>=0)。
        计算逻辑在具体 Preparer 中完成；不足时会截断。
    category_priority: Dict
        类别优先级配置，包含:
            'enable': bool
                是否启用类别优先级分配。
            'values': List[Dict[str,str]]
                按优先级顺序的类别分配模式列表。
                形式: [ {split: mode}, {split: mode}, ... ]，mode 支持:
                    'mean':   尽量均衡该 split 的各类别数量。
                    'origin': 按当前剩余样本的类别比例进行分配。
                未出现的 split 默认使用 'origin'。重复 split 的后续项被忽略。
    fast_load_no_random: bool
        是否以快速模式加载数据集（不随机打乱）。

扩展一个新数据集步骤:
    1. 在 `datas/impl/` 下创建实现 (例如 mydataset.py) 并提供类似 Preparer 接口 (构造接受 config, 提供 get() 返回 {split: Dataset}).
    2. 在这里导入实现中的 Preparer 类。
    3. 将名称与类加入 DATASET_REGISTRY。
    4. 在 Config.dataset_settings 中设置 name 与相关自定义字段。

注意: 这里不做 try/except 包装，错误将直接抛出以便快速发现配置问题。
"""

from typing import Dict, Type, Any, List, Optional
from .impl.mme import MMEPreparer
from .impl.vqa_v2 import VQAV2Preparer
from .impl.pope import POPEPreparer
from .impl.mmb import MMBenchPreparer
from .impl.scienceqa import ScienceQAPreparer
from .impl.gqa import GQAPreparer
from .impl.seed_bench import SEEDBenchPreparer

DATASET_REGISTRY: Dict[str, Type[Any]] = {
    "vqa-mme": MMEPreparer,
    "vqa-vqav2": VQAV2Preparer,
    "vqa-pope": POPEPreparer,
    "vqa-mmb": MMBenchPreparer,
    "vqa-sqa": ScienceQAPreparer,
    "vqa-gqa": GQAPreparer,
    "vqa-seed": SEEDBenchPreparer,
}


def load_dataset(config: Optional[dict] = None) -> Any:
    """根据名称加载并实例化数据集，返回 bundle dict: {'splits':..., 'meta':..., 'judge': callable}。

    judge: 数据集相关答案判定函数，支持:
      - 单条: judge(pred_str, ref_str, sample) -> {correct,total,accuracy}
      - 批量: judge([pred...], [ref...]) -> {correct,total,accuracy}
    """
    name = config["dataset_settings"]["name"] # type: ignore
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' is not registered. 已注册: {list_datasets()}")
    dataset_cls = DATASET_REGISTRY[name]
    return dataset_cls(config=config).get()


def list_datasets() -> List[str]:
	"""返回当前已注册的数据集名称列表。"""
	return list(DATASET_REGISTRY.keys())

