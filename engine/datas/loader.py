"""数据集加载入口 (registry 机制)。

当前支持: 1) MME (键: 'vqa-mme')

配置关键字段 (Config.dataset_settings):
    单数据集模式:
        name: str
            注册表中的数据集键，例如 'vqa-mme'。
        split: Dict[str, float|int]
            目标 split 及其大小。float 表示总样本比例 (0<val<=1)，int 表示绝对数量 (>=0)。

    多数据集模式:
        datasets: List[Dict]
            多个数据集配置列表，每个元素包含:
                name: str - 数据集名称
                split: Dict - 同单数据集模式
                其他数据集特定配置...

    通用配置:
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


def _create_single_dataset_config(base_config: dict, dataset_cfg: Dict[str, Any]) -> dict:
    """为单个数据集创建完整的配置对象

    Args:
        base_config: 原始完整配置
        dataset_cfg: 单个数据集的配置 (包含 name, split 等)

    Returns:
        可用于加载单个数据集的配置对象
    """
    import copy
    # 深拷贝避免修改原始配置
    config = copy.deepcopy(base_config)

    # 用单个数据集的配置替换 dataset_settings
    new_ds_settings = {
        'name': dataset_cfg['name'],
        'split': dataset_cfg.get('split', {}),
        'dataset_name': dataset_cfg['name'],  # 用于日志显示
    }

    # 继承基础配置中的通用设置
    base_ds_settings = base_config.get('dataset_settings', {})
    for key in ['fast_load_no_random', 'category_priority']:
        if key in base_ds_settings and key not in dataset_cfg:
            new_ds_settings[key] = base_ds_settings[key]

    # 合并数据集特定配置（覆盖通用配置）
    for key, value in dataset_cfg.items():
        if key not in ['name']:  # name 已处理
            new_ds_settings[key] = value

    config['dataset_settings'] = new_ds_settings
    return config


def load_multi_datasets(config: dict) -> Dict[str, Any]:
    """加载多个数据集并返回混合训练集和各数据集的测试集

    Args:
        config: 包含 dataset_settings.datasets 的配置对象

    Returns:
        {
            'train_dataset': MixedDataset,          # 混合训练集
            'test_datasets': {name: dataset},       # 各数据集的测试集
            'judges': {name: judge},                # 各数据集的评估函数
            'aggregate_judges': {name: agg_judge},  # 各数据集的聚合评估函数（如有）
            'metas': {name: meta},                  # 各数据集的元信息
            'dataset_names': [name1, name2, ...],   # 数据集名称列表
        }
    """
    from .mixed import MixedDataset

    datasets_cfg = config['dataset_settings']['datasets']
    logger = getattr(config, 'logger', None)

    train_datasets: Dict[str, Any] = {}
    test_datasets: Dict[str, Any] = {}
    judges: Dict[str, Any] = {}
    aggregate_judges: Dict[str, Any] = {}
    metas: Dict[str, Any] = {}
    dataset_names: List[str] = []

    for ds_cfg in datasets_cfg:
        ds_name = ds_cfg['name']
        if ds_name not in DATASET_REGISTRY:
            raise KeyError(f"Dataset '{ds_name}' is not registered. 已注册: {list_datasets()}")

        dataset_names.append(ds_name)

        # 为该数据集创建独立配置
        single_config = _create_single_dataset_config(config, ds_cfg)

        # 加载数据集
        dataset_cls = DATASET_REGISTRY[ds_name]
        bundle = dataset_cls(config=single_config).get()

        splits = bundle['splits']
        metas[ds_name] = bundle['meta']
        judges[ds_name] = bundle['judge']

        # 聚合评估函数（如 MME）
        if 'aggregate_judge' in bundle:
            aggregate_judges[ds_name] = bundle['aggregate_judge']

        # 收集训练集
        if 'train' in splits:
            train_ds = splits['train']
            if len(train_ds) > 0:
                train_datasets[ds_name] = train_ds
                if logger:
                    logger.info(f"[MultiDataset] {ds_name} train: {len(train_ds)} samples")

        # 收集测试集
        if 'test' in splits:
            test_ds = splits['test']
            if len(test_ds) > 0:
                test_datasets[ds_name] = test_ds
                if logger:
                    logger.info(f"[MultiDataset] {ds_name} test: {len(test_ds)} samples")

    # 创建混合训练集
    if not train_datasets:
        raise ValueError("No training data found in any dataset")

    mixed_train = MixedDataset(train_datasets)
    if logger:
        logger.info(f"[MultiDataset] Total train samples: {len(mixed_train)}")
        logger.info(f"[MultiDataset] Dataset sizes: {mixed_train.get_dataset_sizes()}")

    return {
        'train_dataset': mixed_train,
        'test_datasets': test_datasets,
        'judges': judges,
        'aggregate_judges': aggregate_judges,
        'metas': metas,
        'dataset_names': dataset_names,
    }


def is_multi_dataset_mode(config: dict) -> bool:
    """检查是否为多数据集模式

    Args:
        config: 配置对象

    Returns:
        True 如果配置了 datasets 列表，否则 False
    """
    ds_settings = config.get('dataset_settings', {})
    return 'datasets' in ds_settings and isinstance(ds_settings['datasets'], list)


