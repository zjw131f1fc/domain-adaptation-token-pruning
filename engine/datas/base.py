from typing import List, Dict, Any, Tuple
import random


class BsesDataset:
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def to_list(self):
        return self.samples


class BasePreparer:
    """精简版通用父类: 只提供通用操作方法, 不主导流程。

    子类需要完成:
      - 数据加载 (返回单一列表 或 已拆分的 dict)
      - 调用本父类提供的拆分/元信息构建方法
      - 构建 judge 与 report 输出

    保留特性:
      - split 支持: float 比例, int 绝对数量, -1 占位, 'all' 全部
      - 类别优先拆分 (category_priority) + 模式 'mean'/'origin'
      - 字段映射 (field_map)

    去除:
      - 自动流程 prepare()
      - need_manual_split 标志
      - coarse_load 等子类特殊逻辑由子类自行处理
    """

    def __init__(self, config):
        self.config = config
        ds_cfg = self.config.dataset_settings
        self.split_cfg: Dict[str, Any] = ds_cfg["split"] if "split" in ds_cfg else {}
        self.has_category: bool = False
        self.field_map: Dict[str, str] = ds_cfg["field_map"] if "field_map" in ds_cfg else {}
        self.seed: int = self.config.global_settings["seed"]

    # ---- 基础工具 ----
    def apply_field_map(self, samples: List[Dict[str, Any]]) -> Dict[str, Tuple[str, str]]:
        applied: Dict[str, Tuple[str, str]] = {}
        if self.field_map:
            for src, dst in self.field_map.items():
                for s in samples:
                    s[dst] = s[src]
                applied[src] = (src, dst)
        return applied

    def detect_category(self, samples: List[Dict[str, Any]]):
        if not self.has_category and samples:
            if "category" in samples[0]:
                self.has_category = True

    def compute_split_target_sizes(self, total: int) -> Dict[str, int]:
        targets: Dict[str, int] = {}
        for split_name, v in self.split_cfg.items():
            if isinstance(v, float) and 0 < v <= 1:
                targets[split_name] = int(total * v)
            elif isinstance(v, int) and v >= 0:
                targets[split_name] = v if v <= total else total
            elif isinstance(v, int) and v == -1:
                targets[split_name] = 1  # 占位仅需引用一个样本
            elif isinstance(v, str) and v == 'all':
                targets[split_name] = total
            else:
                targets[split_name] = 0
        return targets

    # ---- 拆分场景 1: 单一列表, 需要根据 split_cfg 切分 ----
    def split_from_single(self, samples: List[Dict[str, Any]]) -> Tuple[Dict[str, BsesDataset], List[str]]:
        if self.has_category and "category_priority" in self.config.dataset_settings:
            cat_priority = self.config.dataset_settings["category_priority"]
            if cat_priority["enable"]:
                return self._split_category_priority(samples)
        # 随机顺序拆分
        total = len(samples)
        random.seed(self.seed)
        shuffled = samples[:]
        random.shuffle(shuffled)
        targets = self.compute_split_target_sizes(total)
        result: Dict[str, BsesDataset] = {}
        ptr = 0
        placeholder_splits: List[str] = []
        for name, size in targets.items():
            raw_v = self.split_cfg[name]
            is_placeholder = isinstance(raw_v, int) and raw_v == -1
            is_all = isinstance(raw_v, str) and raw_v == 'all'
            end = ptr + size
            if end > total:
                end = total
            if is_all:
                subset = samples[:]
                placeholder_splits.append(name)
            else:
                subset = shuffled[ptr:end]
                if is_placeholder:
                    placeholder_splits.append(name)
                else:
                    ptr = end
            result[name] = BsesDataset(subset)
        return result, placeholder_splits

    # ---- 拆分场景 2: 已预拆分 dict 输入, 按需求截取数量 ----
    def split_from_presplits(self, presplits: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, BsesDataset], List[str]]:
        placeholder_splits: List[str] = []
        result: Dict[str, BsesDataset] = {}

        # 检查是否需要类别平衡截取
        use_category_balance = (
            self.has_category and
            "category_priority" in self.config.dataset_settings and
            self.config.dataset_settings["category_priority"].get("enable", False)
        )

        for name, samples in presplits.items():
            # 若 split_cfg 中未声明, 直接全量
            if name not in self.split_cfg:
                total_ds = samples
                ds = BsesDataset(total_ds)
                result[name] = ds
                continue
            raw_v = self.split_cfg[name]
            is_placeholder = isinstance(raw_v, int) and raw_v == -1
            is_all = isinstance(raw_v, str) and raw_v == 'all'
            if is_placeholder:
                placeholder_splits.append(name)
                view = samples[:1] if samples else []
                result[name] = BsesDataset(view)
                continue
            if is_all:
                placeholder_splits.append(name)
                result[name] = BsesDataset(samples[:])
                continue
            # 按目标大小截取
            target_sizes = self.compute_split_target_sizes(len(samples))
            need = target_sizes[name] if name in target_sizes else len(samples)
            if need > len(samples):
                need = len(samples)

            # 如果启用类别平衡，使用类别平衡采样
            if use_category_balance and need < len(samples):
                cat_priority = self.config.dataset_settings["category_priority"]
                priority_defs = cat_priority["values"]
                # 查找当前 split 的采样模式
                mode = 'origin'  # 默认模式
                for item in priority_defs:
                    if name in item:
                        mode = item[name]
                        break
                # 使用类别平衡采样
                subset = self._category_balanced_sample(samples, need, mode)
                result[name] = BsesDataset(subset)
            else:
                # 按顺序截取（不打乱，保留原顺序）
                result[name] = BsesDataset(samples[:need])

        # 对于 split_cfg 中声明但未在 presplits 出现的占位/全量, 创建空集合或占位
        for name, v in self.split_cfg.items():
            if name in result:
                continue
            is_placeholder = isinstance(v, int) and v == -1
            is_all = isinstance(v, str) and v == 'all'
            if is_placeholder:
                placeholder_splits.append(name)
                result[name] = BsesDataset([])
            elif is_all:
                placeholder_splits.append(name)
                result[name] = BsesDataset([])
            else:
                result[name] = BsesDataset([])
        return result, placeholder_splits

    # ---- 类别优先拆分 ----
    def _category_balanced_sample(self, samples: List[Dict[str, Any]], need: int, mode: str) -> List[Dict[str, Any]]:
        """从样本列表中按类别平衡采样指定数量的样本

        Args:
            samples: 样本列表
            need: 需要采样的数量
            mode: 采样模式，'mean' 表示均衡各类别，'origin' 表示按原始比例

        Returns:
            采样后的样本列表
        """
        if need >= len(samples):
            return samples[:]

        # 按类别分组
        cat_pool: Dict[Any, List[Dict[str, Any]]] = {}
        for s in samples:
            c = s["category"]
            if c not in cat_pool:
                cat_pool[c] = []
            cat_pool[c].append(s)

        # 打乱每个类别的样本
        random.seed(self.seed)
        for lst in cat_pool.values():
            random.shuffle(lst)

        # 根据模式进行采样
        if mode == 'mean':
            return self._alloc_mean_from_pool(cat_pool, need)
        else:  # 'origin'
            return self._alloc_origin_from_pool(cat_pool, need)

    def _alloc_mean_from_pool(self, cat_pool: Dict[Any, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """从类别池中均衡采样（轮流从各类别取样）"""
        picked: List[Dict[str, Any]] = []
        active = [c for c, lst in cat_pool.items() if lst]
        idx = 0
        while k > 0 and active:
            c = active[idx]
            lst = cat_pool[c]
            if lst:
                picked.append(lst.pop())
                k -= 1
                if not lst:
                    active = [x for x in active if x != c]
                    if active:
                        idx %= len(active)
            idx = (idx + 1) % len(active) if active else 0
        return picked

    def _alloc_origin_from_pool(self, cat_pool: Dict[Any, List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        """从类别池中按原始比例采样"""
        picked: List[Dict[str, Any]] = []
        while k > 0 and any(len(v) > 0 for v in cat_pool.values()):
            remaining = sum(len(v) for v in cat_pool.values())
            if remaining == 0:
                break
            alloc: Dict[Any, int] = {}
            fractional: List[Tuple[float, Any]] = []
            for c, lst in cat_pool.items():
                if not lst:
                    continue
                share = (len(lst) / remaining) * k
                base = int(share)
                alloc[c] = base
                fractional.append((share - base, c))
            allocated = sum(alloc.values())
            leftover = k - allocated
            fractional.sort(key=lambda x: (-x[0], -len(cat_pool[x[1]])))
            for _, c in fractional:
                if leftover <= 0:
                    break
                alloc[c] += 1
                leftover -= 1
            for c, n in alloc.items():
                for _ in range(n):
                    if not cat_pool[c]:
                        break
                    picked.append(cat_pool[c].pop())
            k = 0
        return picked

    def _split_category_priority(self, samples: List[Dict[str, Any]]) -> Tuple[Dict[str, BsesDataset], List[str]]:
        total = len(samples)
        targets = self.compute_split_target_sizes(total)
        cat_pool: Dict[Any, List[Dict[str, Any]]] = {}
        for s in samples:
            c = s["category"]
            if c not in cat_pool:
                cat_pool[c] = []
            cat_pool[c].append(s)
        random.seed(self.seed)
        for lst in cat_pool.values():
            random.shuffle(lst)
        cat_priority = self.config.dataset_settings["category_priority"]
        priority_defs = cat_priority["values"]
        modes: Dict[str, str] = {}
        ordered_priority: List[str] = []
        for item in priority_defs:
            for split_name, mode in item.items():
                if split_name not in modes:
                    modes[split_name] = mode
                    ordered_priority.append(split_name)
        ordered_all: List[str] = []
        for n in ordered_priority:
            ordered_all.append(n)
        for n in self.split_cfg.keys():
            if n not in modes:
                ordered_all.append(n)

        result: Dict[str, BsesDataset] = {}
        placeholder_splits: List[str] = []

        def _alloc_mean(k: int) -> List[Dict[str, Any]]:
            picked: List[Dict[str, Any]] = []
            active = [c for c, lst in cat_pool.items() if lst]
            idx = 0
            while k > 0 and active:
                c = active[idx]
                lst = cat_pool[c]
                if lst:
                    picked.append(lst.pop())
                    k -= 1
                    if not lst:
                        active = [x for x in active if x != c]
                        if active:
                            idx %= len(active)
                idx = (idx + 1) % len(active) if active else 0
            return picked

        def _alloc_origin(k: int) -> List[Dict[str, Any]]:
            picked: List[Dict[str, Any]] = []
            while k > 0 and any(len(v) > 0 for v in cat_pool.values()):
                remaining = sum(len(v) for v in cat_pool.values())
                if remaining == 0:
                    break
                alloc: Dict[Any, int] = {}
                fractional: List[Tuple[float, Any]] = []
                for c, lst in cat_pool.items():
                    if not lst:
                        continue
                    share = (len(lst) / remaining) * k
                    base = int(share)
                    alloc[c] = base
                    fractional.append((share - base, c))
                allocated = sum(alloc.values())
                leftover = k - allocated
                fractional.sort(key=lambda x: (-x[0], -len(cat_pool[x[1]])))
                for _, c in fractional:
                    if leftover <= 0:
                        break
                    alloc[c] += 1
                    leftover -= 1
                for c, n in alloc.items():
                    for _ in range(n):
                        if not cat_pool[c]:
                            break
                        picked.append(cat_pool[c].pop())
                k = 0
            return picked

        for split_name in ordered_all:
            raw_v = self.split_cfg[split_name]
            is_placeholder = isinstance(raw_v, int) and raw_v == -1
            is_all = isinstance(raw_v, str) and raw_v == 'all'
            target_size = targets[split_name] if split_name in targets else 0
            if is_placeholder:
                biggest_cat = None
                biggest_len = -1
                for c, lst in cat_pool.items():
                    if len(lst) > biggest_len:
                        biggest_len = len(lst)
                        biggest_cat = c
                view = [cat_pool[biggest_cat][-1]] if biggest_cat is not None and biggest_len > 0 else []
                result[split_name] = BsesDataset(view)
                placeholder_splits.append(split_name)
                continue
            if is_all:
                result[split_name] = BsesDataset(samples[:])
                placeholder_splits.append(split_name)
                continue
            if target_size <= 0:
                result[split_name] = BsesDataset([])
                continue
            mode = modes[split_name] if split_name in modes else 'origin'
            if mode == 'mean':
                picked = _alloc_mean(target_size)
            else:
                picked = _alloc_origin(target_size)
            result[split_name] = BsesDataset(picked)
        return result, placeholder_splits

    # ---- 元信息与细节 ----
    def build_split_details(self, split_datasets: Dict[str, BsesDataset], placeholder_splits: List[str]) -> Dict[str, Dict[str, Any]]:
        details: Dict[str, Dict[str, Any]] = {}
        priority_modes: Dict[str, str] = {}
        if "category_priority" in self.config.dataset_settings:
            cat_priority = self.config.dataset_settings["category_priority"]
            priority_defs = cat_priority["values"]
            for item in priority_defs:
                for split_name, mode in item.items():
                    if split_name not in priority_modes:
                        priority_modes[split_name] = mode
        for name in split_datasets.keys():
            raw_v = self.split_cfg[name] if name in self.split_cfg else None
            is_placeholder = isinstance(raw_v, int) and raw_v == -1
            is_all = isinstance(raw_v, str) and raw_v == 'all'
            if is_all:
                t = 'all'
            elif is_placeholder:
                t = 'placeholder'
            else:
                t = 'normal'
            mode = None
            if self.has_category and not is_all and not is_placeholder:
                mode = priority_modes[name] if name in priority_modes else ('origin' if "category_priority" in self.config.dataset_settings else None)
            source = 'category' if self.has_category else 'random'
            details[name] = {"type": t, "mode": mode, "source": source}
        return details

    def build_meta(self, all_samples: List[Dict[str, Any]], split_datasets: Dict[str, BsesDataset], applied_field_map: Dict[str, Tuple[str, str]], placeholder_splits: List[str]) -> Dict[str, Any]:
        categories: List[Any] = []
        if self.has_category:
            cat_set: Dict[Any, bool] = {}
            for s in all_samples:
                c = s["category"]
                cat_set[c] = True
            categories = list(cat_set.keys())
        meta = {
            "total": len(all_samples),
            "has_category": self.has_category,
            "categories": categories,
            "split_sizes": {n: len(ds) for n, ds in split_datasets.items()},
            "applied_field_map": applied_field_map,
            "placeholder_splits": placeholder_splits,
        }
        meta["split_details"] = self.build_split_details(split_datasets, placeholder_splits)
        return meta

    # ---- 报告 ----
    def base_report(self, meta: Dict[str, Any]):
        logger = getattr(self.config, "logger", None)
        if logger is None:
            return
        logger.info(f"[Dataset] Total: {meta['total']} | Splits: {meta['split_sizes']}")
        if "split_details" in meta:
            details = meta["split_details"]
            parts = []
            for name in sorted(details.keys()):
                d = details[name]
                mode = d["mode"] if d["mode"] else '-'
                parts.append(f"{name}({d['type']},{mode},{d['source']})")
            logger.info("[Dataset] Split Details: " + ", ".join(parts))
        if meta["has_category"]:
            logger.info(f"[Dataset] Categories: {meta['categories']}")
        if meta["applied_field_map"]:
            desc = ", ".join(f"{src}->{dst}" for src, (_, dst) in meta["applied_field_map"].items())
            logger.info(f"[Dataset] Field Mapping Applied: {desc}")

    # ---- judge 钩子 ----
    def _build_judge(self, meta: Dict[str, Any], splits: Dict[str, BsesDataset]):  # pragma: no cover
        raise NotImplementedError

    def _build_judge_placeholder(self, meta: Dict[str, Any]):
        def _judge(pred, ref, sample=None):
            if isinstance(pred, list):
                return {"correct": 0, "total": len(pred), "accuracy": 0.0}
            return {"correct": 0, "total": 1, "accuracy": 0.0}
        return _judge
