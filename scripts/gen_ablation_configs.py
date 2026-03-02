#!/usr/bin/env python3
"""
Generate ablation config files from a base VQAv2 training config and a base POPE eval config.

This script does NOT run training/eval. It only writes derived YAML configs so that
`scripts/run_ddp.sh` and `scripts/run_eval_ddp.sh` can be used as-is.

Outputs:
  - <out_dir>/manifest.tsv
  - <out_dir>/configs/train_<variant>.yaml
  - <out_dir>/configs/pope_<variant>.yaml

Available variants:
  - full
  - w_o_pruner_topk_attn
  - w_o_adapter
  - w_o_repair_loss
  - repair_mean_only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def _dump_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _ensure_method(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "method_settings" not in cfg or not isinstance(cfg["method_settings"], dict):
        cfg["method_settings"] = {}
    return cfg["method_settings"]


def _ensure_global(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "global_settings" not in cfg or not isinstance(cfg["global_settings"], dict):
        cfg["global_settings"] = {}
    return cfg["global_settings"]


def _apply_common(
    cfg: Dict[str, Any],
    *,
    target_token_num: int,
    study_name: str,
    ablation_flags: Dict[str, bool],
    override_repair_loss_weight: float | None = None,
    override_teacher_forward_enable: bool | None = None,
    override_use_repair_adapter: bool | None = None,
    override_repair_var_weight: float | None = None,
) -> Dict[str, Any]:
    g = _ensure_global(cfg)
    m = _ensure_method(cfg)

    g["study_name"] = study_name
    m["target_token_num"] = int(target_token_num)

    # Fill all four ablation flags explicitly (avoid relying on defaults).
    for k in (
        "ablation_w_o_pruner_topk_attn",
        "ablation_w_o_adapter",
        "ablation_w_o_repair_loss",
        "ablation_repair_mean_only",
    ):
        m[k] = bool(ablation_flags.get(k, False))

    if override_repair_loss_weight is not None:
        m["repair_loss_weight"] = float(override_repair_loss_weight)
    if override_teacher_forward_enable is not None:
        m["teacher_forward_enable"] = bool(override_teacher_forward_enable)
    if override_use_repair_adapter is not None:
        m["use_repair_adapter"] = bool(override_use_repair_adapter)
    if override_repair_var_weight is not None:
        m["repair_var_weight"] = float(override_repair_var_weight)

    return cfg


def build_variants() -> List[Tuple[str, Dict[str, Any]]]:
    """Return list of (variant_name, variant_options) for config generation."""
    return [
        (
            "full",
            dict(
                ablation_flags={},
            ),
        ),
        (
            "w_o_pruner_topk_attn",
            dict(
                ablation_flags={"ablation_w_o_pruner_topk_attn": True},
            ),
        ),
        (
            "w_o_adapter",
            dict(
                # Definition: "no delayed repair" => disable repair structure + disable repair loss.
                ablation_flags={"ablation_w_o_adapter": True},
                override_use_repair_adapter=False,
                override_teacher_forward_enable=False,
                override_repair_loss_weight=0.0,
            ),
        ),
        (
            "w_o_repair_loss",
            dict(
                # Definition: λ_repair=0 but keep delayed repair structure.
                ablation_flags={"ablation_w_o_repair_loss": True},
                override_teacher_forward_enable=False,
                override_repair_loss_weight=0.0,
            ),
        ),
        (
            "repair_mean_only",
            dict(
                # Definition: mean-only vs mean+var => set alpha(var weight)=0.
                ablation_flags={"ablation_repair_mean_only": True},
                override_repair_var_weight=0.0,
            ),
        ),
    ]


def _filter_variants(
    variants: List[Tuple[str, Dict[str, Any]]],
    selected: List[str] | None,
) -> List[Tuple[str, Dict[str, Any]]]:
    if not selected:
        return variants
    name_to_variant = {name: opts for name, opts in variants}
    unknown = [v for v in selected if v not in name_to_variant]
    if unknown:
        available = ", ".join([name for name, _ in variants])
        raise ValueError(f"Unknown variants: {unknown}. Available: [{available}]")
    return [(name, name_to_variant[name]) for name in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_train", type=str, required=True, help="Base VQAv2 training config (yaml)")
    parser.add_argument("--base_pope", type=str, required=True, help="Base POPE eval config (yaml)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--target_token_num", type=int, default=128, help="Target kept vision tokens (default: 128)")
    parser.add_argument("--study_prefix", type=str, default="ab128", help="Prefix for global_settings.study_name")
    parser.add_argument(
        "--variants",
        type=str,
        default="",
        help=(
            "Comma-separated variant names to generate (default: generate all). "
            "Example: 'w_o_pruner_topk_attn,repair_mean_only'."
        ),
    )
    args = parser.parse_args()

    base_train = Path(args.base_train)
    base_pope = Path(args.base_pope)
    out_dir = Path(args.out_dir)

    train_base_cfg = _load_yaml(base_train)
    pope_base_cfg = _load_yaml(base_pope)

    configs_dir = out_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.tsv"
    rows: List[str] = ["variant\ttrain_config\tpope_config"]

    selected = [v.strip() for v in (args.variants or "").split(",") if v.strip()]
    for variant, opts in _filter_variants(build_variants(), selected):
        train_cfg = yaml.safe_load(yaml.safe_dump(train_base_cfg, sort_keys=False, allow_unicode=True))
        pope_cfg = yaml.safe_load(yaml.safe_dump(pope_base_cfg, sort_keys=False, allow_unicode=True))

        study_train = f"{args.study_prefix}_{variant}"
        study_pope = f"{args.study_prefix}_{variant}_pope"

        _apply_common(
            train_cfg,
            target_token_num=args.target_token_num,
            study_name=study_train,
            ablation_flags=opts.get("ablation_flags", {}),
            override_repair_loss_weight=opts.get("override_repair_loss_weight"),
            override_teacher_forward_enable=opts.get("override_teacher_forward_enable"),
            override_use_repair_adapter=opts.get("override_use_repair_adapter"),
            override_repair_var_weight=opts.get("override_repair_var_weight"),
        )
        _apply_common(
            pope_cfg,
            target_token_num=args.target_token_num,
            study_name=study_pope,
            ablation_flags=opts.get("ablation_flags", {}),
            override_repair_loss_weight=opts.get("override_repair_loss_weight"),
            override_teacher_forward_enable=opts.get("override_teacher_forward_enable"),
            override_use_repair_adapter=opts.get("override_use_repair_adapter"),
            override_repair_var_weight=opts.get("override_repair_var_weight"),
        )

        train_out = configs_dir / f"train_{variant}.yaml"
        pope_out = configs_dir / f"pope_{variant}.yaml"
        _dump_yaml(train_out, train_cfg)
        _dump_yaml(pope_out, pope_cfg)
        rows.append(f"{variant}\t{train_out.as_posix()}\t{pope_out.as_posix()}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(manifest_path.as_posix())


if __name__ == "__main__":
    main()
