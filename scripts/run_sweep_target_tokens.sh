#!/bin/bash
# Sweep target_token_num with the existing DDP launcher.
#
# Runs 3 experiments sequentially:
#   target_token_num in {192, 128, 64}
#
# It generates per-target config copies under `configs/sweeps/` (does NOT modify the base config).
#
# Usage:
#   ./scripts/run_sweep_target_tokens.sh
#   ./scripts/run_sweep_target_tokens.sh "0,1,4,5"
#   ./scripts/run_sweep_target_tokens.sh "0,1,4,5" configs/vision_token_pruning.yaml
#
# Notes:
# - Config loader uses `global_settings.study_name` as a prefix of experiment_tag.
#   We auto-set it to `t{target}` for easier bookkeeping.

set -euo pipefail

GPU_IDS="${1:-0,1,4,5}"
BASE_CONFIG="${2:-configs/vision_token_pruning.yaml}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ ! -f "$BASE_CONFIG" ]; then
  echo "[ERR] Base config not found: $BASE_CONFIG" >&2
  exit 1
fi

SWEEP_DIR="configs/sweeps"
mkdir -p "$SWEEP_DIR"

make_config() {
  local target="$1"
  local out="${SWEEP_DIR}/vision_token_pruning_target${target}.yaml"

  cp "$BASE_CONFIG" "$out"

  # Set study_name for nicer experiment_tag prefix (insert if missing, else replace).
  if grep -qE '^[[:space:]]+study_name:' "$out"; then
    sed -i -E "s/^([[:space:]]+study_name:).*/\\1 \"t${target}\"/" "$out"
  else
    # Insert right after the seed line (keeps indentation consistent with this repo's configs).
    sed -i -E "/^[[:space:]]+seed:/a\\
  study_name: \"t${target}\"
" "$out"
  fi

  # Set target_token_num.
  sed -i -E "s/^([[:space:]]*target_token_num:)[[:space:]]*[0-9]+/\\1 ${target}/" "$out"

  echo "$out"
}

run_one() {
  local target="$1"
  local cfg
  cfg="$(make_config "$target")"

  echo "============================================================"
  echo "[SWEEP] target_token_num=${target}"
  echo "[SWEEP] GPUs=${GPU_IDS}"
  echo "[SWEEP] config=${cfg}"
  echo "============================================================"

  ./scripts/run_ddp.sh "$GPU_IDS" "$cfg"
}

run_one 192
run_one 128
run_one 64
