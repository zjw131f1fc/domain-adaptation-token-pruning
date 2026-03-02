#!/bin/bash
# Run all paper ablations (target-token base) sequentially:
#   1) Train / eval on VQAv2 via configs/vision_token_pruning.yaml (auto-eval at end of training)
#   2) Eval the trained checkpoint on POPE via configs/vision_token_pruning_pope.yaml
#
# This script uses the existing launchers:
#   - scripts/run_ddp.sh
#   - scripts/run_eval_ddp.sh
#
# Usage:
#   ./scripts/run_ablation_suite.sh <GPU_IDS> [TARGET_TOKEN_NUM]
# Example (2 GPUs):
#   ./scripts/run_ablation_suite.sh 4,5 128
#   ./scripts/run_ablation_suite.sh 4,5 64
#
# Outputs:
#   - logs/ddp_runs/train_*.log            (from run_ddp.sh)
#   - logs/eval_runs/eval_*.log            (from run_eval_ddp.sh)
#   - logs/ablation_sweeps/<run_id>/*      (manifest + mapping)

set -euo pipefail

# Ensure we run from repo root (so relative paths like logs/ and configs/ resolve).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

GPU_IDS="${1:-}"
if [[ -z "$GPU_IDS" ]]; then
  echo "Error: GPU_IDS is required (use your 2 GPUs), e.g. '4,5'." >&2
  exit 1
fi
TARGET_TOKEN_NUM="${2:-64}"
if ! [[ "$TARGET_TOKEN_NUM" =~ ^[0-9]+$ ]]; then
  echo "Error: TARGET_TOKEN_NUM must be an integer (got '$TARGET_TOKEN_NUM')." >&2
  exit 1
fi
STUDY_PREFIX="ab${TARGET_TOKEN_NUM}"
NUM_GPUS="$(echo "$GPU_IDS" | tr ',' '\n' | wc -l | tr -d ' ')"
if [[ "$NUM_GPUS" != "2" ]]; then
  echo "Error: this suite expects exactly 2 GPUs (got '$GPU_IDS' => $NUM_GPUS GPUs)." >&2
  exit 1
fi

# Base configs (user-provided)
BASE_TRAIN_CFG="configs/vision_token_pruning.yaml"
BASE_POPE_CFG="configs/vision_token_pruning_pope.yaml"

if [[ ! -f "$BASE_TRAIN_CFG" ]]; then
  echo "Error: base train config not found: $BASE_TRAIN_CFG" >&2
  exit 1
fi
if [[ ! -f "$BASE_POPE_CFG" ]]; then
  echo "Error: base pope config not found: $BASE_POPE_CFG" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
SWEEP_DIR="logs/ablation_sweeps/${RUN_ID}"
mkdir -p "$SWEEP_DIR"

echo "[AblationSuite] Run ID: $RUN_ID"
echo "[AblationSuite] Sweep dir: $SWEEP_DIR"
echo "[AblationSuite] Target token num: $TARGET_TOKEN_NUM"

MANIFEST_PATH="$(python scripts/gen_ablation_configs.py \
  --base_train "$BASE_TRAIN_CFG" \
  --base_pope "$BASE_POPE_CFG" \
  --out_dir "$SWEEP_DIR" \
  --target_token_num "$TARGET_TOKEN_NUM" \
  --study_prefix "$STUDY_PREFIX")"

echo "[AblationSuite] Manifest: $MANIFEST_PATH"

MAPPING_PATH="${SWEEP_DIR}/run_map.tsv"
echo -e "variant\ttrain_cfg\tpope_cfg\ttrain_wrapper_log\tfinal_ckpt\ttrain_task_log\tpope_wrapper_log\tpope_task_log" > "$MAPPING_PATH"

function _latest_train_log() {
  ls -1t logs/ddp_runs/train_*.log 2>/dev/null | head -n 1 || true
}
function _latest_eval_log() {
  ls -1t logs/eval_runs/eval_*.log 2>/dev/null | head -n 1 || true
}

function _extract_final_ckpt() {
  local train_log="$1"
  local ckpt
  ckpt="$(grep -F 'Training completed. Final checkpoint saved to' "$train_log" | tail -n 1 | sed -E 's/^.*Final checkpoint saved to //')"
  if [[ -z "$ckpt" ]]; then
    # Fallback: infer from latest task checkpoint (useful if log parsing changes).
    ckpt="$(ls -1t outputs/tasks/*/checkpoints/checkpoint_final.pt 2>/dev/null | head -n 1 || true)"
  fi
  echo "$ckpt"
}

function _task_log_from_ckpt() {
  local ckpt="$1"
  local ckpt_dir task_dir tag
  ckpt_dir="$(dirname "$ckpt")"          # .../checkpoints
  task_dir="$(dirname "$ckpt_dir")"      # .../outputs/tasks/<tag>
  tag="$(basename "$task_dir")"
  echo "${task_dir}/logs/${tag}.log"
}

function _find_new_log_after() {
  local before="$1"
  local kind="$2" # train|eval
  local latest=""
  if [[ "$kind" == "train" ]]; then
    latest="$(_latest_train_log)"
  else
    latest="$(_latest_eval_log)"
  fi
  if [[ -z "$latest" ]]; then
    echo ""
    return 0
  fi
  if [[ "$latest" == "$before" ]]; then
    # Fallback: still return latest; caller can decide if acceptable.
    echo "$latest"
    return 0
  fi
  echo "$latest"
}

while IFS=$'\t' read -r VARIANT TRAIN_CFG POPE_CFG; do
  echo "============================================================"
  echo "[AblationSuite] Variant: $VARIANT"
  echo "[AblationSuite] Train config: $TRAIN_CFG"
  echo "[AblationSuite] POPE config: $POPE_CFG"
  echo "============================================================"

  BEFORE_TRAIN_LOG="$(_latest_train_log)"
  ./scripts/run_ddp.sh "$GPU_IDS" "$TRAIN_CFG"
  TRAIN_WRAPPER_LOG="$(_find_new_log_after "$BEFORE_TRAIN_LOG" "train")"

  if [[ -z "$TRAIN_WRAPPER_LOG" || ! -f "$TRAIN_WRAPPER_LOG" ]]; then
    echo "Error: cannot locate train wrapper log in logs/ddp_runs/." >&2
    exit 1
  fi

  FINAL_CKPT="$(_extract_final_ckpt "$TRAIN_WRAPPER_LOG" || true)"
  if [[ -z "$FINAL_CKPT" || ! -f "$FINAL_CKPT" ]]; then
    echo "Error: cannot extract checkpoint_final.pt from train log: $TRAIN_WRAPPER_LOG" >&2
    echo "Hint: grep for 'Final checkpoint' in the log to debug." >&2
    exit 1
  fi
  TRAIN_TASK_LOG="$(_task_log_from_ckpt "$FINAL_CKPT")"

  BEFORE_EVAL_LOG="$(_latest_eval_log)"
  ./scripts/run_eval_ddp.sh "$GPU_IDS" --config "$POPE_CFG" --checkpoint "$FINAL_CKPT"
  POPE_WRAPPER_LOG="$(_find_new_log_after "$BEFORE_EVAL_LOG" "eval")"

  if [[ -z "$POPE_WRAPPER_LOG" || ! -f "$POPE_WRAPPER_LOG" ]]; then
    echo "Error: cannot locate eval wrapper log in logs/eval_runs/." >&2
    exit 1
  fi

  # Derive POPE task log by scanning the newest outputs/tasks directory created/updated.
  # (eval creates its own task dir via config loader)
  POPE_TASK_DIR="$(ls -1dt outputs/tasks/* 2>/dev/null | head -n 1 || true)"
  POPE_TASK_LOG=""
  if [[ -n "$POPE_TASK_DIR" ]]; then
    POPE_TAG="$(basename "$POPE_TASK_DIR")"
    POPE_TASK_LOG="${POPE_TASK_DIR}/logs/${POPE_TAG}.log"
  fi

  echo -e "${VARIANT}\t${TRAIN_CFG}\t${POPE_CFG}\t${TRAIN_WRAPPER_LOG}\t${FINAL_CKPT}\t${TRAIN_TASK_LOG}\t${POPE_WRAPPER_LOG}\t${POPE_TASK_LOG}" >> "$MAPPING_PATH"

  echo "[AblationSuite] Done: $VARIANT"
  echo "  - VQAv2 train wrapper log: $TRAIN_WRAPPER_LOG"
  echo "  - VQAv2 final checkpoint:  $FINAL_CKPT"
  echo "  - VQAv2 task log:         $TRAIN_TASK_LOG"
  echo "  - POPE eval wrapper log:  $POPE_WRAPPER_LOG"
  echo "  - POPE task log:          $POPE_TASK_LOG"
done < <(tail -n +2 "$MANIFEST_PATH")

echo "============================================================"
echo "[AblationSuite] All variants completed."
echo "[AblationSuite] Mapping TSV: $MAPPING_PATH"
echo "============================================================"
