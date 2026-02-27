TODO (Domain Adaptation Token Pruning)

- Eval/generate path is not updated for delayed repair adapter (gen_answer-only repair).
  - Training forward applies delayed repair on gen_answer hidden states (student) and uses teacher keep-all forward
    to compute distribution alignment loss.
  - Current `hard` eval uses `generate_with_hard_pruning()` which does NOT apply delayed repair, so accuracy metrics
    do not reflect repair benefits.
  - Temporary workaround: limit eval to 1 sample in `configs/vision_token_pruning.yaml` to avoid wasting time and
    misleading comparisons.

- Next steps (tomorrow):
  - Option A: add a new eval mode (e.g., `hard_repair`) that runs a teacher-forcing forward with `apply_repair=True`
    and reports NLL/entropy deltas for answer/gen_answer region.
  - Option B: integrate delayed repair into `generate_with_hard_pruning()` prefill (and possibly decode), so `hard`
    eval accuracy reflects repair.
  - Decide whether repair should be applied only in prefill, or also during decode (KV cache consistency).

