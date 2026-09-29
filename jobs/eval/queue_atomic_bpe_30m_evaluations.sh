#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"
JOB_DIR="$REPO/jobs/eval"
RESULTS_DIR="$ROOT/results/eval/chinese_pipeline_atomic_bpe_30m"

names=(
  gpt2-within
  qwen2-within
  gpt2-cross
  qwen2-cross
)
model_dirs=(
  "$REPO/artifacts/models/30m/hf_nk_babylm_zho_gpt2_atomic_bpe_within"
  "$REPO/artifacts/models/30m/hf_nk_babylm_zho_qwen2_atomic_bpe_within"
  "$REPO/artifacts/models/30m/hf_nk_babylm_zho_gpt2_atomic_bpe_cross"
  "$REPO/artifacts/models/30m/hf_nk_babylm_zho_qwen2_atomic_bpe_cross"
)

dependency_args=()
if [[ -n "${AFTEROK:-}" ]]; then
  dependency_args=(--dependency="afterok:$AFTEROK")
fi

for model_dir in "${model_dirs[@]}"; do
  for required in config.json model.safetensors tokenizer_config.json; do
    if [[ ! -s "$model_dir/$required" ]]; then
      echo "Incomplete Transformers export: $model_dir/$required" >&2
      exit 2
    fi
  done
done

mkdir -p "$ROOT/logs/eval" "$RESULTS_DIR"
cd "$ROOT"

ids_file="$ROOT/logs/eval/atomic_bpe_30m_eval_jobs_$(date +%Y%m%d_%H%M%S).txt"

for i in "${!names[@]}"; do
  official_job="$(sbatch --parsable \
    "${dependency_args[@]}" \
    --job-name="off30-${names[$i]}" \
    --export="ALL,MODEL_DIR=${model_dirs[$i]}" \
    "$JOB_DIR/eval_nk_official_zho.sbatch")"

  chinese_job="$(sbatch --parsable \
    "${dependency_args[@]}" \
    --job-name="zh30-${names[$i]}" \
    --export="ALL,MODEL_DIR=${model_dirs[$i]},RESULTS_DIR=$RESULTS_DIR,TASKS=all" \
    "$JOB_DIR/eval_nk_chinese_pipeline.sbatch")"

  pos_job="$(sbatch --parsable \
    --dependency="afterok:$official_job" \
    --job-name="pos30-${names[$i]}" \
    --export="ALL,MODEL_DIR=${model_dirs[$i]}" \
    "$JOB_DIR/rerun_nk_pos_only.sbatch")"

  printf '%-14s official=%s chinese_pipeline=%s pos_repair=%s\n' \
    "${names[$i]}" "$official_job" "$chinese_job" "$pos_job" | tee -a "$ids_file"
done

printf 'Job IDs saved to %s\n' "$ids_file"
