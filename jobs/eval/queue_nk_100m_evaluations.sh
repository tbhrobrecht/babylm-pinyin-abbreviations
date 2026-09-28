#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"
JOB_DIR="$REPO/jobs/eval"
RESULTS_DIR="$ROOT/results/eval/chinese_pipeline_100m"

names=(
  gpt2-hyb
  qwen2-hyb
  gpt2-bpe
)
model_dirs=(
  "$REPO/hf_nk_babylm_zho_gpt2_hybrid_100m"
  "$REPO/hf_nk_babylm_zho_qwen2_hybrid_100m"
  "$REPO/hf_nk_babylm_zho_gpt2_bpe_100m"
)

for required in \
  "$ROOT/env/babylm-eval-official/bin/python" \
  "$ROOT/env/chinese-babylm-eval/bin/python" \
  "$ROOT/code/babylm-eval/multilingual/scripts/eval_model_full.sh" \
  "$ROOT/code/chinese-babylm-eval-pipeline/pipeline.py" \
  "$JOB_DIR/eval_nk_official_zho.sbatch" \
  "$JOB_DIR/eval_nk_chinese_pipeline.sbatch"; do
  if [[ ! -e "$required" ]]; then
    echo "Required evaluation input does not exist: $required" >&2
    exit 2
  fi
done

for model_dir in "${model_dirs[@]}"; do
  for required in config.json model.safetensors tokenizer_config.json; do
    if [[ ! -s "$model_dir/$required" ]]; then
      echo "Incomplete Hugging Face export: $model_dir/$required" >&2
      exit 2
    fi
  done
done

mkdir -p "$ROOT/logs/eval" "$RESULTS_DIR"
cd "$ROOT"

ids_file="$ROOT/logs/eval/nk_100m_eval_jobs_$(date +%Y%m%d_%H%M%S).txt"

for i in "${!names[@]}"; do
  official_job="$(sbatch --parsable \
    --job-name="off100-${names[$i]}" \
    --export="ALL,MODEL_DIR=${model_dirs[$i]}" \
    "$JOB_DIR/eval_nk_official_zho.sbatch")"

  chinese_job="$(sbatch --parsable \
    --job-name="zh100-${names[$i]}" \
    --export="ALL,MODEL_DIR=${model_dirs[$i]},RESULTS_DIR=$RESULTS_DIR,TASKS=all" \
    "$JOB_DIR/eval_nk_chinese_pipeline.sbatch")"

  printf '%-10s official=%s chinese_pipeline=%s\n' \
    "${names[$i]}" "$official_job" "$chinese_job" | tee -a "$ids_file"
done

printf 'Job IDs saved to %s\n' "$ids_file"
