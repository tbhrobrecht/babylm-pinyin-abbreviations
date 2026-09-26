#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
JOB_DIR="$ROOT/code/babylm-pinyin-abbreviations/jobs/eval"

mkdir -p "$ROOT/logs/eval" "$ROOT/results/eval"
cd "$ROOT"

setup_job="$(sbatch --parsable "$JOB_DIR/setup_chinese_eval_env.sbatch")"
data_job="$(sbatch --parsable \
  --dependency="afterok:$setup_job" \
  "$JOB_DIR/download_chinese_eval_data.sbatch")"

names=(
  gpt2_hybrid
  qwen2_hybrid
  gpt2_bpe
  qwen2_bpe
)
train_jobs=(
  5804010
  5804011
  5804012
  5804013
)
model_dirs=(
  "$ROOT/code/babylm-pinyin-abbreviations/hf_nk_babylm_zho_gpt2_hybrid"
  "$ROOT/code/babylm-pinyin-abbreviations/hf_nk_babylm_zho_qwen2_hybrid"
  "$ROOT/code/babylm-pinyin-abbreviations/hf_nk_babylm_zho_gpt2_bpe"
  "$ROOT/code/babylm-pinyin-abbreviations/hf_nk_babylm_zho_qwen2_bpe"
)

printf 'setup environment: %s\n' "$setup_job"
printf 'download datasets: %s\n' "$data_job"

for i in "${!names[@]}"; do
  official_job="$(sbatch --parsable \
    --dependency="afterok:${train_jobs[$i]}" \
    --job-name="off-${names[$i]}" \
    --export="ALL,MODEL_DIR=${model_dirs[$i]}" \
    "$JOB_DIR/eval_nk_official_zho.sbatch")"

  chinese_job="$(sbatch --parsable \
    --dependency="afterok:${train_jobs[$i]}:$data_job" \
    --job-name="zh-${names[$i]}" \
    --export="ALL,MODEL_DIR=${model_dirs[$i]}" \
    "$JOB_DIR/eval_nk_chinese_pipeline.sbatch")"

  printf '%-14s official=%s chinese_pipeline=%s\n' \
    "${names[$i]}" "$official_job" "$chinese_job"
done
