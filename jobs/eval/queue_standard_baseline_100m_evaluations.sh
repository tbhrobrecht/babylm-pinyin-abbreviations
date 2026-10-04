#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"
JOB_DIR="$REPO/jobs/eval"
RESULTS_DIR="$ROOT/results/eval/chinese_pipeline_standard_baselines_100m"

names=(
  hanzi
  full-pinyin
)
model_dirs=(
  "$REPO/artifacts/models/100m/hf_nk_babylm_zho_gpt2_hanzi_bpe_100m"
  "$REPO/artifacts/models/100m/hf_nk_babylm_zho_gpt2_full_pinyin_bpe_100m"
)

for required in \
  "$ROOT/env/babylm-eval-official/bin/python" \
  "$ROOT/env/chinese-babylm-eval/bin/python" \
  "$ROOT/code/babylm-eval/multilingual/scripts/eval_model_full.sh" \
  "$ROOT/code/chinese-babylm-eval-pipeline/pipeline.py" \
  "$JOB_DIR/eval_nk_official_zho.sbatch" \
  "$JOB_DIR/eval_nk_chinese_pipeline.sbatch" \
  "$JOB_DIR/rerun_nk_pos_only.sbatch"; do
  if [[ ! -e "$required" ]]; then
    echo "Required evaluation input does not exist: $required" >&2
    exit 2
  fi
done

for model_dir in "${model_dirs[@]}"; do
  for required in config.json model.safetensors tokenizer_config.json tokenizer.model; do
    if [[ ! -s "$model_dir/$required" ]]; then
      echo "Incomplete Transformers export: $model_dir/$required" >&2
      exit 2
    fi
  done
done

mkdir -p "$ROOT/logs/eval" "$RESULTS_DIR"
cd "$ROOT"

ids_file="$ROOT/logs/eval/standard_baseline_100m_eval_jobs_$(date +%Y%m%d_%H%M%S).txt"

for i in "${!names[@]}"; do
  official_job="$(sbatch --parsable \
    --job-name="off100b-${names[$i]}" \
    --time=02:00:00 \
    --export="ALL,MODEL_DIR=${model_dirs[$i]}" \
    "$JOB_DIR/eval_nk_official_zho.sbatch")"

  chinese_job="$(sbatch --parsable \
    --job-name="zh100b-${names[$i]}" \
    --time=04:00:00 \
    --export="ALL,MODEL_DIR=${model_dirs[$i]},RESULTS_DIR=$RESULTS_DIR,TASKS=all" \
    "$JOB_DIR/eval_nk_chinese_pipeline.sbatch")"

  pos_job="$(sbatch --parsable \
    --dependency="afterok:$official_job" \
    --job-name="pos100b-${names[$i]}" \
    --time=01:00:00 \
    --export="ALL,MODEL_DIR=${model_dirs[$i]}" \
    "$JOB_DIR/rerun_nk_pos_only.sbatch")"

  printf '%-12s official=%s chinese_pipeline=%s pos_repair=%s\n' \
    "${names[$i]}" "$official_job" "$chinese_job" "$pos_job" | tee -a "$ids_file"
done

printf 'Job IDs saved to %s\n' "$ids_file"
