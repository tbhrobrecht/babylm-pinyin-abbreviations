#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"
JOB_DIR="$REPO/jobs/eval"

mkdir -p "$ROOT/logs/eval"
cd "$ROOT"

repair_job="$(sbatch --parsable "$JOB_DIR/repair_eval_dependencies.sbatch")"
ids_file="$ROOT/logs/eval/final-missing-reruns-$repair_job.ids"
printf 'repair=%s\n' "$repair_job" >"$ids_file"

names=(
  gpt2-hybrid
  qwen2-hybrid
  gpt2-bpe
  qwen2-bpe
)
model_dirs=(
  "$REPO/hf_nk_babylm_zho_gpt2_hybrid"
  "$REPO/hf_nk_babylm_zho_qwen2_hybrid"
  "$REPO/hf_nk_babylm_zho_gpt2_bpe"
  "$REPO/hf_nk_babylm_zho_qwen2_bpe"
)
printf 'repair=%s\n' "$repair_job"
terminal_jobs=()
for i in "${!names[@]}"; do
  model_dir="${model_dirs[$i]}"
  [[ -f "$model_dir/config.json" ]] || {
    echo "Missing converted model: $model_dir" >&2
    exit 1
  }

  if [[ "${names[$i]}" == *-hybrid ]]; then
    pos_job="$(sbatch --parsable \
      --dependency="afterok:$repair_job" \
      --job-name="pos-${names[$i]}" \
      --export="ALL,MODEL_DIR=$model_dir" \
      "$JOB_DIR/rerun_nk_pos_only.sbatch")"
    terminal_jobs+=("$pos_job")
  else
    pos_job="reuse-existing"
  fi

  if [[ "${names[$i]}" == *-hybrid ]]; then
    missing_chinese_tasks="zhoblimp:hanzi_structure:hanzi_pinyin:word_fmri:fmri"
  else
    missing_chinese_tasks="hanzi_structure:hanzi_pinyin:word_fmri:fmri"
  fi

  chinese_job="$(sbatch --parsable \
    --dependency="afterok:$repair_job" \
    --job-name="zhmiss-${names[$i]}" \
    --export="ALL,MODEL_DIR=$model_dir,TASKS=$missing_chinese_tasks" \
    "$JOB_DIR/eval_nk_chinese_pipeline.sbatch")"
  terminal_jobs+=("$chinese_job")

  printf '%s pos=%s chinese=%s tasks=%s\n' \
    "${names[$i]}" "$pos_job" "$chinese_job" "$missing_chinese_tasks" | tee -a "$ids_file"
done

dependency="$(IFS=:; echo "${terminal_jobs[*]}")"
finalize_job="$(sbatch --parsable \
  --dependency="afterok:$dependency" \
  "$JOB_DIR/finalize_nk_evaluations.sbatch")"
printf 'finalize=%s dependency=%s\n' "$finalize_job" "$dependency" | tee -a "$ids_file"

printf 'job_ids=%s\n' "$ids_file"
