#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
JOB_DIR="$ROOT/code/babylm-pinyin-abbreviations/jobs/new_pipeline"

mkdir -p "$ROOT/logs/new_pipeline"

prep_job="$(sbatch --parsable "$JOB_DIR/prepare_standard_baselines.sbatch")"
echo "prepare=$prep_job"

previous_job="$prep_job"
for baseline in hanzi full-pinyin; do
    short_name="${baseline//-/_}"
    job_name="nk100-gpt2-${short_name}"
    job_id="$(sbatch \
        --parsable \
        --dependency="afterok:$previous_job" \
        --job-name="$job_name" \
        --export="ALL,BASELINE=$baseline" \
        "$JOB_DIR/train_standard_baseline_100m_a100.sbatch")"
    printf '%-32s %s\n' "$job_name" "$job_id"
    previous_job="$job_id"
done
