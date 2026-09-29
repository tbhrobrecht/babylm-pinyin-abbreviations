#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
JOB_DIR="$ROOT/code/babylm-pinyin-abbreviations/jobs/new_pipeline"

mkdir -p "$ROOT/logs/new_pipeline"

prep_job="$(sbatch --parsable "$JOB_DIR/prepare_atomic_bpe_30m.sbatch")"
echo "prepare=$prep_job"

for tokenizer in atomic_bpe_within atomic_bpe_cross; do
    for architecture in gpt2 qwen2; do
        short_tokenizer="${tokenizer#atomic_bpe_}"
        job_name="nk30-${architecture}-${short_tokenizer}"
        job_id="$(sbatch \
            --parsable \
            --dependency="afterok:$prep_job" \
            --job-name="$job_name" \
            --export="ALL,ARCH=$architecture,TOK=$tokenizer" \
            "$JOB_DIR/train_atomic_bpe_30m_a100.sbatch")"
        printf '%-32s %s\n' "$job_name" "$job_id"
    done
done
