#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
JOB_DIR="$ROOT/code/babylm-pinyin-abbreviations/jobs/new_pipeline"

mkdir -p "$ROOT/logs/new_pipeline"

previous_job=""
for tokenizer in atomic_bpe_within atomic_bpe_cross; do
    for architecture in gpt2 qwen2; do
        short_tokenizer="${tokenizer#atomic_bpe_}"
        job_name="nk100-${architecture}-${short_tokenizer}"
        dependency=()
        if [[ -n "$previous_job" ]]; then
            dependency=(--dependency="afterok:$previous_job")
        fi
        job_id="$(sbatch \
            --parsable \
            "${dependency[@]}" \
            --job-name="$job_name" \
            --export="ALL,ARCH=$architecture,TOK=$tokenizer" \
            "$JOB_DIR/train_atomic_bpe_100m_a100.sbatch")"
        printf '%-34s %s\n' "$job_name" "$job_id"
        previous_job="$job_id"
    done
done
