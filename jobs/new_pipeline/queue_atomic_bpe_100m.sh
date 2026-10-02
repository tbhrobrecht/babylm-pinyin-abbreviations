#!/bin/bash

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
JOB_DIR="$ROOT/code/babylm-pinyin-abbreviations/jobs/new_pipeline"

mkdir -p "$ROOT/logs/new_pipeline"

# Validate the complete matrix before submitting anything, so a typo or stale
# output cannot leave a partially queued experiment.
for tokenizer in atomic_bpe_within atomic_bpe_cross; do
    tokenizer_dir="$ROOT/code/babylm-pinyin-abbreviations/tokenizers/nk_babylm_zho_${tokenizer}"
    train_dataset="$ROOT/code/babylm-pinyin-abbreviations/data/datasets/nk_babylm_zho_train_${tokenizer}.bin"
    valid_dataset="$ROOT/code/babylm-pinyin-abbreviations/data/datasets/nk_babylm_zho_valid_${tokenizer}.bin"
    for required in "$tokenizer_dir/vocab.json" "$train_dataset" "$valid_dataset"; do
        if [[ ! -s "$required" ]]; then
            echo "Required non-empty input does not exist: $required" >&2
            exit 2
        fi
    done
    for architecture in gpt2 qwen2; do
        model_dir="$ROOT/code/babylm-pinyin-abbreviations/models/nk_babylm_zho_${architecture}_${tokenizer}_100m"
        hf_dir="$ROOT/code/babylm-pinyin-abbreviations/artifacts/models/100m/hf_nk_babylm_zho_${architecture}_${tokenizer}_100m"
        if [[ -e "$model_dir" || -e "$hf_dir" ]]; then
            echo "Refusing to queue over an existing output: $model_dir or $hf_dir" >&2
            exit 2
        fi
    done
done

previous_job=""
for tokenizer in atomic_bpe_within atomic_bpe_cross; do
    for architecture in gpt2 qwen2; do
        short_tokenizer="${tokenizer#atomic_bpe_}"
        job_name="nk100-${architecture}-${short_tokenizer}"
        dependency=()
        if [[ -n "$previous_job" ]]; then
            # Serialize the disk-heavy jobs, but do not strand the remaining
            # experiments if one model encounters an unrelated failure.
            dependency=(--dependency="afterany:$previous_job")
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
