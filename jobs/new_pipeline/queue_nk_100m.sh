#!/bin/bash

set -euo pipefail

ROOT="$HOME/babylm-lrz"
SCRIPT="$ROOT/code/babylm-pinyin-abbreviations/jobs/new_pipeline/train_nk_100m_a100.sbatch"

if [[ ! -f "$SCRIPT" ]]; then
    echo "Training script not found: $SCRIPT" >&2
    exit 2
fi

submit() {
    local architecture="$1"
    local tokenizer="$2"
    local job_name="$3"
    sbatch \
        --parsable \
        --job-name="$job_name" \
        --export="ALL,ARCH=$architecture,TOK=$tokenizer" \
        "$SCRIPT"
}

echo "gpt2_hybrid=$(submit gpt2 hybrid nk100-gpt2-hyb)"
echo "qwen2_hybrid=$(submit qwen2 hybrid nk100-qwen2-hyb)"
echo "gpt2_bpe=$(submit gpt2 bpe nk100-gpt2-bpe)"
echo "qwen2_bpe=$(submit qwen2 bpe nk100-qwen2-bpe)"
