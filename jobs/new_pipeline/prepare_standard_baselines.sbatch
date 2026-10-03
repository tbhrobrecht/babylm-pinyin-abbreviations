#!/bin/bash
#SBATCH --job-name=prep-standard-baselines
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.out
#SBATCH --error=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"
RAW_CORPUS="data/nk_babylm_zho.jsonl"

cd "$REPO"
source "$ROOT/env/babylm/bin/activate"
source "$ROOT/env/paths.sh"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

if [[ ! -s "$RAW_CORPUS" ]]; then
    echo "Required non-empty raw corpus does not exist: $RAW_CORPUS" >&2
    exit 2
fi

for baseline in hanzi full-pinyin; do
    artifact_name="${baseline//-/_}"
    processed="data/processed/nk_babylm_zho_${artifact_name}.txt"
    tokenizer_name="nk_babylm_zho_${artifact_name}_bpe"
    train_dataset="data/datasets/nk_babylm_zho_${artifact_name}_train_spm.bin"
    valid_dataset="data/datasets/nk_babylm_zho_${artifact_name}_valid_spm.bin"

    # SentencePiece enforces max_sentence_length on UTF-8 bytes, whereas the
    # repository splitter counts Python characters. Hanzi can require three
    # bytes per character, so use a conservative character ceiling there.
    # Full pinyin is ASCII apart from rare u-umlaut characters.
    if [[ "$baseline" == "hanzi" ]]; then
        split_max_chars=30000
    else
        split_max_chars=60000
    fi

    python scripts/train_babylm_from_scratch.py \
      --model-name "nk_babylm_zho_${artifact_name}_bpe" \
      --corpus-name nk_babylm_zho \
      --architecture gpt2 \
      --tokenizer-kind bpe \
      --transliteration "$baseline" \
      --raw-output "$RAW_CORPUS" \
      --processed-output "$processed" \
      --tokenizer-name "$tokenizer_name" \
      --train-dataset "$train_dataset" \
      --validation-dataset "$valid_dataset" \
      --vocab-size 16000 \
      --sentencepiece-split-max-chars "$split_max_chars" \
      --block-size 512 \
      --stride 512 \
      --preprocess-workers "$SLURM_CPUS_PER_TASK" \
      --start-at preprocess \
      --stop-after dataset \
      --resume

    for required in \
        "$processed" \
        "tokenizers/${tokenizer_name}.model" \
        "$train_dataset" \
        "$valid_dataset"; do
        if [[ ! -s "$required" ]]; then
            echo "Baseline preparation did not create required artifact: $required" >&2
            exit 1
        fi
    done
done

echo "Prepared independent 16k SentencePiece BPE corpora and datasets for Hanzi and full pinyin."
