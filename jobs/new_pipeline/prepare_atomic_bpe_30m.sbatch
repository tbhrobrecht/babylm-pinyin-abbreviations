#!/bin/bash
#SBATCH --job-name=prep-atomic-bpe
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

cd "$REPO"
source "$ROOT/env/babylm/bin/activate"
source "$ROOT/env/paths.sh"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

for tokenizer_kind in atomic_bpe_within atomic_bpe_cross; do
    python scripts/train_babylm_from_scratch.py \
      --model-name "nk_babylm_zho_${tokenizer_kind}" \
      --corpus-name nk_babylm_zho \
      --architecture gpt2 \
      --tokenizer-kind "$tokenizer_kind" \
      --tokenizer-name "nk_babylm_zho_${tokenizer_kind}" \
      --processed-output data/processed/nk_babylm_zho.txt \
      --train-dataset "data/datasets/nk_babylm_zho_train_${tokenizer_kind}.bin" \
      --validation-dataset "data/datasets/nk_babylm_zho_valid_${tokenizer_kind}.bin" \
      --vocab-size 16000 \
      --block-size 512 \
      --stride 512 \
      --preprocess-workers "$SLURM_CPUS_PER_TASK" \
      --hybrid-permissive \
      --start-at tokenizer \
      --stop-after dataset \
      --resume
done
