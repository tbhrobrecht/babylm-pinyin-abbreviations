#!/bin/bash
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.out
#SBATCH --error=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"

: "${ARCH:?Submit with ARCH=gpt2 or ARCH=qwen2}"
: "${TOK:?Submit with TOK=atomic_bpe_within or TOK=atomic_bpe_cross}"

case "$ARCH" in
    gpt2)
        EXPECTED_PARAMETERS=33674240
        ARCH_ARGS=()
        ;;
    qwen2)
        EXPECTED_PARAMETERS=31408640
        ARCH_ARGS=(--num-key-value-heads 4 --intermediate-size 1376)
        ;;
    *)
        echo "Unsupported ARCH=$ARCH; expected gpt2 or qwen2" >&2
        exit 2
        ;;
esac

case "$TOK" in
    atomic_bpe_within|atomic_bpe_cross) ;;
    *)
        echo "Unsupported TOK=$TOK; expected atomic_bpe_within or atomic_bpe_cross" >&2
        exit 2
        ;;
esac

TOKENIZER_NAME="nk_babylm_zho_${TOK}"
TRAIN_DATASET="data/datasets/nk_babylm_zho_train_${TOK}.bin"
VALID_DATASET="data/datasets/nk_babylm_zho_valid_${TOK}.bin"
MODEL_DIR="models/nk_babylm_zho_${ARCH}_${TOK}"
HF_DIR="artifacts/models/30m/hf_nk_babylm_zho_${ARCH}_${TOK}"

cd "$REPO"
mkdir -p "$(dirname "$HF_DIR")"

for required in \
    "tokenizers/$TOKENIZER_NAME" \
    "$TRAIN_DATASET" \
    "$VALID_DATASET"; do
    if [[ ! -e "$required" ]]; then
        echo "Required input does not exist: $required" >&2
        exit 2
    fi
done

if [[ -e "$MODEL_DIR" || -e "$HF_DIR" ]]; then
    echo "Refusing to overwrite an existing output: $MODEL_DIR or $HF_DIR" >&2
    exit 2
fi

source "$ROOT/env/babylm/bin/activate"
source "$ROOT/env/paths.sh"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "architecture=$ARCH tokenizer=$TOK expected_parameters=$EXPECTED_PARAMETERS"
nvidia-smi

python scripts/train_babylm_from_scratch.py \
  --model-name "nk_babylm_zho_${ARCH}_${TOK}" \
  --corpus-name nk_babylm_zho \
  --architecture "$ARCH" \
  --tokenizer-kind "$TOK" \
  --tokenizer-name "$TOKENIZER_NAME" \
  --train-dataset "$TRAIN_DATASET" \
  --validation-dataset "$VALID_DATASET" \
  --model-output-dir "$MODEL_DIR" \
  --hf-output-dir "$HF_DIR" \
  --device cuda \
  --vocab-size 16000 \
  --block-size 512 \
  --stride 512 \
  --n-layer 8 \
  --n-head 8 \
  --n-embd 512 \
  "${ARCH_ARGS[@]}" \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --num-workers 4 \
  --attn-implementation sdpa \
  --amp \
  --amp-dtype bfloat16 \
  --tf32 \
  --fused-adamw \
  --hybrid-permissive \
  --start-at train \
  --stop-after convert
