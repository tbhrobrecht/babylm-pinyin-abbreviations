#!/bin/bash
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.out
#SBATCH --error=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"

: "${ARCH:?Submit with ARCH=gpt2 or ARCH=qwen2}"
: "${TOK:?Submit with TOK=atomic_bpe_within or TOK=atomic_bpe_cross}"

case "$ARCH" in
    gpt2)
        EXPECTED_PARAMETERS=97737216
        ARCH_ARGS=()
        ;;
    qwen2)
        EXPECTED_PARAMETERS=97260288
        ARCH_ARGS=(--num-key-value-heads 6 --intermediate-size 2304)
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
MODEL_DIR="models/nk_babylm_zho_${ARCH}_${TOK}_100m"
HF_DIR="artifacts/models/100m/hf_nk_babylm_zho_${ARCH}_${TOK}_100m"

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
echo "model_output=$MODEL_DIR hf_output=$HF_DIR"
nvidia-smi

python scripts/train_babylm_from_scratch.py \
  --model-name "nk_babylm_zho_${ARCH}_${TOK}_100m" \
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
  --n-layer 12 \
  --n-head 12 \
  --n-embd 768 \
  "${ARCH_ARGS[@]}" \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --preprocess-workers 8 \
  --num-workers 4 \
  --attn-implementation sdpa \
  --amp \
  --amp-dtype bfloat16 \
  --tf32 \
  --fused-adamw \
  --hybrid-permissive \
  --start-at train \
  --stop-after convert

for required in \
    "$MODEL_DIR/best.pt" \
    "$MODEL_DIR/last.pt" \
    "$MODEL_DIR/final.pt" \
    "$MODEL_DIR/metrics.jsonl" \
    "$HF_DIR/config.json" \
    "$HF_DIR/model.safetensors" \
    "$HF_DIR/tokenizer_config.json"; do
    if [[ ! -s "$required" ]]; then
        echo "Training completed without required artifact: $required" >&2
        exit 1
    fi
done

# Interval checkpoints are useful during training but redundant after the
# validated final/best/last checkpoints and Transformers export exist. Removing
# them here bounds quota use before the next serialized 100M job starts.
find "$MODEL_DIR" -maxdepth 1 -type f -name 'chck_*' -print -delete

echo "Validated 100M export and pruned redundant interval checkpoints: $HF_DIR"
