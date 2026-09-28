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

ROOT="$HOME/babylm-lrz"
REPO="$ROOT/code/babylm-pinyin-abbreviations"

: "${ARCH:?Submit with ARCH=gpt2 or ARCH=qwen2}"
: "${TOK:?Submit with TOK=hybrid or TOK=bpe}"

case "$ARCH" in
    gpt2)
        EXPECTED_PARAMETERS=97737216
        ARCH_ARGS=()
        ;;
    qwen2)
        # Keep grouped-query attention (12 query heads / 6 KV heads) while
        # matching the GPT2 model size as closely as practical.
        EXPECTED_PARAMETERS=97260288
        ARCH_ARGS=(--num-key-value-heads 6 --intermediate-size 2304)
        ;;
    *)
        echo "Unsupported ARCH=$ARCH; expected gpt2 or qwen2" >&2
        exit 2
        ;;
esac

case "$TOK" in
    hybrid)
        TRAIN_DATASET="data/datasets/nk_babylm_zho_train_hybrid.bin"
        VALID_DATASET="data/datasets/nk_babylm_zho_valid_hybrid.bin"
        TOKENIZER_NAME="nk_babylm_zho_hybrid"
        TOKENIZER_ARTIFACT="tokenizers/nk_babylm_zho_hybrid"
        ;;
    bpe)
        TRAIN_DATASET="data/datasets/nk_babylm_zho_train_spm.bin"
        VALID_DATASET="data/datasets/nk_babylm_zho_valid_spm.bin"
        TOKENIZER_NAME="nk_babylm_zho"
        TOKENIZER_ARTIFACT="tokenizers/nk_babylm_zho.model"
        ;;
    *)
        echo "Unsupported TOK=$TOK; expected hybrid or bpe" >&2
        exit 2
        ;;
esac

MODEL_DIR="models/nk_babylm_zho_${ARCH}_${TOK}_100m"
HF_DIR="hf_nk_babylm_zho_${ARCH}_${TOK}_100m"

cd "$REPO"

for required in "$TRAIN_DATASET" "$VALID_DATASET" "$TOKENIZER_ARTIFACT"; do
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

echo "========================================"
echo "Architecture:        $ARCH"
echo "Tokenizer:           $TOK"
echo "Training dataset:    $TRAIN_DATASET"
echo "Validation dataset:  $VALID_DATASET"
echo "Tokenizer name:      $TOKENIZER_NAME"
echo "Model output:        $MODEL_DIR"
echo "HF output:           $HF_DIR"
echo "Layers / heads:      12 / 12"
echo "Hidden size:         768"
echo "Expected parameters: $EXPECTED_PARAMETERS"
echo "Node:                $(hostname)"
echo "========================================"

nvidia-smi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

python scripts/train_babylm_from_scratch.py \
  --model-name nk_babylm_zho_100m \
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
  --start-at train \
  --stop-after convert
