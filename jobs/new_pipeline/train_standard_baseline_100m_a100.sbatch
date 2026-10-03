#!/bin/bash
#SBATCH --partition=lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.out
#SBATCH --error=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/new_pipeline/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"

: "${BASELINE:?Submit with BASELINE=hanzi or BASELINE=full-pinyin}"
case "$BASELINE" in
    hanzi|full-pinyin) ;;
    *)
        echo "Unsupported BASELINE=$BASELINE; expected hanzi or full-pinyin" >&2
        exit 2
        ;;
esac

ARTIFACT_NAME="${BASELINE//-/_}"
TOKENIZER_NAME="nk_babylm_zho_${ARTIFACT_NAME}_bpe"
TOKENIZER_ARTIFACT="tokenizers/${TOKENIZER_NAME}.model"
TRAIN_DATASET="data/datasets/nk_babylm_zho_${ARTIFACT_NAME}_train_spm.bin"
VALID_DATASET="data/datasets/nk_babylm_zho_${ARTIFACT_NAME}_valid_spm.bin"
MODEL_DIR="models/nk_babylm_zho_gpt2_${ARTIFACT_NAME}_bpe_100m"
HF_DIR="artifacts/models/100m/hf_nk_babylm_zho_gpt2_${ARTIFACT_NAME}_bpe_100m"

cd "$REPO"
mkdir -p "$(dirname "$HF_DIR")"

for required in "$TOKENIZER_ARTIFACT" "$TRAIN_DATASET" "$VALID_DATASET"; do
    if [[ ! -s "$required" ]]; then
        echo "Required non-empty input does not exist: $required" >&2
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

echo "baseline=$BASELINE architecture=gpt2 tokenizer=bpe expected_parameters=97737216"
echo "model_output=$MODEL_DIR hf_output=$HF_DIR"
nvidia-smi

python scripts/train_babylm_from_scratch.py \
  --model-name "nk_babylm_zho_gpt2_${ARTIFACT_NAME}_bpe_100m" \
  --corpus-name nk_babylm_zho \
  --architecture gpt2 \
  --tokenizer-kind bpe \
  --transliteration "$BASELINE" \
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
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 3e-4 \
  --num-workers 4 \
  --amp \
  --amp-dtype bfloat16 \
  --tf32 \
  --fused-adamw \
  --babylm-checkpoint-targets 100000000 \
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

echo "Validated baseline checkpoint and Transformers export: $HF_DIR"
