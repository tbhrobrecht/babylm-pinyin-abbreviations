#!/bin/bash
#SBATCH --job-name=nk-official-zho
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-eval/multilingual"
ENV_DIR="$ROOT/env/babylm-eval-official"

: "${MODEL_DIR:?Submit with --export=ALL,MODEL_DIR=/absolute/path/to/hf_model}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR is not a converted Hugging Face model: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  echo "Missing official evaluation environment: $ENV_DIR" >&2
  exit 1
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/logs/eval" "$HF_HOME" "$TMPDIR"

source "$ENV_DIR/bin/activate"
cd "$REPO"

# This corpus was trained without an explicitly prepended BOS token.
bash scripts/eval_model_full.sh \
  --model_name "$MODEL_DIR" \
  --langs "zho" \
  --bos_fix 0
