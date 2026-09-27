#!/bin/bash
#SBATCH --job-name=nk-meco
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-eval/multilingual"
ENV_DIR="$ROOT/env/babylm-eval-official"

: "${MODEL_DIR:?Submit with --export=ALL,MODEL_DIR=/absolute/path/to/hf_model}"
[[ -f "$MODEL_DIR/config.json" ]] || {
  echo "MODEL_DIR is not a converted Hugging Face model: $MODEL_DIR" >&2
  exit 1
}

export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/cache/evaluation/official_datasets_v4"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$HF_HOME/modules_offsets_v2"
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled

mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_MODULES_CACHE" "$TMPDIR"

source "$ENV_DIR/bin/activate"
cd "$REPO"

bash scripts/meco_model.sh \
  --model_name "$MODEL_DIR" \
  --langs "zho" \
  --revision main

python scripts/collate_results.py \
  --model_name "$MODEL_DIR" \
  --fast
