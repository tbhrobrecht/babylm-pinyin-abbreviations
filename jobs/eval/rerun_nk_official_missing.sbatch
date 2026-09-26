#!/bin/bash
#SBATCH --job-name=nk-off-missing
#SBATCH --partition=lrz-v100x2
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -u

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-eval/multilingual"
ENV_DIR="$ROOT/env/babylm-eval-official"

: "${MODEL_DIR:?Submit with --export=ALL,MODEL_DIR=/absolute/path/to/hf_model}"

export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false

source "$ENV_DIR/bin/activate"
cd "$REPO"

status=0
bash scripts/meco_model.sh \
  --model_name "$MODEL_DIR" \
  --langs "zho" \
  --revision main || status=1

bash scripts/finetune_model.sh \
  --model_name "$MODEL_DIR" \
  --langs "zho" || status=1

python scripts/collate_results.py \
  --model_name "$MODEL_DIR" \
  --fast || status=1

exit "$status"
