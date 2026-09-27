#!/bin/bash
#SBATCH --job-name=nk-pos-only
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
EVAL_REPO="$ROOT/code/babylm-eval/multilingual"
ENV_DIR="$ROOT/env/babylm-eval-official"

: "${MODEL_DIR:?Submit with --export=ALL,MODEL_DIR=/absolute/path/to/hf_model}"
model_basename="$(basename "$MODEL_DIR")"
output_dir="$EVAL_REPO/finetune/results/$model_basename/pos/zh"

export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$ROOT/cache/evaluation/official_datasets_v4"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_MODULES_CACHE="$ROOT/cache/huggingface/modules_offsets_v2"
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled

mkdir -p "$ROOT/logs/eval" "$output_dir" "$TMPDIR"
source "$ENV_DIR/bin/activate"
cd "$EVAL_REPO"

python finetune/finetune_token_classification.py \
  --model_name_or_path "$MODEL_DIR" \
  --language zh \
  --output_dir "$output_dir" \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 3.0e-5 \
  --num_train_epochs 10 \
  --patience 3 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --overwrite_output_dir \
  --seed 42

test -s "$output_dir/eval_results.json"
test -s "$output_dir/predictions.txt"
echo "Validated POS artifacts for $model_basename."
