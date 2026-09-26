#!/bin/bash
#SBATCH --job-name=nk-zh-pipeline
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
REPO="$ROOT/code/chinese-babylm-eval-pipeline"
PYTHON="$ROOT/env/chinese-babylm-eval/bin/python"
EVAL_DIR="${EVAL_DIR:-$ROOT/cache/evaluation/chinese_pipeline}"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/eval/chinese_pipeline}"
TASKS="${TASKS:-all}"

: "${MODEL_DIR:?Submit with --export=ALL,MODEL_DIR=/absolute/path/to/hf_model}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "MODEL_DIR is not a converted Hugging Face model: $MODEL_DIR" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing evaluation environment: $PYTHON" >&2
  exit 1
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/logs/eval" "$RESULTS_DIR" "$HF_HOME" "$TMPDIR"

CONFIG_FILE="$(mktemp "$TMPDIR/nk-zh-eval.XXXXXX.yaml")"
trap 'rm -f "$CONFIG_FILE"' EXIT

cat >"$CONFIG_FILE" <<EOF
models:
  - path: $MODEL_DIR
    backend: causal
tasks:
  zero_shot:
    - zhoblimp
    - hanzi_structure
    - hanzi_pinyin
  cogbench:
    - word_fmri
    - fmri
  finetune:
    - afqmc
    - ocnli
    - tnews
    - cluewsc2020
eval_dir: $EVAL_DIR
results_dir: $RESULTS_DIR
save_item_with_unk: true
finetune_hparams:
  lr: 3.0e-5
  batch_size: 32
  max_epochs: 10
  sequence_length: 128
  seed: 42
  task_overrides:
    cluewsc2020:
      max_epochs: 30
EOF

cd "$REPO"
if [[ "$TASKS" == "all" ]]; then
  TASK_ARGS=(zhoblimp hanzi_structure hanzi_pinyin word_fmri fmri afqmc ocnli tnews cluewsc2020)
else
  read -r -a TASK_ARGS <<< "$TASKS"
fi

echo "Model: $MODEL_DIR"
echo "Tasks: ${TASK_ARGS[*]}"
"$PYTHON" pipeline.py eval \
  --config "$CONFIG_FILE" \
  --tasks "${TASK_ARGS[@]}"

"$PYTHON" pipeline.py gather --config "$CONFIG_FILE"
