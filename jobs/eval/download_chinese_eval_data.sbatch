#!/bin/bash
#SBATCH --job-name=download-zh-eval
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/chinese-babylm-eval-pipeline"
PYTHON="$ROOT/env/chinese-babylm-eval/bin/python"
EVAL_DIR="${EVAL_DIR:-$ROOT/cache/evaluation/chinese_pipeline}"

export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TMPDIR="$ROOT/tmp"

mkdir -p "$ROOT/logs/eval" "$EVAL_DIR" "$HF_HOME" "$TMPDIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing evaluation environment: $PYTHON" >&2
  exit 1
fi

cd "$REPO"

# TASKS may be "all" or a space-separated subset accepted by pipeline.py.
TASKS="${TASKS:-all}"
if [[ "$TASKS" == "all" ]]; then
  "$PYTHON" pipeline.py download --eval_dir "$EVAL_DIR"
else
  read -r -a TASK_ARGS <<< "$TASKS"
  "$PYTHON" pipeline.py download --eval_dir "$EVAL_DIR" --tasks "${TASK_ARGS[@]}"
fi

# Upstream CogBench currently extracts into a sibling `evaluation_data`
# directory instead of the requested eval directory. Preserve the extracted
# data and expose it at the path consumed by pipeline.py.
COGBENCH_EXPECTED="$EVAL_DIR/cogbench-fmri-0415"
COGBENCH_EXTRACTED="$(dirname "$EVAL_DIR")/evaluation_data/cogbench-fmri-0415"
if [[ ! -e "$COGBENCH_EXPECTED" && -d "$COGBENCH_EXTRACTED" ]]; then
  ln -s "$COGBENCH_EXTRACTED" "$COGBENCH_EXPECTED"
fi

if [[ ! -d "$COGBENCH_EXPECTED/train/word" || ! -d "$COGBENCH_EXPECTED/dev/story" ]]; then
  echo "CogBench data layout is incomplete at $COGBENCH_EXPECTED" >&2
  exit 1
fi

echo "Chinese evaluation data is ready: $EVAL_DIR"
