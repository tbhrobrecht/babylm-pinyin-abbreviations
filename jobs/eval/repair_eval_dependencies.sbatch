#!/bin/bash
#SBATCH --job-name=repair-eval-envs
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
OFFICIAL_PYTHON="$ROOT/env/babylm-eval-official/bin/python"
CHINESE_PYTHON="$ROOT/env/chinese-babylm-eval/bin/python"
EVAL_DIR="$ROOT/cache/evaluation/chinese_pipeline"

export UV_CACHE_DIR="$ROOT/cache/uv"
export TMPDIR="$ROOT/tmp"
mkdir -p "$ROOT/logs/eval" "$UV_CACHE_DIR" "$TMPDIR"

"$UV_BIN" pip install --python "$OFFICIAL_PYTHON" \
  "transformers==4.51.3" \
  "tokenizers==0.21.1" \
  "pyreadr==0.5.6"

"$UV_BIN" pip install --python "$CHINESE_PYTHON" \
  "accelerate==1.10.1"

COGBENCH_EXPECTED="$EVAL_DIR/cogbench-fmri-0415"
COGBENCH_EXTRACTED="$(dirname "$EVAL_DIR")/evaluation_data/cogbench-fmri-0415"
if [[ ! -e "$COGBENCH_EXPECTED" ]]; then
  [[ -d "$COGBENCH_EXTRACTED" ]] || {
    echo "Missing extracted CogBench data: $COGBENCH_EXTRACTED" >&2
    exit 1
  }
  ln -s "$COGBENCH_EXTRACTED" "$COGBENCH_EXPECTED"
fi

[[ -d "$COGBENCH_EXPECTED/train/word" ]]
[[ -d "$COGBENCH_EXPECTED/dev/story" ]]

"$OFFICIAL_PYTHON" - <<'PY'
import pyreadr
import tokenizers
import transformers
from transformers import TrainingArguments

TrainingArguments(output_dir="/tmp/babylm-argument-check", overwrite_output_dir=True)
print("official transformers", transformers.__version__)
print("official tokenizers", tokenizers.__version__)
print("official pyreadr", pyreadr.__version__)
PY

"$CHINESE_PYTHON" - <<'PY'
import accelerate
import transformers

print("chinese accelerate", accelerate.__version__)
print("chinese transformers", transformers.__version__)
PY

echo "Evaluation environments and CogBench layout repaired."
