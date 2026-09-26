#!/bin/bash
#SBATCH --job-name=setup-zh-eval
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/chinese-babylm-eval-pipeline"
ENV_DIR="$ROOT/env/chinese-babylm-eval"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

export UV_CACHE_DIR="$ROOT/cache/uv"
export TMPDIR="$ROOT/tmp"

mkdir -p "$ROOT/logs/eval" "$UV_CACHE_DIR" "$TMPDIR"

if [[ ! -d "$REPO/.git" ]]; then
  echo "Missing evaluation repository: $REPO" >&2
  echo "Clone https://github.com/SiyuanSong2004/chinese-babylm-eval-pipeline there first." >&2
  exit 1
fi

if [[ ! -x "$UV_BIN" ]]; then
  echo "uv is not executable at $UV_BIN" >&2
  exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$UV_BIN" venv --python 3.11 "$ENV_DIR"
fi

"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  --requirements "$REPO/requirements.txt"

# Required by the local PinyinCode tokenizer, but not declared upstream.
"$UV_BIN" pip install \
  --python "$ENV_DIR/bin/python" \
  "sentencepiece==0.2.1" \
  "jieba==0.42.1" \
  "pypinyin==0.55.0" \
  "accelerate==1.10.1"

cd "$REPO"
"$ENV_DIR/bin/python" - <<'PY'
import datasets
import h5py
import sklearn
import torch
import transformers
import yaml

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("datasets", datasets.__version__)
print("h5py", h5py.__version__)
print("sklearn", sklearn.__version__)
print("yaml", yaml.__version__)
PY

"$ENV_DIR/bin/python" pipeline.py --help
"$ENV_DIR/bin/python" -m evaluation_pipeline.sentence_zero_shot.run --help >/dev/null

echo "Chinese evaluation environment is ready: $ENV_DIR"
