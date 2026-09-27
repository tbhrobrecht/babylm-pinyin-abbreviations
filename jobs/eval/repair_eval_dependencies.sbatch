#!/bin/bash
#SBATCH --job-name=repair-eval-envs
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"
OFFICIAL_PYTHON="$ROOT/env/babylm-eval-official/bin/python"
CHINESE_PYTHON="$ROOT/env/chinese-babylm-eval/bin/python"
EVAL_DIR="$ROOT/cache/evaluation/chinese_pipeline"
OFFICIAL_REPO="$ROOT/code/babylm-eval/multilingual"
OFFICIAL_DATASETS_CACHE="$ROOT/cache/evaluation/official_datasets_v4"
UPOS_PATCH="$ROOT/code/babylm-pinyin-abbreviations/jobs/eval/patches/babylm_eval_datasets4_upos.patch"

export UV_CACHE_DIR="$ROOT/cache/uv"
export TMPDIR="$ROOT/tmp"
export HF_HOME="$ROOT/cache/huggingface"
export HF_DATASETS_CACHE="$OFFICIAL_DATASETS_CACHE"
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p \
  "$ROOT/logs/eval" \
  "$UV_CACHE_DIR" \
  "$TMPDIR" \
  "$HF_DATASETS_CACHE" \
  "$HF_HUB_CACHE"

"$UV_BIN" pip install --python "$OFFICIAL_PYTHON" \
  "transformers==4.51.3" \
  "tokenizers==0.21.1" \
  "pyreadr==0.5.6" \
  "datasets>=4.0,<5" \
  "wordfreq==3.1.1" \
  "minicons==0.3.38" \
  "requests>=2.32.3,<3" \
  "chardet>=5,<6"

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

# datasets 4.x represents UD `upos` tags as plain values instead of ClassLabel.
# Apply the evaluator compatibility patch once, while accepting an already
# patched checkout on subsequent repair runs.
if patch --dry-run --forward --strip=1 --directory="$ROOT/code/babylm-eval" \
    --input="$UPOS_PATCH" >/dev/null 2>&1; then
  patch --forward --strip=1 --directory="$ROOT/code/babylm-eval" \
    --input="$UPOS_PATCH"
elif patch --dry-run --reverse --strip=1 --directory="$ROOT/code/babylm-eval" \
    --input="$UPOS_PATCH" >/dev/null 2>&1; then
  echo "BabyLM evaluation datasets-v4 UPOS patch already applied."
else
  echo "Cannot apply or verify UPOS compatibility patch: $UPOS_PATCH" >&2
  exit 1
fi

"$OFFICIAL_PYTHON" - <<'PY'
import datasets
from importlib.metadata import version
import pyreadr
import tokenizers
import transformers
from transformers import TrainingArguments

TrainingArguments(output_dir="/tmp/babylm-argument-check", overwrite_output_dir=True)
major = int(datasets.__version__.split(".", 1)[0])
assert major == 4, datasets.__version__
print("official datasets", datasets.__version__)
print("official transformers", transformers.__version__)
print("official tokenizers", tokenizers.__version__)
print("official pyreadr", pyreadr.__version__)
print("official wordfreq", version("wordfreq"))
print("official minicons", version("minicons"))
PY

# Populate the new datasets-v4 cache once on CPU. This avoids both the old
# datasets-3 metadata and parallel .incomplete-file races in the GPU jobs.
cd "$OFFICIAL_REPO"
"$OFFICIAL_PYTHON" - <<'PY'
from pathlib import Path

from datasets import load_dataset

root = Path.cwd()
for task in ("arc", "belebele", "bmlama", "include", "mnli", "sib200", "truthfulqa", "xnli"):
    task_dir = root / "finetune" / "data" / "multilingual" / "zh" / task
    train = task_dir / f"{task}_zh.train.jsonl"
    validation = task_dir / f"{task}_zh.valid.jsonl"
    if not train.is_file() or not validation.is_file():
        raise FileNotFoundError(f"Missing local finetune data for {task}: {task_dir}")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train), "validation": str(validation)},
    )
    print("prefetched", task, {split: len(rows) for split, rows in dataset.items()})

ud = load_dataset(
    "universal-dependencies/universal_dependencies",
    "zh_gsdsimp",
    trust_remote_code=True,
)
print("prefetched zh_gsdsimp", {split: len(rows) for split, rows in ud.items()})
PY

"$OFFICIAL_PYTHON" -m py_compile \
  "$OFFICIAL_REPO/finetune/finetune_token_classification.py"

"$CHINESE_PYTHON" - <<'PY'
import accelerate
import transformers

print("chinese accelerate", accelerate.__version__)
print("chinese transformers", transformers.__version__)
PY

echo "Evaluation environments and CogBench layout repaired."
