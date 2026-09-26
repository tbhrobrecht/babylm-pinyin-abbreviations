#!/bin/bash
#SBATCH --job-name=smoke-nk-exports
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"
PYTHON="${PYTHON:-$ROOT/env/babylm/bin/python}"

export HF_HOME="$ROOT/cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TMPDIR="$ROOT/tmp"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ROOT/logs/eval" "$HF_HOME" "$TMPDIR"
cd "$REPO"

"$PYTHON" - <<'PY'
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root = Path.home() / "babylm-lrz/code/babylm-pinyin-abbreviations"
model_names = [
    "hf_nk_babylm_zho_gpt2_hybrid",
    "hf_nk_babylm_zho_qwen2_hybrid",
    "hf_nk_babylm_zho_gpt2_bpe",
    "hf_nk_babylm_zho_qwen2_bpe",
]

for name in model_names:
    path = root / name
    print(f"\n=== {name} ===", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
    )
    model.eval()

    encoded = tokenizer("今天天气很好。", return_tensors="pt")
    with torch.inference_mode():
        output = model(**encoded)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    print("tokenizer_class:", type(tokenizer).__name__)
    print("model_class:", type(model).__name__)
    print("vocab_size:", len(tokenizer))
    print("input_shape:", tuple(encoded["input_ids"].shape))
    print("logits_shape:", tuple(output.logits.shape))
    print("parameters:", parameters)
    print("finite_logits:", bool(torch.isfinite(output.logits).all()))

    del output, encoded, model, tokenizer
    gc.collect()

print("\nALL_EXPORTS_LOAD_OK")
PY
