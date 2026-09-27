#!/bin/bash
#SBATCH --job-name=nk-eval-finalize
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval/%x-%j.out
#SBATCH --error=logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
MODEL_REPO="$ROOT/code/babylm-pinyin-abbreviations"
EVAL_REPO="$ROOT/code/babylm-eval/multilingual"
ENV_DIR="$ROOT/env/babylm-eval-official"
OUT_DIR="$ROOT/results/eval/official"
CHINESE_RESULTS="$ROOT/results/eval/chinese_pipeline"

models=(
  hf_nk_babylm_zho_gpt2_hybrid
  hf_nk_babylm_zho_qwen2_hybrid
  hf_nk_babylm_zho_gpt2_bpe
  hf_nk_babylm_zho_qwen2_bpe
)

mkdir -p "$OUT_DIR"

# Preserve the valid BPE POS runs and expose them at the language-specific path
# expected by collate_results.py.
for model in hf_nk_babylm_zho_gpt2_bpe hf_nk_babylm_zho_qwen2_bpe; do
  source_dir="$EVAL_REPO/finetune/results/$model/pos"
  target_dir="$source_dir/zh"
  test -s "$source_dir/eval_results.json"
  test -s "$source_dir/predictions.txt"
  mkdir -p "$target_dir"
  install -m 0644 "$source_dir/eval_results.json" "$target_dir/eval_results.json"
  install -m 0644 "$source_dir/predictions.txt" "$target_dir/predictions.txt"
done

# Refuse to create apparently complete submissions if any requested Chinese
# pipeline task or official POS artifact is still absent.
for model in "${models[@]}"; do
  test -s "$EVAL_REPO/finetune/results/$model/pos/zh/eval_results.json"
  test -s "$EVAL_REPO/finetune/results/$model/pos/zh/predictions.txt"
  base="$CHINESE_RESULTS/$model/main"
  for task_dir in \
    zero_shot/causal/zhoblimp \
    zero_shot/causal/hanzi_structure \
    zero_shot/causal/hanzi_pinyin \
    cogbench/word_fmri \
    cogbench/fmri \
    finetune/afqmc \
    finetune/ocnli \
    finetune/tnews \
    finetune/cluewsc2020; do
    path="$base/$task_dir"
    [[ -n "$(find -L "$path" -type f -size +0c -print -quit 2>/dev/null)" ]] || {
      echo "Missing Chinese-pipeline result: $model/$task_dir" >&2
      exit 1
    }
  done
done

source "$ENV_DIR/bin/activate"
cd "$EVAL_REPO"
for model in "${models[@]}"; do
  python scripts/collate_results.py \
    --model_name "$model" \
    --output "$OUT_DIR/${model}_submission.json" \
    --output_predictions "$OUT_DIR/${model}_predictions.json"
  test -s "$OUT_DIR/${model}_submission.json"
  test -s "$OUT_DIR/${model}_predictions.json"
done

echo "Generated and validated four final submissions in $OUT_DIR."
