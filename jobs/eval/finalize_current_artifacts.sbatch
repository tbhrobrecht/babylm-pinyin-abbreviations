#!/bin/bash
#SBATCH --job-name=finalize-artifacts
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/eval/%x-%j.out
#SBATCH --error=/dss/dsshome1/0D/go56yiq2/babylm-lrz/logs/eval/%x-%j.err

set -euo pipefail

ROOT="${BABYLM_LRZ_ROOT:-$HOME/babylm-lrz}"
REPO="$ROOT/code/babylm-pinyin-abbreviations"
EVAL_REPO="$ROOT/code/babylm-eval/multilingual"
ARTIFACTS="$REPO/artifacts"

models_100m=(
  hf_nk_babylm_zho_gpt2_hybrid_100m
  hf_nk_babylm_zho_qwen2_hybrid_100m
  hf_nk_babylm_zho_gpt2_bpe_100m
  hf_nk_babylm_zho_qwen2_bpe_100m
)
models_atomic=(
  hf_nk_babylm_zho_gpt2_atomic_bpe_within
  hf_nk_babylm_zho_qwen2_atomic_bpe_within
  hf_nk_babylm_zho_gpt2_atomic_bpe_cross
  hf_nk_babylm_zho_qwen2_atomic_bpe_cross
)

copy_official_results() {
  local destination="$1"
  shift
  local model source

  mkdir -p \
    "$destination/official/zeroshot" \
    "$destination/official/meco" \
    "$destination/official/finetune"

  for model in "$@"; do
    source="$(find "$EVAL_REPO/results/main" -mindepth 1 -maxdepth 1 -type d -name "*${model}" -print -quit)"
    test -n "$source"
    cp -a "$source" "$destination/official/zeroshot/"

    source="$(find "$EVAL_REPO/meco/results/main" -mindepth 1 -maxdepth 1 -type d -name "*${model}" -print -quit)"
    test -n "$source"
    cp -a "$source" "$destination/official/meco/"

    test -d "$EVAL_REPO/finetune/results/$model"
    cp -a "$EVAL_REPO/finetune/results/$model" "$destination/official/finetune/"
  done
}

mkdir -p \
  "$ARTIFACTS/evaluations/100m/chinese" \
  "$ARTIFACTS/evaluations/100m/logs" \
  "$ARTIFACTS/evaluations/30m_atomic/chinese" \
  "$ARTIFACTS/evaluations/30m_atomic/logs" \
  "$ARTIFACTS/downloads"

cp -a "$ROOT/results/eval/chinese_pipeline_100m" \
  "$ARTIFACTS/evaluations/100m/chinese/"
copy_official_results "$ARTIFACTS/evaluations/100m" "${models_100m[@]}"

cp -a "$ROOT/results/eval/chinese_pipeline_atomic_bpe_30m" \
  "$ARTIFACTS/evaluations/30m_atomic/chinese/"
copy_official_results "$ARTIFACTS/evaluations/30m_atomic" "${models_atomic[@]}"

find "$ROOT/logs/eval" "$REPO/logs/eval" -maxdepth 1 -type f \
  \( -name '*5815647*' -o -name '*5815648*' -o -name '*5815649*' \) \
  -exec cp -a {} "$ARTIFACTS/evaluations/100m/logs/" \;

find "$ROOT/logs/eval" "$REPO/logs/eval" -maxdepth 1 -type f \
  \( -name 'off30-*' -o -name 'zh30-*' -o -name 'pos30-*' \
     -o -name 'atomic_bpe_30m_eval_jobs_*.txt' \) \
  -exec cp -a {} "$ARTIFACTS/evaluations/30m_atomic/logs/" \;

archive_100m="$ARTIFACTS/downloads/babylm_100m_models_and_evaluations.tar.zst"
archive_atomic="$ARTIFACTS/downloads/babylm_30m_atomic_bpe_models_and_evaluations.tar.zst"
test ! -e "$archive_100m"
test ! -e "$archive_atomic"

tar --zstd -cf "$archive_100m.partial" -C "$ARTIFACTS" \
  models/100m evaluations/100m partial_checkpoints README.md
mv "$archive_100m.partial" "$archive_100m"
sha256sum "$archive_100m" | tee "$archive_100m.sha256"
tar --zstd -cf "$archive_atomic.partial" -C "$ARTIFACTS" \
  models/30m evaluations/30m_atomic README.md
mv "$archive_atomic.partial" "$archive_atomic"
sha256sum "$archive_atomic" | tee "$archive_atomic.sha256"

tar --zstd -tf "$archive_100m" >/dev/null
tar --zstd -tf "$archive_atomic" >/dev/null
du -sh "$ARTIFACTS/evaluations/100m" "$ARTIFACTS/evaluations/30m_atomic" \
  "$archive_100m" "$archive_atomic"
