# Results analysis snapshot

This directory contains a compact, integrity-checked copy of the evaluation
reports produced on LRZ and a standard-library Python processor for comparing
the four Mandarin BabyLM models.

## Contents

- `raw_snapshot/`: original compact reports copied from LRZ.
- `raw_snapshot/SOURCE_MANIFEST.sha256`: hashes calculated on LRZ before transfer.
- `build_results_summary.py`: deterministic parser and comparison generator.
- `processed/`: generated readable CSV tables and `summary.md`.

Large raw prediction arrays and model weights are intentionally excluded. The
hidden-task manifest records their coverage, counts, and finite-value checks,
but hidden tasks are never assigned local scores.

## Rebuild

From the repository root:

```bash
python results_analysis/build_results_summary.py
```

To use different locations:

```bash
python results_analysis/build_results_summary.py \
  --input results_analysis/raw_snapshot \
  --output results_analysis/processed
```

The processor verifies every downloaded source file against the SHA-256
manifest before generating any result tables.

## Reading the outputs

- Start with `processed/summary.md` for the main findings and cautions.
- Use `processed/primary_scores_wide.csv` for direct four-model comparisons.
- Use `processed/pairwise_effects.csv` to isolate tokenizer and architecture effects.
- Use `processed/metrics_long.csv` for plotting or further analysis in Python/R.
- Use `processed/cogbench_detail.csv` for subject/region-level CogBench analysis.

The unweighted official macro-average is descriptive only. The evaluation tasks
have different sizes and variances, and one trained model is available per
condition, so small differences should not be interpreted as seed-robust effects.
