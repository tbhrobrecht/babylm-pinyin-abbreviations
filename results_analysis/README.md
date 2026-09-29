# Results analysis snapshot

This directory contains an integrity-checked evaluation snapshot and a
standard-library Python processor for comparing all twelve Mandarin BabyLM
models: the original four 30M models, their four 100M counterparts, and the
four 30M within-word/cross-word atomic-BPE models.

## Contents

- `raw_snapshot/`: original compact reports copied from LRZ.
- `raw_snapshot/SOURCE_MANIFEST.sha256`: hashes calculated on LRZ before transfer.
- `models.json`: model registry and result/config locations. Add future models here.
- `build_results_summary.py`: deterministic parser and comparison generator.
- `processed/`: generated readable CSV tables and `summary.md`.

Large raw prediction arrays and model weights are excluded from the legacy
snapshot. Its hidden-task manifest remains available as archival coverage for
the original four models, but hidden tasks are never assigned invented scores.

## Rebuild

From the repository root:

```bash
python results_analysis/build_results_summary.py
```

To use a different registry or output location:

```bash
python results_analysis/build_results_summary.py \
  --registry results_analysis/models.json \
  --output results_analysis/processed
```

The processor verifies the legacy downloaded snapshot against its SHA-256
manifest and requires all registered models to contain the same 25 primary
metrics. New model and evaluation artifacts remain under gitignored `artifacts/`.

## Reading the outputs

- Start with `processed/summary.md` for the main findings and cautions.
- Use `processed/primary_scores_wide.csv` for direct twelve-model comparisons.
- Use `processed/suite_averages.csv` for per-suite descriptive averages.
- Use `processed/pairwise_effects.csv` for task-level size, tokenizer,
  boundary-policy, and architecture contrasts.
- Use `processed/contrast_summary.csv` for those contrasts averaged by suite.
- Use `processed/metrics_long.csv` for plotting or further analysis in Python/R.
- Use `processed/cogbench_detail.csv` for subject/region-level CogBench analysis.
- Use `processed/hidden_tasks.csv` for the original snapshot's prediction-only
  hidden-task coverage; it is not a scored twelve-model comparison.

All unweighted suite averages are descriptive only. The evaluation tasks have
different sizes and variances, and one trained model is available per condition,
so small differences should not be interpreted as seed-robust effects.
