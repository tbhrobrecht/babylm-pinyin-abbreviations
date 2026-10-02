# Cross-task composite analysis

## Prespecified primary summary

Each task is converted to a within-task percentile rank across the registered models. Tasks are averaged within each evaluation suite, and the four suite means are then weighted equally. This avoids directly averaging accuracy, F1, MCC, and correlation.

| Rank | Model | Composite percentile | 95% task-resampling interval | Task percentile SD | Sensitivity rank range |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 100M GPT2 Atomic Within | 75.17 | 68.09–82.17 | 29.46 | 1–2 |
| 2 | 100M GPT2 Hybrid | 72.88 | 62.08–83.73 | 28.28 | 1–2 |
| 3 | 30M Qwen2 Hybrid | 61.94 | 54.96–69.10 | 22.94 | 3–5 |
| 4 | 30M GPT2 Hybrid | 59.97 | 54.29–66.01 | 25.53 | 4–7 |
| 5 | 100M GPT2 Atomic Cross | 59.86 | 51.28–68.98 | 29.53 | 4–9 |
| 6 | 30M Qwen2 Atomic Within | 58.72 | 48.61–68.87 | 26.99 | 3–6 |
| 7 | 100M Qwen2 Hybrid | 55.89 | 44.17–67.60 | 30.53 | 6–9 |
| 8 | 100M Qwen2 Atomic Within | 55.07 | 43.04–66.16 | 30.13 | 3–10 |
| 9 | 30M GPT2 Atomic Within | 53.44 | 46.20–60.42 | 26.59 | 8–9 |
| 10 | 30M GPT2 Atomic Cross | 50.90 | 39.18–63.65 | 31.42 | 7–10 |
| 11 | 30M Qwen2 Atomic Cross | 45.09 | 38.91–51.53 | 23.65 | 11–11 |
| 12 | 100M Qwen2 Atomic Cross | 44.29 | 31.13–57.45 | 26.56 | 12–12 |
| 13 | 30M Qwen2 BPE | 30.10 | 21.42–39.86 | 21.75 | 13–15 |
| 14 | 30M GPT2 BPE | 28.06 | 16.61–39.91 | 29.67 | 14–15 |
| 15 | 100M GPT2 BPE | 26.68 | 16.27–37.26 | 32.76 | 13–15 |
| 16 | 100M Qwen2 BPE | 21.94 | 13.44–31.44 | 31.03 | 16–16 |

## Interpretation safeguards

- The composite is relative to the current 16-model comparison set; it is not an absolute accuracy score.
- The interval resamples observed tasks within suites. It measures sensitivity to task composition, not training-seed uncertainty or conventional statistical significance.
- Task percentile SD describes consistency of relative placement across tasks; lower is more consistent, not necessarily more accurate.
- One trained seed is available per condition, so small ordering differences must not be described as seed-robust effects.
- Rankings from all five aggregation definitions are retained in `composite_sensitivity.csv`; conclusions should emphasize models that remain strong across definitions.

## Reproducibility

The task bootstrap used `10000` paired replicates with seed `20261002`. See `composite_methodology.json` for the machine-readable specification.
