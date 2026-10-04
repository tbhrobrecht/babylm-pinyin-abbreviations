# Cross-task composite analysis

## Prespecified primary summary

Each task is converted to a within-task percentile rank across the registered models. Tasks are averaged within each evaluation suite, and the four suite means are then weighted equally. This avoids directly averaging accuracy, F1, MCC, and correlation.

| Rank | Model | Composite percentile | 95% task-resampling interval | Task percentile SD | Sensitivity rank range |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 100M GPT2 Atomic Within | 72.03 | 64.25–79.73 | 28.35 | 1–3 |
| 2 | 100M GPT2 Hanzi BPE | 71.48 | 54.84–87.30 | 31.42 | 1–2 |
| 3 | 100M GPT2 Hybrid | 70.53 | 59.87–81.14 | 27.63 | 2–3 |
| 4 | 30M Qwen2 Hybrid | 60.74 | 53.98–67.57 | 21.45 | 4–6 |
| 5 | 30M Qwen2 Atomic Within | 58.44 | 48.42–68.49 | 26.54 | 4–5 |
| 6 | 100M GPT2 Atomic Cross | 58.24 | 50.02–67.16 | 29.10 | 6–12 |
| 7 | 30M GPT2 Hybrid | 58.20 | 52.07–64.74 | 24.66 | 7–10 |
| 8 | 100M Qwen2 Hybrid | 54.70 | 44.30–65.18 | 28.32 | 7–10 |
| 9 | 100M Qwen2 Atomic Within | 54.24 | 42.42–64.78 | 27.78 | 5–11 |
| 10 | 30M GPT2 Atomic Within | 51.85 | 44.81–58.50 | 25.53 | 9–12 |
| 11 | 30M GPT2 Atomic Cross | 51.32 | 39.78–63.59 | 31.34 | 8–12 |
| 12 | 100M GPT2 Full-pinyin BPE | 47.00 | 39.34–54.40 | 33.49 | 7–12 |
| 13 | 30M Qwen2 Atomic Cross | 44.87 | 38.65–51.42 | 23.65 | 13–13 |
| 14 | 100M Qwen2 Atomic Cross | 42.98 | 30.62–55.13 | 25.82 | 14–14 |
| 15 | 30M Qwen2 BPE | 28.83 | 20.50–38.39 | 21.36 | 15–17 |
| 16 | 30M GPT2 BPE | 27.73 | 16.48–39.40 | 28.68 | 16–17 |
| 17 | 100M GPT2 BPE | 26.06 | 15.98–36.46 | 32.48 | 15–17 |
| 18 | 100M Qwen2 BPE | 20.77 | 12.91–29.55 | 30.50 | 18–18 |

## Interpretation safeguards

- The composite is relative to the current 18-model comparison set; it is not an absolute accuracy score.
- The interval resamples observed tasks within suites. It measures sensitivity to task composition, not training-seed uncertainty or conventional statistical significance.
- Task percentile SD describes consistency of relative placement across tasks; lower is more consistent, not necessarily more accurate.
- One trained seed is available per condition, so small ordering differences must not be described as seed-robust effects.
- Rankings from all five aggregation definitions are retained in `composite_sensitivity.csv`; conclusions should emphasize models that remain strong across definitions.

## Reproducibility

The task bootstrap used `10000` paired replicates with seed `20261002`. See `composite_methodology.json` for the machine-readable specification.
