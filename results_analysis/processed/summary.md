# Mandarin BabyLM model comparison

## Snapshot integrity

- Verified `46` downloaded source files against `SOURCE_MANIFEST.sha256`.
- Parsed `820` scored metric rows and `25` directly comparable primary metrics.
- Hidden tasks are listed as `predictions_only`; no hidden score is invented or treated as zero.
- CogBench reports show `fast=False`, indicating the full configured runs.

## Model metadata

| Model | Architecture | Tokenizer | Stored tensor parameters | Layers | Hidden | Vocab |
| --- | --- | --- | --- | --- | --- | --- |
| GPT2 hybrid | GPT2 | Hybrid | 33,674,240 | 8 | 512 | 16000 |
| Qwen2 hybrid | Qwen2 | Hybrid | 31,408,640 | 8 | 512 | 16000 |
| GPT2 BPE | GPT2 | BPE | 33,674,240 | 8 | 512 | 16000 |
| Qwen2 BPE | Qwen2 | BPE | 31,408,640 | 8 | 512 | 16000 |

`Stored tensor parameters` is counted directly from the safetensors header. It is not an estimate based on file size.

## Official Chinese results

| Model | Unweighted macro-average | Task wins (ties count for each) |
| --- | --- | --- |
| GPT2 hybrid | 0.4446 | 7 |
| Qwen2 hybrid | 0.4422 | 5 |
| GPT2 BPE | 0.4192 | 4 |
| Qwen2 BPE | 0.4201 | 2 |

The macro-average is descriptive: tasks differ in size, difficulty, and variance.

### Mean paired effects across the 16 official tasks

| Contrast | Mean score difference |
| --- | --- |
| GPT2: Hybrid - BPE | +0.0254 |
| Qwen2: Hybrid - BPE | +0.0222 |
| Hybrid: Qwen2 - GPT2 | -0.0024 |
| BPE: Qwen2 - GPT2 | +0.0009 |
| Tokenizer x architecture interaction | -0.0032 |

Positive tokenizer effects favor Hybrid. Positive architecture effects favor Qwen2.

### Task-level official scores

| Task | GPT2 hybrid | Qwen2 hybrid | GPT2 BPE | Qwen2 BPE | Winner |
| --- | --- | --- | --- | --- | --- |
| arc | 0.2562 | 0.2604 | 0.2646 | 0.2542 | GPT2 BPE |
| belebele | 0.2557 | 0.2045 | 0.1989 | 0.2102 | GPT2 hybrid |
| bmlama | 0.1730 | 0.1589 | 0.1672 | 0.1623 | GPT2 hybrid |
| global_piqa_nonparallel_zh | 0.5300 | 0.5000 | 0.5300 | 0.4800 | GPT2 hybrid; GPT2 BPE |
| global_piqa_parallel_zh | 0.2233 | 0.2621 | 0.2330 | 0.2136 | Qwen2 hybrid |
| hellaswag_zh_mubench | 0.2663 | 0.2671 | 0.2683 | 0.2673 | GPT2 BPE |
| include | 0.2589 | 0.2411 | 0.2589 | 0.2679 | Qwen2 BPE |
| mnli | 0.5236 | 0.5546 | 0.5512 | 0.5366 | Qwen2 hybrid |
| pos | 0.8446 | 0.8435 | 0.8112 | 0.8212 | GPT2 hybrid |
| sib200 | 0.7700 | 0.7700 | 0.6400 | 0.7050 | GPT2 hybrid; Qwen2 hybrid |
| truthfulqa | 0.2679 | 0.2411 | 0.2500 | 0.2321 | GPT2 hybrid |
| winogrande_zh_mubench | 0.5004 | 0.5037 | 0.5054 | 0.4922 | GPT2 BPE |
| xcomps_zh | 0.5408 | 0.5364 | 0.5109 | 0.5117 | GPT2 hybrid |
| xnli | 0.4510 | 0.4630 | 0.4730 | 0.4905 | Qwen2 BPE |
| xstorycloze_zh_mubench | 0.5101 | 0.5108 | 0.4497 | 0.4574 | Qwen2 hybrid |
| zhoblimp | 0.7423 | 0.7586 | 0.5948 | 0.6187 | Qwen2 hybrid |

## Chinese-pipeline primary metrics

### Zero-shot average accuracy

| Task | GPT2 hybrid | Qwen2 hybrid | GPT2 BPE | Qwen2 BPE | Winner |
| --- | --- | --- | --- | --- | --- |
| hanzi_pinyin | 0.9745 | 0.9825 | 0.7815 | 0.7660 | Qwen2 hybrid |
| hanzi_structure | 0.5460 | 0.5635 | 0.5510 | 0.5620 | Qwen2 hybrid |
| zhoblimp | 0.7193 | 0.7586 | 0.6044 | 0.6230 | Qwen2 hybrid |

### Fine-tuned accuracy

| Task | GPT2 hybrid | Qwen2 hybrid | GPT2 BPE | Qwen2 BPE | Winner |
| --- | --- | --- | --- | --- | --- |
| afqmc | 0.6895 | 0.6879 | 0.6918 | 0.6902 | GPT2 BPE |
| cluewsc2020 | 0.6349 | 0.6316 | 0.6349 | 0.6316 | GPT2 hybrid; GPT2 BPE |
| ocnli | 0.6156 | 0.6366 | 0.6017 | 0.6115 | Qwen2 hybrid |
| tnews | 0.5264 | 0.5327 | 0.4865 | 0.5103 | Qwen2 hybrid |

F1 and MCC are preserved in `metrics_long.csv` as supplementary metrics; they are not mixed into the accuracy comparison.

## CogBench aggregates

| Task | GPT2 hybrid | Qwen2 hybrid | GPT2 BPE | Qwen2 BPE | Winner |
| --- | --- | --- | --- | --- | --- |
| fmri | 0.0900 | 0.0852 | 0.0791 | 0.0803 | GPT2 hybrid |
| word_fmri | 0.5573 | 0.5535 | 0.5471 | 0.5462 | GPT2 hybrid |

The original CogBench logs emitted ill-conditioned ridge-regression warnings. Outputs were complete, but small differences may be numerically sensitive.

## Hidden tasks

The snapshot records `12` model/task prediction manifests covering `hanzi_pinyin`, `hanzi_structure`, and `meco_l1`.
Authoritative scores require the evaluation server and are intentionally absent here.

## Files produced

- `metrics_long.csv`: every parsed scored metric, including supplementary metrics and zero-shot subsections.
- `primary_scores_wide.csv`: one comparable row per primary task metric.
- `pairwise_effects.csv`: tokenizer, architecture, and interaction contrasts.
- `cogbench_detail.csv`: subject/region-level CogBench values.
- `model_metadata.csv`: exact exported-model metadata.
- `hidden_tasks.csv`: predictions-only task coverage and integrity counts.

## Interpretation cautions

1. Do not combine accuracy, F1, MCC, and fMRI correlations into a single inferential statistic.
2. The descriptive macro-average weights tasks equally, not examples equally.
3. No confidence intervals are available from aggregate-only files; example-level predictions would be needed for bootstrap testing.
4. Differences between architectures and tokenizers are based on one trained model per condition, so they should not be presented as seed-robust effects.
