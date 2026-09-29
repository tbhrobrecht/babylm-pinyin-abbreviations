# Mandarin BabyLM 12-model comparison

## Validation and scope

- Compared `12` models across `25` comparable primary metrics.
- Parsed `2460` total metric rows, including supplementary metrics.
- Verified `46` legacy snapshot files against `SOURCE_MANIFEST.sha256`.
- Raw official outputs use evaluator collation rules: Global PIQA uses normalized accuracy; other zero-shot tasks use accuracy.
- Server-scored tasks are excluded from official local scores; Chinese-pipeline Hanzi scores remain a separate local suite.

## Model metadata

| Model | Scale | Architecture | Tokenizer | Stored parameters | Layers | Hidden | Vocab |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30M GPT2 Hybrid | 30M | GPT2 | Hybrid | 33,674,240 | 8 | 512 | 16000 |
| 30M Qwen2 Hybrid | 30M | Qwen2 | Hybrid | 31,408,640 | 8 | 512 | 16000 |
| 30M GPT2 BPE | 30M | GPT2 | BPE | 33,674,240 | 8 | 512 | 16000 |
| 30M Qwen2 BPE | 30M | Qwen2 | BPE | 31,408,640 | 8 | 512 | 16000 |
| 100M GPT2 Hybrid | 100M | GPT2 | Hybrid | 97,737,216 | 12 | 768 | 16000 |
| 100M Qwen2 Hybrid | 100M | Qwen2 | Hybrid | 97,260,288 | 12 | 768 | 16000 |
| 100M GPT2 BPE | 100M | GPT2 | BPE | 97,737,216 | 12 | 768 | 16000 |
| 100M Qwen2 BPE | 100M | Qwen2 | BPE | 97,260,288 | 12 | 768 | 16000 |
| 30M GPT2 Atomic Within | 30M | GPT2 | Atomic BPE within | 33,674,240 | 8 | 512 | 16000 |
| 30M Qwen2 Atomic Within | 30M | Qwen2 | Atomic BPE within | 31,408,640 | 8 | 512 | 16000 |
| 30M GPT2 Atomic Cross | 30M | GPT2 | Atomic BPE cross | 33,674,240 | 8 | 512 | 16000 |
| 30M Qwen2 Atomic Cross | 30M | Qwen2 | Atomic BPE cross | 31,408,640 | 8 | 512 | 16000 |

## Suite-level descriptive averages

| Model | Official (16) | Chinese zero-shot (3) | Chinese fine-tune (4) | CogBench (2) |
| --- | --- | --- | --- | --- |
| 30M GPT2 Hybrid | 0.4446 | 0.7466 | 0.6166 | 0.3236 |
| 30M Qwen2 Hybrid | 0.4422 | 0.7682 | 0.6222 | 0.3194 |
| 30M GPT2 BPE | 0.4192 | 0.6456 | 0.6037 | 0.3131 |
| 30M Qwen2 BPE | 0.4201 | 0.6503 | 0.6109 | 0.3133 |
| 100M GPT2 Hybrid | 0.4512 | 0.7628 | 0.6188 | 0.3236 |
| 100M Qwen2 Hybrid | 0.4373 | 0.7638 | 0.6170 | 0.3198 |
| 100M GPT2 BPE | 0.4237 | 0.6335 | 0.6101 | 0.3136 |
| 100M Qwen2 BPE | 0.4122 | 0.6343 | 0.6081 | 0.3135 |
| 30M GPT2 Atomic Within | 0.4461 | 0.7404 | 0.6181 | 0.3227 |
| 30M Qwen2 Atomic Within | 0.4493 | 0.7623 | 0.6161 | 0.3200 |
| 30M GPT2 Atomic Cross | 0.4403 | 0.7488 | 0.6085 | 0.3227 |
| 30M Qwen2 Atomic Cross | 0.4351 | 0.7588 | 0.6082 | 0.3201 |

These are unweighted descriptive averages within each suite; do not average across suites because their metrics and scales differ.

## Selected mean paired contrasts

| Suite | Contrast | Mean difference |
| --- | --- | --- |
| BabyLM official Chinese | 30M GPT2: Atomic cross - within | -0.0058 |
| BabyLM official Chinese | 30M Qwen2: Atomic cross - within | -0.0142 |
| BabyLM official Chinese | GPT2 BPE: 100M - 30M | +0.0046 |
| BabyLM official Chinese | GPT2 Hybrid: 100M - 30M | +0.0066 |
| BabyLM official Chinese | Qwen2 BPE: 100M - 30M | -0.0078 |
| BabyLM official Chinese | Qwen2 Hybrid: 100M - 30M | -0.0050 |
| Chinese fine-tune | 30M GPT2: Atomic cross - within | -0.0096 |
| Chinese fine-tune | 30M Qwen2: Atomic cross - within | -0.0079 |
| Chinese fine-tune | GPT2 BPE: 100M - 30M | +0.0064 |
| Chinese fine-tune | GPT2 Hybrid: 100M - 30M | +0.0022 |
| Chinese fine-tune | Qwen2 BPE: 100M - 30M | -0.0028 |
| Chinese fine-tune | Qwen2 Hybrid: 100M - 30M | -0.0052 |
| Chinese zero-shot | 30M GPT2: Atomic cross - within | +0.0083 |
| Chinese zero-shot | 30M Qwen2: Atomic cross - within | -0.0035 |
| Chinese zero-shot | GPT2 BPE: 100M - 30M | -0.0122 |
| Chinese zero-shot | GPT2 Hybrid: 100M - 30M | +0.0162 |
| Chinese zero-shot | Qwen2 BPE: 100M - 30M | -0.0161 |
| Chinese zero-shot | Qwen2 Hybrid: 100M - 30M | -0.0044 |
| CogBench | 30M GPT2: Atomic cross - within | -0.0000 |
| CogBench | 30M Qwen2: Atomic cross - within | +0.0001 |
| CogBench | GPT2 BPE: 100M - 30M | +0.0005 |
| CogBench | GPT2 Hybrid: 100M - 30M | -0.0000 |
| CogBench | Qwen2 BPE: 100M - 30M | +0.0003 |
| CogBench | Qwen2 Hybrid: 100M - 30M | +0.0005 |

Positive values favor the condition before the minus sign.

## Official task scores by cohort

### Original 30M

| Task | 30M GPT2 Hybrid | 30M Qwen2 Hybrid | 30M GPT2 BPE | 30M Qwen2 BPE | Best in cohort |
| --- | --- | --- | --- | --- | --- |
| arc | 0.2562 | 0.2604 | 0.2646 | 0.2542 | 30M GPT2 BPE |
| belebele | 0.2557 | 0.2045 | 0.1989 | 0.2102 | 30M GPT2 Hybrid |
| bmlama | 0.1730 | 0.1589 | 0.1672 | 0.1623 | 30M GPT2 Hybrid |
| global_piqa_nonparallel_zh | 0.5300 | 0.5000 | 0.5300 | 0.4800 | 30M GPT2 Hybrid; 30M GPT2 BPE |
| global_piqa_parallel_zh | 0.2233 | 0.2621 | 0.2330 | 0.2136 | 30M Qwen2 Hybrid |
| hellaswag_zh_mubench | 0.2663 | 0.2671 | 0.2683 | 0.2673 | 30M GPT2 BPE |
| include | 0.2589 | 0.2411 | 0.2589 | 0.2679 | 30M Qwen2 BPE |
| mnli | 0.5236 | 0.5546 | 0.5512 | 0.5366 | 30M Qwen2 Hybrid |
| pos | 0.8446 | 0.8435 | 0.8112 | 0.8212 | 30M GPT2 Hybrid |
| sib200 | 0.7700 | 0.7700 | 0.6400 | 0.7050 | 30M GPT2 Hybrid; 30M Qwen2 Hybrid |
| truthfulqa | 0.2679 | 0.2411 | 0.2500 | 0.2321 | 30M GPT2 Hybrid |
| winogrande_zh_mubench | 0.5004 | 0.5037 | 0.5054 | 0.4922 | 30M GPT2 BPE |
| xcomps_zh | 0.5408 | 0.5364 | 0.5109 | 0.5117 | 30M GPT2 Hybrid |
| xnli | 0.4510 | 0.4630 | 0.4730 | 0.4905 | 30M Qwen2 BPE |
| xstorycloze_zh_mubench | 0.5101 | 0.5108 | 0.4497 | 0.4574 | 30M Qwen2 Hybrid |
| zhoblimp | 0.7423 | 0.7586 | 0.5948 | 0.6187 | 30M Qwen2 Hybrid |

### 100M

| Task | 100M GPT2 Hybrid | 100M Qwen2 Hybrid | 100M GPT2 BPE | 100M Qwen2 BPE | Best in cohort |
| --- | --- | --- | --- | --- | --- |
| arc | 0.2729 | 0.2521 | 0.2458 | 0.2688 | 100M GPT2 Hybrid |
| belebele | 0.2102 | 0.2159 | 0.2045 | 0.1591 | 100M Qwen2 Hybrid |
| bmlama | 0.1664 | 0.1382 | 0.1589 | 0.1233 | 100M GPT2 Hybrid |
| global_piqa_nonparallel_zh | 0.4900 | 0.4700 | 0.5100 | 0.5600 | 100M Qwen2 BPE |
| global_piqa_parallel_zh | 0.2718 | 0.2621 | 0.2524 | 0.2039 | 100M GPT2 Hybrid |
| hellaswag_zh_mubench | 0.2735 | 0.2717 | 0.2706 | 0.2691 | 100M GPT2 Hybrid |
| include | 0.1607 | 0.2500 | 0.2857 | 0.2946 | 100M Qwen2 BPE |
| mnli | 0.5861 | 0.5619 | 0.5462 | 0.5343 | 100M GPT2 Hybrid |
| pos | 0.8525 | 0.8517 | 0.8217 | 0.8205 | 100M GPT2 Hybrid |
| sib200 | 0.8300 | 0.7350 | 0.6150 | 0.5800 | 100M GPT2 Hybrid |
| truthfulqa | 0.2946 | 0.1696 | 0.3304 | 0.2232 | 100M GPT2 BPE |
| winogrande_zh_mubench | 0.5037 | 0.4938 | 0.5136 | 0.5021 | 100M GPT2 BPE |
| xcomps_zh | 0.5374 | 0.5418 | 0.5107 | 0.5145 | 100M Qwen2 Hybrid |
| xnli | 0.4980 | 0.5060 | 0.4630 | 0.4830 | 100M Qwen2 Hybrid |
| xstorycloze_zh_mubench | 0.5201 | 0.5201 | 0.4489 | 0.4474 | 100M GPT2 Hybrid; 100M Qwen2 Hybrid |
| zhoblimp | 0.7513 | 0.7560 | 0.6024 | 0.6119 | 100M Qwen2 Hybrid |

### Atomic BPE 30M

| Task | 30M GPT2 Atomic Within | 30M Qwen2 Atomic Within | 30M GPT2 Atomic Cross | 30M Qwen2 Atomic Cross | Best in cohort |
| --- | --- | --- | --- | --- | --- |
| arc | 0.2438 | 0.2583 | 0.2479 | 0.2396 | 30M Qwen2 Atomic Within |
| belebele | 0.2386 | 0.2614 | 0.2557 | 0.2216 | 30M Qwen2 Atomic Within |
| bmlama | 0.1697 | 0.1854 | 0.1995 | 0.1474 | 30M GPT2 Atomic Cross |
| global_piqa_nonparallel_zh | 0.5300 | 0.5800 | 0.4500 | 0.5100 | 30M Qwen2 Atomic Within |
| global_piqa_parallel_zh | 0.2718 | 0.2427 | 0.2718 | 0.2913 | 30M Qwen2 Atomic Cross |
| hellaswag_zh_mubench | 0.2656 | 0.2675 | 0.2711 | 0.2726 | 30M Qwen2 Atomic Cross |
| include | 0.1786 | 0.2679 | 0.2946 | 0.2143 | 30M GPT2 Atomic Cross |
| mnli | 0.5220 | 0.5608 | 0.5203 | 0.5608 | 30M Qwen2 Atomic Within; 30M Qwen2 Atomic Cross |
| pos | 0.8475 | 0.8428 | 0.8397 | 0.8326 | 30M GPT2 Atomic Within |
| sib200 | 0.8100 | 0.7350 | 0.7700 | 0.6950 | 30M GPT2 Atomic Within |
| truthfulqa | 0.2946 | 0.2321 | 0.2589 | 0.2679 | 30M GPT2 Atomic Within |
| winogrande_zh_mubench | 0.5054 | 0.5111 | 0.4831 | 0.4880 | 30M Qwen2 Atomic Within |
| xcomps_zh | 0.5343 | 0.5292 | 0.5337 | 0.5259 | 30M GPT2 Atomic Within |
| xnli | 0.4705 | 0.4530 | 0.4485 | 0.4695 | 30M GPT2 Atomic Within |
| xstorycloze_zh_mubench | 0.5015 | 0.5077 | 0.4899 | 0.4946 | 30M Qwen2 Atomic Within |
| zhoblimp | 0.7535 | 0.7533 | 0.7104 | 0.7307 | 30M GPT2 Atomic Within |

## Generated files

- `metrics_long.csv`: every parsed metric with scale and cohort metadata.
- `primary_scores_wide.csv`: all 12 models side by side for every primary task.
- `suite_averages.csv`: descriptive per-model means within each suite.
- `pairwise_effects.csv`: task-level size, tokenizer, boundary, and architecture contrasts.
- `contrast_summary.csv`: mean paired contrast by suite.
- `cogbench_detail.csv`: subject/region-level CogBench results.
- `model_metadata.csv`: configuration and exact stored tensor counts.
- `hidden_tasks.csv`: archival prediction-only coverage for the original four models.

## Interpretation cautions

1. Do not combine accuracy, F1, MCC, and fMRI correlation into one score.
2. Suite averages weight tasks equally, not examples equally.
3. Aggregate files do not provide confidence intervals.
4. One seed exists per condition, so differences are not seed-robust evidence.
5. 30M/100M are experiment families; exact stored parameter counts appear above.
