# Draft main-text figure captions

## M1. Experimental overview

**Mandarin BabyLM experimental pipeline.** A common Mandarin BabyLM corpus is
used to train GPT2 and Qwen2 models that cross two parameter scales with four
tokenizer conditions: hybrid, conventional BPE, within-word atomic BPE, and
cross-word atomic BPE. Models are assessed with the official BabyLM evaluation,
Chinese-specific evaluations, and CogBench. Matched contrasts isolate model
scale, architecture, tokenizer family, and atomic-BPE boundary policy. The cell
count is generated from the current model registry.

## M2. Scaling trajectories

**Performance trajectories with increasing model scale.** Lines connect 30M
and 100M models with the same architecture and tokenizer. Color denotes
tokenizer and marker shape denotes architecture. Each point is the unweighted
mean of the primary task scores within the named suite. Panels use independent
vertical ranges and should not be compared as a combined overall score. Atomic
BPE trajectories are added automatically once both scales have complete
evaluations.

## M3. Boundary-policy trajectories

**Effect of atomic-BPE boundary policy.** Lines connect models differing only in
whether atomic BPE merges are constrained within words or may cross word
boundaries. Color and marker shape denote architecture; line style denotes
parameter scale. Each point is the unweighted mean of primary task scores within
the named suite. Panels use independent vertical ranges. A line is drawn only
when both members of the matched pair are available.

## M4. Task-family profiles

**Relative performance profiles across task families.** Models are ranked
within each task, with the lowest-performing model assigned 0 and the
highest-performing model assigned 100; tied observations receive their average
rank. Values are then averaged within task families. This transformation avoids
directly averaging incompatible accuracy, F1, MCC, and correlation scales. Lines
are categorical profiles and do not imply a continuous ordering of task
families. Color denotes tokenizer, line style denotes parameter scale, and
panels separate architectures.

Task families are defined before aggregation as follows:

- **Reasoning and commonsense:** ARC, Global PIQA, HellaSwag, TruthfulQA,
  WinoGrande, XStoryCloze, and XCOMPS.
- **Cross-lingual understanding:** Belebele, BMLAMA, INCLUDE, MNLI, SIB200, and
  XNLI.
- **Chinese linguistic form:** POS, ZhoBLiMP, Hanzi–pinyin, and Hanzi structure.
- **Chinese downstream:** AFQMC, CLUE WSC 2020, OCNLI, and TNEWS.
- **Cognitive alignment:** CogBench fMRI and word-fMRI tasks.
