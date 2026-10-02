# Draft figure captions

1. **Task-level performance across Mandarin BabyLM models.** Cell labels give
   raw scores multiplied by 100. Color represents each model's deviation from
   the mean for that task, so color comparisons are meaningful within rows and
   are not dominated by differences in task difficulty. CogBench values are
   correlation-based. Models are grouped by the original 30M and 100M cohorts
   and the atomic-BPE cohorts at both scales.

2. **Matched evaluation contrasts across tasks.** Small translucent points are
   task-level paired score differences and diamonds are unweighted means within
   each evaluation suite. Positive values favor the first condition named on
   each axis or contrast label. Dispersion is descriptive and is not a
   confidence interval because only one trained seed is available per model.

3. **Factorial experimental design.** Cells cross parameter scale (30M or
   100M), architecture (GPT2 or Qwen2), and tokenizer condition (hybrid, BPE,
   within-word atomic BPE, or cross-word atomic BPE). Filled cells have complete
   evaluations in the current results registry; hatched cells would indicate
   any future planned or in-progress condition.

4. **Within-word versus cross-word atomic BPE.** Connected markers compare the
   two boundary policies for matched architectures and tasks at the parameter
   scale named in the figure title. The generator produces one page per scale.
   Panels use independent horizontal ranges to preserve resolution. CogBench
   panels show correlation-based scores; the other panels show their
   suite-specific primary metrics.

5. **Descriptive performance averages within evaluation suites.** Each point is
   an unweighted mean over the tasks in one suite. Color denotes tokenizer,
   shape denotes architecture, and point size denotes parameter scale. Because
   tasks and metrics differ between suites, horizontal positions should only be
   compared within panels and should not be interpreted as a global score.

6. **CogBench fMRI performance by brain region.** Cell labels show mean
   subject-level correlations multiplied by 100. Color gives the deviation from
   the mean across models within each region, emphasizing relative model
   differences while retaining the raw values as annotations.
