# Paper-focused figure captions

These figures compare the four coded-pinyin systems and two standard baselines
selected for the main paper. All six are 100M-family GPT-2 models (97.7M trainable
parameters), use a 16k vocabulary and 512-token context, and were trained for five
epochs on alternative representations of the same source documents. There is one
trained seed per condition.

## P1. Focused experimental design

Controlled six-model comparison used in the main paper. The four coded-pinyin
systems (Hybrid, Encoded BPE, Atomic within, and Atomic cross) are compared with
Hanzi BPE and tone-number Full-pinyin BPE. Architecture, parameter count,
vocabulary size, context length, training epochs, source documents, data split,
and evaluation are held constant; the manipulated factor is representation and
tokenization.

## P2. Suite profiles

Descriptive mean performance in the four evaluation suites. Each point is the
unweighted mean of the primary task scores in that suite (score multiplied by
100). Axis ranges differ between panels because CogBench correlations and task
accuracies occupy different numerical ranges. These values summarize profiles;
they should not be read as a single cross-suite performance scale.

## P3. Task performance heatmap

Performance across all 25 primary metrics. Cell text reports the original metric
value multiplied by 100. Color reports a model's percentile rank among all 18
evaluated models on that task, preserving the complete analysis normalization
while displaying only the six focal models. Color is comparable across tasks
even where raw metric types differ. White rules separate evaluation suites.

## P4. Task-family profiles

Mean relative performance in five substantively defined task families. Models
are percentile-ranked within each task across all 18 evaluated models before
ranks are averaged within family; this avoids directly averaging accuracy, F1,
MCC, and correlation. Only the six focal models are displayed.

## P5. Composite performance and task wins

Panel A filters the established complete-experiment equal-suite composite to the
six focal models: within-task percentile ranks across all 18 models are averaged
within each suite and the four suite means are weighted equally. Error
bars are 95% intervals from resampling observed tasks within suites; they describe
task-set sensitivity and do not estimate uncertainty across training seeds.
Panel B counts tasks on which each model obtains the best raw score among the six;
tied wins are divided fractionally. Aggregate performance and breadth of task
leadership are deliberately shown together because they answer different
questions.

## P6. Efficiency trade-offs

Representation size, tokenized training length, and the performance-token
trade-off for the same six models. Panel A reports UTF-8 processed-text size and
is therefore a storage/representation measure, not evidence that models saw
different source information. Panel B reports uint32 training tokens per epoch;
all models train for five epochs. Panel C relates those token counts to the
complete-experiment equal-suite composite. Model architecture and parameter count are fixed,
so differences reflect representation/tokenizer efficiency rather than a smaller
network.

## P7. Atomic-within benefits relative to standard baselines

Resource and performance scorecard for Atomic-within, Hanzi BPE, and
tone-number Full-pinyin BPE. All conditions use the same source documents,
97.7M-parameter GPT-2 architecture, 16k vocabulary, 512-token context, and five
training epochs. The first four panels report costs for which lower is better:
UTF-8 processed-text storage, training tokens per epoch, optimizer steps over
five epochs, and observed elapsed time on one A100. The elapsed times are single
observed training runs rather than replicated hardware benchmarks. Fixed-context
coverage is proportional to the amount of the common source corpus represented
by a 512-token window and is indexed to Atomic-within = 100. Composite
performance uses the established normalization over all 18 evaluated models.
Atomic-within uses fewer tokens and steps than either standard baseline while
matching Hanzi BPE on the aggregate composite; these efficiencies are not caused
by a smaller model.
