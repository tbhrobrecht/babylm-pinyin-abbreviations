# LRZ Chinese evaluation setup

These jobs keep evaluation dependencies separate from the training environment:

- Official BabyLM multilingual suite, restricted to `zho`: `~/babylm-lrz/env/babylm-eval-official`
- Chinese BabyLM evaluation pipeline: `~/babylm-lrz/env/chinese-babylm-eval`

Run all commands from `~/babylm-lrz` so the relative Slurm log paths resolve to
`~/babylm-lrz/logs/eval`.

## One-time setup

```bash
mkdir -p ~/babylm-lrz/logs/eval ~/babylm-lrz/results/eval

cd ~/babylm-lrz
setup_job=$(sbatch --parsable code/babylm-pinyin-abbreviations/jobs/eval/setup_chinese_eval_env.sbatch)
data_job=$(sbatch --parsable --dependency=afterok:${setup_job} code/babylm-pinyin-abbreviations/jobs/eval/download_chinese_eval_data.sbatch)
printf 'setup=%s data=%s\n' "$setup_job" "$data_job"
```

To prepare only the three zero-shot datasets first:

```bash
sbatch --export=ALL,TASKS="zhoblimp hanzi_structure hanzi_pinyin" \
  code/babylm-pinyin-abbreviations/jobs/eval/download_chinese_eval_data.sbatch
```

## Evaluate one model

The model must be a completed local Hugging Face export containing `config.json`.

```bash
MODEL="$HOME/babylm-lrz/code/babylm-pinyin-abbreviations/hf_nk_babylm_zho_gpt2_hybrid"

sbatch --export=ALL,MODEL_DIR="$MODEL" \
  code/babylm-pinyin-abbreviations/jobs/eval/eval_nk_official_zho.sbatch

sbatch --export=ALL,MODEL_DIR="$MODEL" \
  code/babylm-pinyin-abbreviations/jobs/eval/eval_nk_chinese_pipeline.sbatch
```

For a quick first pass through only the zero-shot tasks:

```bash
sbatch --export=ALL,MODEL_DIR="$MODEL",TASKS="zhoblimp hanzi_structure hanzi_pinyin" \
  code/babylm-pinyin-abbreviations/jobs/eval/eval_nk_chinese_pipeline.sbatch
```

Valid Chinese-pipeline tasks are `zhoblimp`, `hanzi_structure`, `hanzi_pinyin`,
`word_fmri`, `fmri`, `afqmc`, `ocnli`, `tnews`, and `cluewsc2020`. Existing
per-task results are skipped unless the upstream command is run with
`--force-redo`.

## Queue all four models

`queue_nk_evaluations.sh` contains the current four training job IDs and model
paths. It submits the environment and data jobs, then queues both evaluation
suites with `afterok` dependencies on the appropriate training/data jobs:

```bash
bash code/babylm-pinyin-abbreviations/jobs/eval/queue_nk_evaluations.sh
```

Review the IDs in that file before rerunning it after a retraining submission;
they currently refer to training jobs `5804010` through `5804013`.
