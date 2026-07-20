# babylm-pinyin-abbreviations

Small utilities for working with the gated
[`BabyLM-community/babylm-zho`](https://huggingface.co/datasets/BabyLM-community/babylm-zho)
dataset.

## Install dependencies

Install the project dependencies:

```powershell
py -m pip install -r requirements.txt
```

## Extract the data

Accept the dataset terms on Hugging Face, then either log in:

```powershell
huggingface-cli login
```

or set a token:

```powershell
$env:HF_TOKEN = "hf_..."
```

Extract the training split to JSONL:

```powershell
py preprocessing\extract_babylm_zho.py --output data\babylm_zho.jsonl
```

For a quick sample:

```powershell
py preprocessing\extract_babylm_zho.py --max-docs 100 --output data\sample.jsonl
```

Useful filters:

```powershell
py preprocessing\extract_babylm_zho.py --category child-books --script Hans --language zho
```

By default the script writes each original dataset row as JSON. Add
`--text-only` to keep just `doc_id` and `text`.

## Preprocess the data

Convert extracted Mandarin JSONL into the default pinyin-code format used for
tokenizer training:

```powershell
py preprocessing\preprocess.py --input data\10k_babylm_zho.jsonl --output data\processed\10k_babylm_zho.txt
```

The output file contains one preprocessed document per line. Chinese words are
segmented with `jieba`, converted to compact pinyin-code tokens, and preserved
as whitespace-delimited atomic tokens for downstream tokenizer training. The
preprocessor also keeps visible punctuation, symbols, and non-Mandarin
alphanumeric words such as English product names, while normalizing URLs, math
blocks, and standalone numbers to stable markers.

To disable `jieba` and preprocess Chinese character-by-character instead, add
`--no-jieba`:

```powershell
py preprocessing\preprocess.py --input data\10k_babylm_zho.jsonl --output data\processed\10k_babylm_zho_char.txt --no-jieba
```

To create a lowercase pinyin-first-letter corpus instead, use:

```powershell
py preprocessing\preprocess.py --input data\10k_babylm_zho.jsonl --output data\processed\10k_babylm_zho_initials.txt --transliteration pinyin-initial
```

For example, the default `pinyin-code` transliteration keeps tone/length casing
such as `Z4g2`, while `pinyin-initial` emits `zg`.

To keep the same preprocessing pipeline but leave Mandarin words as segmented
Hanzi instead of pinyin, use:

```powershell
py preprocessing\preprocess.py --input data\10k_babylm_zho.jsonl --output data\processed\10k_babylm_zho_hanzi.txt --transliteration hanzi
```

## Export statistics as a LaTeX table

Convert the tone and syllable-length statistics summary into a formatted
academic table:

```powershell
py preprocessing\statistics_to_latex.py
```

This writes `tables\10k_statistics_table.tex`, a ready-to-include table using the
LaTeX `booktabs` and `siunitx` packages. Override `--input`, `--output`,
`--caption`, or `--label` when preparing another corpus or paper.

## Train the SentencePiece tokenizer

Train the default BPE tokenizer from the preprocessed 10k sample:

```powershell
py train_sentencepiece.py
```

This writes:

- `tokenizers\babylm_zho_pinyin_spm.model`
- `tokenizers\babylm_zho_pinyin_spm.vocab`

The default tokenizer uses an 8,000-piece BPE vocabulary, preserves the
preprocessing special tokens such as `<NUM>`, `<MATH>`, and `<URL>`, and avoids
splitting pinyin-code digits away from their letters. SentencePiece is allowed to
learn BPE pieces that span adjacent whitespace-delimited atomic tokens, so
frequent multi-token pinyin-code patterns can become single tokenizer pieces.
For BPE training, long processed lines are split at whitespace boundaries before
calling SentencePiece by default; this avoids SentencePiece's per-line 16-bit
position limit on large corpora while preserving all tokens. Use
`--no-split-long-lines` only if you are sure your input lines are already short
enough.

Useful options:

```powershell
py train_sentencepiece.py --vocab-size 4000 --model-name babylm_zho_pinyin_spm_4k
py train_sentencepiece.py --model-type unigram --output-dir tokenizers\unigram
py train_sentencepiece.py --input data\processed\10k_babylm_zho.txt data\processed\extra.txt
```

`--hard-vocab-limit` is disabled by default so SentencePiece can still finish if
the corpus cannot support the exact requested vocabulary size.

## Train the hybrid Jieba-word tokenizer

The SentencePiece tokenizer remains the BPE baseline. For experiments that keep
Jieba word boundaries explicit, build the hybrid tokenizer from the same
preprocessed pinyin-code corpus:

```powershell
py train_hybrid_tokenizer.py --input data\processed\10k_babylm_zho.txt --output-dir tokenizers\babylm_zho_hybrid_16k --vocab-size 16000 --min-word-frequency 20
```

The hybrid vocabulary uses fixed IDs for `<pad>`, `<unk>`, `<s>`, `</s>`, and
`<mask>`, includes the preprocessing markers such as `<QUESTION>` and `<NUM>`,
adds every configurable Initial+Digit atom (`A0` through `z9` by default), and
then adds frequent multi-atom Jieba words by corpus frequency. Valid encoded
words that are not in the whole-word vocabulary fall back to their two-character
atoms, so they do not become `<unk>`.

Useful variants:

```powershell
py train_hybrid_tokenizer.py --input data\processed\10k_babylm_zho.txt --output-dir tokenizers\babylm_zho_atomic --atomic-only
py train_hybrid_tokenizer.py --input data\processed\10k_babylm_zho.txt --output-dir tokenizers\babylm_zho_hybrid_2k --vocab-size 2000
py train_hybrid_tokenizer.py --input data\processed\10k_babylm_zho.txt --output-dir tokenizers\babylm_zho_hybrid_4k --vocab-size 4000
py train_hybrid_tokenizer.py --input data\processed\10k_babylm_zho.txt --output-dir tokenizers\babylm_zho_hybrid_8k --vocab-size 8000
py train_hybrid_tokenizer.py --input data\processed\10k_babylm_zho.txt --output-dir tokenizers\babylm_zho_hybrid_16k --vocab-size 16000
```

Inspect a built tokenizer and optional corpus coverage:

```powershell
py scripts\inspect_hybrid_tokenizer.py --tokenizer-dir tokenizers\babylm_zho_hybrid_16k --input data\processed\10k_babylm_zho.txt --example "Y0J7 H2 X4Q3"
```

Use the tokenizer directly:

```python
from hf.tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer

tokenizer = HybridPinyinCodeTokenizer.from_pretrained("tokenizers/babylm_zho_hybrid_16k")
print(tokenizer.tokenize("Y0J7 H2 X4Q3"))
```

The tokenizer directory also includes remote-code metadata, so this works in a
Transformers environment when the directory contains
`tokenization_hybrid_pinyin_code.py`:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "tokenizers/babylm_zho_hybrid_16k",
    trust_remote_code=True,
)
```

Decoding is deterministic back to encoded text, but it is not a lossless
reconstruction of the original Hanzi or exact Jieba segmentation. Whole-word
tokens decode as complete encoded words, while consecutive fallback atoms may be
concatenated. Pass `readable=True` to `decode` when you want spaces between the
emitted tokenizer pieces.

Current SentencePiece BPE:

- learns recursive frequency-based merges;
- may produce partial-word pieces;
- may split letters and digits;
- may merge across Jieba boundaries depending on configuration.

New hybrid tokenizer:

- uses complete Jieba words selected directly by frequency;
- never creates partial-word lexical tokens;
- guarantees atomic Initial+Digit fallback;
- never needs `<unk>` for valid encoded words;
- uses whitespace only as preprocessing boundary metadata.

## Create a tokenized dataset

Build a chunked JSONL language-modeling dataset from the processed text and
trained SentencePiece tokenizer:

```powershell
py create_dataset.py
```

This writes `data\datasets\10k_babylm_zho_spm.jsonl` by default. Each line has
fixed-length `input_ids` for causal language modeling. Add `--include-labels` if
you want the JSONL file to also contain labels identical to `input_ids`;
`train_model.py` does not need them.

For a cleaner validation signal, split by original processed document before
chunking instead of randomly splitting adjacent chunks after dataset creation:

```powershell
py create_dataset.py --input data\processed\10k_babylm_zho.txt --output data\datasets\10k_train_spm.jsonl --validation-output data\datasets\10k_valid_spm.jsonl --validation-fraction 0.05
```

For faster training startup and lower parsing overhead, write compact binary
chunk files instead:

```powershell
py create_dataset.py --format bin --input data\processed\10k_babylm_zho.txt --output data\datasets\10k_train_spm.bin --validation-output data\datasets\10k_valid_spm.bin --validation-fraction 0.05
```

For a hybrid tokenizer experiment, pass the tokenizer directory instead of a
SentencePiece `.model` file:

```powershell
py create_dataset.py --format bin --input data\processed\10k_babylm_zho.txt --output data\datasets\10k_train_hybrid_16k.bin --validation-output data\datasets\10k_valid_hybrid_16k.bin --validation-fraction 0.05 --tokenizer tokenizers\babylm_zho_hybrid_16k --block-size 512 --stride 512
```

Binary datasets write raw int32 token chunks plus a `.meta.json` sidecar.
`train_model.py` automatically detects either JSONL or binary datasets.

## Train the language model

Train a compact GPT-style causal language model on the tokenized dataset:

```powershell
py train_model.py
```

`train_model.py` supports two decoder-only architectures:

- `gpt2`: the repository's original compact GPT-style PyTorch model. This is
  the default, so older commands and checkpoints continue to use it.
- `qwen2`: a Hugging Face `Qwen2ForCausalLM` initialized from scratch. It does
  not download or load pretrained Qwen weights.

Select the architecture with `--architecture`. Existing commands that omit this
flag are equivalent to `--architecture gpt2`.

This writes checkpoints to `models\pinyin-code-gpt-small`:

- `last.pt`
- `best.pt`
- `final.pt`
- BabyLM interval checkpoints named `chck_1M`, `chck_2M`, ..., `chck_10M`,
  then `chck_20M`, ..., `chck_100M`

It also writes structured metrics to `metrics.jsonl` in the output directory by
default. Training events include `train_loss`, learning rate, step, epoch, and
token throughput; validation events include `validation_loss`, `best_loss`,
step, and epoch. Pass `--metrics-log path\to\metrics.jsonl` to choose another
location.

BabyLM interval checkpoints are saved when the run crosses the corresponding
number of training-token exposures. This means tokenizer-id tokens consumed by
training batches, including repeated exposure across epochs; it is not a count
of original corpus words or Jieba words. Each interval checkpoint uses the
BabyLM revision-style name directly, for example `chck_1M`, so it can be
converted with `hf\convert_to_transformers.py --checkpoint models\...\chck_1M`
and uploaded to the matching Hugging Face revision.

The default model uses 6 Transformer layers, 8 attention heads, 256 hidden
dimensions, 128-token context windows, and the tokenizer's 8,000-token
vocabulary. Use `--device cuda` on a CUDA-capable GPU, or `--device cpu` to force
CPU training.

At startup, `train_model.py` prints the selected architecture, total parameter
count, trainable parameter count, vocabulary size, layer count, hidden size,
attention-head count, Qwen2 key-value-head count when relevant, maximum sequence
length, selected device, CUDA device name, tokens per epoch, and whether
AMP/TF32/compile fast paths are active. If it prints `device=cpu` or
`cuda_name=none`, training will be much slower than CUDA-based repository
baselines. A CPU-only PyTorch install cannot use `--device cuda`; install a
CUDA-enabled PyTorch build first.

CUDA training enables automatic mixed precision, TF32, and fused AdamW by
default when available. Disable them with `--no-amp`, `--no-tf32`, or
`--no-fused-adamw` for debugging. `--compile` opts into `torch.compile`, which
can improve longer GPU runs after a startup compilation cost. Checkpoints omit
optimizer state by default to reduce disk I/O; pass `--save-optimizer` if you
need optimizer state for manual resuming.

Use gradient accumulation when you want a larger effective batch size than fits
comfortably in VRAM. The learning-rate schedule is applied per optimizer update,
not per mini-batch; cosine decay is the default, with optional linear warmup and
a nonzero floor:

```powershell
py train_model.py --dataset data\datasets\10k_train_spm.bin --validation-dataset data\datasets\10k_valid_spm.bin --device cuda --batch-size 64 --gradient-accumulation-steps 4 --warmup-steps 200 --min-learning-rate 3e-5
```

On a CUDA machine, such as a workstation with an RTX GPU, keep using
`--device cuda`. If the startup line prints your CUDA device name, the CUDA fast
paths are active.

When you created a separate validation dataset, pass it during training:

```powershell
py train_model.py --dataset data\datasets\10k_train_spm.bin --validation-dataset data\datasets\10k_valid_spm.bin --device cuda
```

For Qwen2, keep the tokenizer, dataset, and preprocessing pipeline unchanged and
only switch the model architecture and shape fields:

```powershell
py train_model.py --architecture qwen2 --dataset data\datasets\10k_train_hybrid_16k.bin --validation-dataset data\datasets\10k_valid_hybrid_16k.bin --output-dir models\pinyin-code-qwen2-small --vocab-size 16000 --block-size 512 --n-layer 8 --n-head 8 --n-embd 512 --num-key-value-heads 4 --intermediate-size 1376 --device cuda
```

The Qwen2-specific shape fields are `--num-key-value-heads`,
`--intermediate-size`, `--rms-norm-eps`, `--rope-theta`,
`--attention-dropout`, `--tie-word-embeddings`, and
`--attn-implementation`. The required comparable-size fields are still
`--n-embd`, `--n-layer`, `--n-head`, `--num-key-value-heads`,
`--intermediate-size`, and `--block-size`. The builder validates that hidden
size is divisible by attention heads and that attention heads are divisible by
key-value heads. See `configs\qwen2_33m.yaml` for an example experiment record.

Minimal CPU smoke test for Qwen2, using any tiny JSONL/bin chunk dataset:

```powershell
py train_model.py --architecture qwen2 --dataset data\datasets\tiny.jsonl --output-dir models\qwen2-smoke --vocab-size 64 --block-size 16 --n-layer 1 --n-head 2 --n-embd 16 --num-key-value-heads 1 --intermediate-size 32 --epochs 1 --batch-size 1 --device cpu --save-optimizer
```

Resume a run from a previous checkpoint with `--resume`. Checkpoints only include
optimizer state when they were written with `--save-optimizer`; otherwise the
model weights resume and the optimizer starts fresh. New checkpoints store an
explicit `architecture` field. Legacy GPT-style checkpoints without this field
are still inferred as `gpt2`; explicit GPT2/Qwen2 architecture conflicts fail
before weights are loaded.

## Optional DPO fine-tuning

DPO is used as a post-pretraining preference optimization stage to reduce
ambiguity introduced by compact pinyin-initial encoding. Preference pairs are
constructed from gold encoded continuations versus model-generated incorrect
continuations.

This is an offline preference optimization stage, not full RLHF or PPO, and it
does not replace the base language-model pretraining pipeline. Because the
project trains a pure causal LM rather than an explicit prompt/target model, the
DPO dataset builder splits each encoded sequence into a prompt from the first
30-50% of tokens and a gold continuation from the remaining tokens. The
pretrained model samples alternative continuations from the same prompt, and
usable incorrect continuations become rejected responses.

Run the base pretraining stage as usual:

```powershell
py train_model.py --dataset data\datasets\10k_train_spm.bin --validation-dataset data\datasets\10k_valid_spm.bin --output-dir models\pinyin-code-gpt-small --vocab-size 8000 --block-size 512 --device cuda
```

Build preference pairs from raw Hanzi JSONL or plain-text lines:

```powershell
py scripts\build_dpo_dataset.py --input-hanzi data\10k_babylm_zho.jsonl --output data\dpo_preferences.jsonl --model-checkpoint models\pinyin-code-gpt-small\best.pt --tokenizer tokenizers\babylm_zho_pinyin_spm.model --num-samples 1000 --num-candidates 4 --max-length 512 --device cuda
```

Fine-tune the policy checkpoint with DPO:

```powershell
py train_dpo.py --dpo-dataset data\dpo_preferences.jsonl --base-checkpoint models\pinyin-code-gpt-small\best.pt --output-dir models\pinyin-code-gpt-small-dpo --tokenizer tokenizers\babylm_zho_pinyin_spm.model --beta 0.1 --learning-rate 5e-6 --epochs 1 --batch-size 4 --gradient-accumulation-steps 8 --max-length 512 --device cuda
```

Compare the base and DPO checkpoints on the same preference records:

```powershell
py evaluate_dpo.py --dpo-dataset data\dpo_preferences.jsonl --base-checkpoint models\pinyin-code-gpt-small\best.pt --dpo-checkpoint models\pinyin-code-gpt-small-dpo\best.pt --tokenizer tokenizers\babylm_zho_pinyin_spm.model --max-length 512 --device cuda --examples 3
```

`train_dpo.py` keeps a frozen reference copy of the base checkpoint, initializes
the trainable policy from that same checkpoint, and computes log-probabilities
only over completion tokens. The DPO output directory contains `last.pt`,
`best.pt`, `final.pt`, and `dpo_training_config.json`; the checkpoint format
keeps `model_state_dict` and `model_config`, so `generate.py` can load DPO
checkpoints the same way it loads pretrained checkpoints. If `--max-length`
exceeds the checkpoint context window, the DPO scripts clamp it to the model's
`block_size`.

## Convert to a Transformers model folder

Convert the existing PyTorch checkpoint without retraining:

```powershell
py hf\convert_to_transformers.py
```

This writes `hf_pinyin_code_model\` with custom `trust_remote_code` files,
`config.json`, `model.safetensors`, `tokenizer.model`, tokenizer metadata, and
`generation_config.json`.

Load it with:

```python
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

config = AutoConfig.from_pretrained(
    "hf_pinyin_code_model",
    trust_remote_code=True,
)
base_model = AutoModel.from_pretrained(
    "hf_pinyin_code_model",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "hf_pinyin_code_model",
    trust_remote_code=True,
)
classifier = AutoModelForSequenceClassification.from_pretrained(
    "hf_pinyin_code_model",
    trust_remote_code=True,
    num_labels=3,
)
tokenizer = AutoTokenizer.from_pretrained(
    "hf_pinyin_code_model",
    trust_remote_code=True,
)
```

Pass the same `--transliteration` value here that you used when preprocessing
the training corpus. The exported tokenizer stores that value and applies it to
raw Mandarin/Hanzi benchmark prompts before SentencePiece tokenization, so
`lm_eval` can evaluate `pinyin-code`, `pinyin-initial`, or `hanzi` models with
the matching input format. Also pass `--no-jieba` when converting a model trained
on character-level Chinese preprocessing, so benchmark prompts use the same
segmentation.

## Transformers and external evaluation compatibility

The exported model is a causal language model and can also be loaded with
`AutoModelForSequenceClassification` to reuse the same GPT-style backbone with a
fresh classifier head. For external evaluation repositories, set only the
evaluation config to point at the local model folder or uploaded Hugging Face
repo ID, enable `trust_remote_code`, and select the `causal` backend for
language-model evaluation.

Runtime dependencies for the exported model are declared in `requirements.txt`.
For a minimal evaluation environment, install:

```powershell
py -m pip install torch transformers safetensors sentencepiece pypinyin jieba
```

`sentencepiece` is required by `AutoTokenizer`. `pypinyin` is required for raw
Mandarin-to-pinyin tokenization. `jieba` is required for model exports created
with the default jieba segmentation.

Run the compatibility smoke test against a local export or repo ID:

```powershell
py tests\hf_compatibility_smoke.py hf_models\hf_full_chinese_gpu3
```

The smoke test verifies `AutoConfig`, `AutoTokenizer`, `AutoModel`,
`AutoModelForCausalLM`, and `AutoModelForSequenceClassification`, plus CPU
`torch.no_grad()` forward passes with logits shapes
`[batch, sequence_length, vocab_size]` and `[batch, num_labels]`.

The slow tokenizer also accepts `return_offsets_mapping=True` for compatibility
with evaluators that need completion-span masks, and the model supports
`output_hidden_states=True` for representation extraction tasks.

Exports set `patch_pathlib_utf8_open=true` in `config.json`. When the model is
loaded with `trust_remote_code=True`, the config installs a narrow Windows
compatibility shim so later text-mode `Path.open("r")` calls without an
explicit encoding default to UTF-8. This helps evaluation repositories that
read UTF-8 Chinese JSONL files without passing `encoding="utf-8"`. Set
`PINYIN_CODE_DISABLE_UTF8_OPEN_PATCH=1` before loading the model to disable the
shim.

## Upload the model to Hugging Face

Log in first:

```powershell
huggingface-cli login
```

or set a token:

```powershell
$env:HF_TOKEN = "hf_..."
```

After conversion, upload the generated Transformers model folder:

```powershell
hf upload your-username/pinyin-code-gpt-small hf_pinyin_code_model --repo-type model
```

For BabyLM checkpoint revisions, convert each local interval checkpoint to its
own temporary/export folder, then upload that folder to the matching revision
name, for example:

```powershell
py hf\convert_to_transformers.py --checkpoint models\pinyin-code-gpt-small\chck_1M --tokenizer tokenizers\babylm_zho_pinyin_spm.model --output-dir hf_pinyin_code_model_chck_1M
hf upload your-username/pinyin-code-gpt-small hf_pinyin_code_model_chck_1M --repo-type model --revision chck_1M
```

For a private model, create the Hugging Face model repo as private first, then
upload the same converted folder.


## tldr pipeline
Run the automated pretraining pipeline with one command. The default is the
repository GPT-style architecture with the hybrid Jieba-word tokenizer:

```powershell
python scripts/train_babylm_from_scratch.py --model-name nk_babylm_zho --device cuda
```

For the four main pretraining runs in one HPC job, cross both model
architectures with both tokenizer families:

```powershell
python scripts/train_babylm_from_scratch.py --model-name nk_babylm_zho --architectures gpt2 qwen2 --tokenizer-kinds hybrid bpe --device cuda --vocab-size 16000 --block-size 512 --stride 512 --epochs 5 --batch-size 64 --learning-rate 3e-4 --preprocess-workers 8 --num-workers 4 --resume
```

This creates shared corpus artifacts and tokenizer-specific datasets, then
trains separate model/export folders:

- `models\nk_babylm_zho_gpt2_hybrid` and `hf_nk_babylm_zho_gpt2_hybrid`
- `models\nk_babylm_zho_qwen2_hybrid` and `hf_nk_babylm_zho_qwen2_hybrid`
- `models\nk_babylm_zho_gpt2_bpe` and `hf_nk_babylm_zho_gpt2_bpe`
- `models\nk_babylm_zho_qwen2_bpe` and `hf_nk_babylm_zho_qwen2_bpe`

Use `--dry-run` to print the underlying commands without running them. Use
`--resume` to skip steps whose expected outputs already exist; within a matrix
run, extraction/preprocessing are shared across all runs, and tokenizer/dataset
creation is shared across the two architectures for each tokenizer kind.

Use `--start-at` and `--stop-after` to run only part of the pipeline. For
example, prepare corpus/tokenizers/datasets without training:

```powershell
python scripts/train_babylm_from_scratch.py --model-name nk_babylm_zho --architectures gpt2 qwen2 --tokenizer-kinds hybrid bpe --stop-after dataset --device cuda --resume
```

Then continue later from model training:

```powershell
python scripts/train_babylm_from_scratch.py --model-name nk_babylm_zho --architectures gpt2 qwen2 --tokenizer-kinds hybrid bpe --start-at train --device cuda --resume
```

You can still skip individual steps with `--skip-extract`, `--skip-preprocess`,
`--skip-tokenizer`, `--skip-dataset`, `--skip-train`, `--skip-convert`, or
`--skip-upload` when you already have specific artifacts.

Single-run examples:

```powershell
python scripts/train_babylm_from_scratch.py --model-name nk_babylm_zho_qwen2_bpe --architecture qwen2 --tokenizer-kind bpe --device cuda --vocab-size 16000 --block-size 512 --n-layer 8 --n-head 8 --n-embd 512 --num-key-value-heads 4 --intermediate-size 1376
python scripts/train_babylm_from_scratch.py --model-name nk_babylm_zho_gpt2_hybrid --architecture gpt2 --tokenizer-kind hybrid --device cuda --vocab-size 16000 --block-size 512
```

If you already have an extracted JSONL on the HPC filesystem, skip extraction
and point the pipeline at it:

```powershell
python scripts/train_babylm_from_scratch.py --model-name nk_babylm_zho --raw-output data/nk_babylm_zho.jsonl --skip-extract --architectures gpt2 qwen2 --tokenizer-kinds hybrid bpe --device cuda --resume
```

If `--hf-repo` is provided for a single run, the converted folder is uploaded
after conversion. Matrix runs intentionally do not accept a single `--hf-repo`
because each export needs its own model repo or an explicit manual upload:

```powershell
hf upload [username]/[model name] [model saved directory] --repo-type model
```

For external evaluation after upload:

```powershell
cd multilingual
bash scripts/zeroshot_model.sh --model_name [username]/[model name] --langs "zho" --revision main
(bash scripts/zeroshot_model.sh --model_name YOUR_MODEL --langs "zho" --pinyin_format initials)

python -m lm_eval --model hf --model_args "pretrained=[username]/[model name],revision=main,trust_remote_code=True" --tasks zeroshot_zho --device cuda --output_path ../results/main --batch_size auto:10 --num_fewshot 0 --log_samples --include_path tasks/
```
