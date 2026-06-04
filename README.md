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
as whitespace-delimited atomic tokens for downstream tokenizer training.

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
preprocessing special tokens such as `<NUM>` and `<MATH>`, and avoids splitting
pinyin-code digits away from their letters. SentencePiece is allowed to learn BPE
pieces that span adjacent whitespace-delimited atomic tokens, so frequent
multi-token pinyin-code patterns can become single tokenizer pieces.

Useful options:

```powershell
py train_sentencepiece.py --vocab-size 4000 --model-name babylm_zho_pinyin_spm_4k
py train_sentencepiece.py --model-type unigram --output-dir tokenizers\unigram
py train_sentencepiece.py --input data\processed\10k_babylm_zho.txt data\processed\extra.txt
```

`--hard-vocab-limit` is disabled by default so SentencePiece can still finish if
the corpus cannot support the exact requested vocabulary size.

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

Binary datasets write raw int32 token chunks plus a `.meta.json` sidecar.
`train_model.py` automatically detects either JSONL or binary datasets.

## Train the language model

Train a compact GPT-style causal language model on the tokenized dataset:

```powershell
py train_model.py
```

This writes checkpoints to `models\pinyin-code-gpt-small`:

- `last.pt`
- `best.pt`

The default model uses 6 Transformer layers, 8 attention heads, 256 hidden
dimensions, 128-token context windows, and the tokenizer's 8,000-token
vocabulary. Use `--device cuda` on a CUDA-capable GPU, or `--device cpu` to force
CPU training.

At startup, `train_model.py` prints the selected device, CUDA device name,
parameter count, tokens per epoch, and whether AMP/TF32/compile fast paths are
active. If it prints `device=cpu` or `cuda_name=none`, training will be much
slower than CUDA-based repository baselines. A CPU-only PyTorch install cannot
use `--device cuda`; install a CUDA-enabled PyTorch build first.

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

Resume a run from a previous checkpoint with `--resume`. Checkpoints only include
optimizer state when they were written with `--save-optimizer`; otherwise the
model weights resume and the optimizer starts fresh.

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
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "hf_pinyin_code_model",
    trust_remote_code=True,
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

For a private model, create the Hugging Face model repo as private first, then
upload the same converted folder.


## tldr pipeline
python preprocessing/extract_babylm_zho.py 

python preprocessing/preprocess.py --input data/nk_babylm_zho.jsonl --output data/processed/nk_babylm_zho.txt 

python train_sentencepiece.py --input data/processed/nk_babylm_zho.txt --output-dir tokenizers --model-name babylm_zho_pinyin_spm --vocab-size 16000 

python create_dataset.py --format bin --input data/processed/nk_babylm_zho.txt --output data/datasets/nk_babylm_zho_train_spm.bin --validation-output data/datasets/nk_babylm_zho_valid_spm.bin --validation-fraction 0.05 --tokenizer tokenizers/[model name].model --block-size 512 --stride 512

python train_model.py --dataset data/datasets/nk_babylm_zho_train_spm.bin --validation-dataset data/datasets/nk_babylm_zho_valid_spm.bin --output-dir models/[model name] --vocab-size 16000 --block-size 512 --n-layer 8 --n-head 8 --n-embd 512 --epochs 5 --batch-size 64 --learning-rate 3e-4 --device cuda

python hf/convert_to_transformers.py --checkpoint models/[model name]/best.pt --tokenizer tokenizers/[model name].model --output-dir hf_[model name] --transliteration pinyin-code
(add --no-jieba here if the training corpus was preprocessed with --no-jieba)

hf upload [username]/[model name] [model saved directory]

cd multilingual 

bash scripts/zeroshot_model.sh --model_name [username]/[model name] --langs "zho" --revision main
(bash scripts/zeroshot_model.sh --model_name YOUR_MODEL --langs "zho" --pinyin_format initials)

python -m lm_eval --model hf --model_args "pretrained=[username]/[model name],revision=main,trust_remote_code=True" --tasks zeroshot_zho --device cuda --output_path ../results/main --batch_size auto:10 --num_fewshot 0 --log_samples --include_path tasks/
