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
with whitespace boundaries for downstream tokenizer training.

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

## Train the SentencePiece tokenizer

Train the default BPE tokenizer from the preprocessed 10k sample:

```powershell
py train_sentencepiece.py
```

This writes:

- `tokenizers\babylm_zho_pinyin_spm.model`
- `tokenizers\babylm_zho_pinyin_spm.vocab`

The default tokenizer uses an 8,000-piece BPE vocabulary, preserves the
preprocessing special tokens such as `<NUM>` and `<MATH>`, keeps whitespace-based
pretokenization, and avoids splitting pinyin-code digits away from their letters.

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
fixed-length `input_ids` and matching `labels` for causal language modeling.

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

## Upload the model to Hugging Face

Log in first:

```powershell
huggingface-cli login
```

or set a token:

```powershell
$env:HF_TOKEN = "hf_..."
```

Then upload the best checkpoint and SentencePiece tokenizer:

```powershell
py upload_to_hf.py your-username/pinyin-code-gpt-small
```

For a private repo:

```powershell
py upload_to_hf.py your-username/pinyin-code-gpt-small --private
```

To inspect the files before uploading:

```powershell
py upload_to_hf.py your-username/pinyin-code-gpt-small --dry-run --staging-dir hf_upload
```

The upload script packages `best.pt`, `pytorch_model.bin`, `config.json`, the
SentencePiece tokenizer, local inference code, and a generated Hugging Face model
card. This model uses custom PyTorch code rather than the Transformers model API.

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
the matching input format.


## tldr pipeline
python preprocessing/extract_babylm_zho.py 

python preprocessing/preprocess.py --input data/nk_babylm_zho.jsonl --output data/processed/nk_babylm_zho.txt 

python train_sentencepiece.py --input data/processed/nk_babylm_zho.txt --output-dir tokenizers --model-name babylm_zho_pinyin_spm --vocab-size 16000 

python create_dataset.py --input data/processed/nk_babylm_zho.txt --output data/datasets/nk_babylm_zho_spm.jsonl --tokenizer tokenizers/[name.model] --block-size 512 --stride 512

python train_model.py --dataset data/datasets/nk_spm.jsonl --output-dir models/[model name] --vocab-size 16000 --block-size 512 --n-layer 8 --n-head 8 --n-embd 512 --epochs 5 --batch-size 64 --learning-rate 3e-4 --device cuda

python hf/convert_to_transformers.py --checkpoint models/[model name]/best.pt --tokenizer tokenizers/[model name].model --output-dir hf_[model name] --transliteration pinyin-code

hf upload [username]/[model name] [model saved directory]

cd multilingual 

bash scripts/zeroshot_model.sh --model_name [username]/[model name] --langs "zho" --revision main
(bash scripts/zeroshot_model.sh --model_name YOUR_MODEL --langs "zho" --pinyin_format initials)

python -m lm_eval --model hf --model_args "pretrained=[username]/[model name],revision=main,trust_remote_code=True" --tasks zeroshot_zho --device cuda --output_path ../results/main --batch_size auto:10 --num_fewshot 0 --log_samples --include_path tasks/
