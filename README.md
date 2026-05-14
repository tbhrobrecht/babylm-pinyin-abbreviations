# babylm-pinyin-abbreviations

Small utilities for working with the gated
[`BabyLM-community/babylm-zho`](https://huggingface.co/datasets/BabyLM-community/babylm-zho)
dataset.

## Extract the data

Install the one dependency:

```powershell
py -m pip install datasets
```

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
py extract_babylm_zho.py --output data/babylm_zho.jsonl
```

For a quick sample:

```powershell
py extract_babylm_zho.py --max-docs 100 --output data/sample.jsonl
```

Useful filters:

```powershell
py extract_babylm_zho.py --category child-books --script Hans --language zho
```

By default the script writes each original dataset row as JSON. Add
`--text-only` to keep just `doc_id` and `text`.
