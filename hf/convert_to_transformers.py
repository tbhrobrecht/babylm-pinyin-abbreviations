"""Convert the trained custom PyTorch checkpoint to a Transformers model repo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import sentencepiece as spm
import torch
from transformers import GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hf.configuration_pinyin_code import PinyinCodeConfig
from hf.modeling_pinyin_code import PinyinCodeForCausalLM, PinyinCodeForMaskedLM
from hf.tokenization_pinyin_code import EncodedMandarinTokenizer


DEFAULT_OUTPUT_DIR = Path("hf_pinyin_code_model")
DEFAULT_TOKENIZER = Path("tokenizers/babylm_zho_pinyin_spm.model")
DEFAULT_CHECKPOINT_CANDIDATES = (
    Path("models/pinyin-code-gpt-small/best.pt"),
    Path("models/pinyin-code-gpt-small/best_model.pt"),
)


def default_checkpoint() -> Path:
    """Return the first existing default checkpoint path."""
    for path in DEFAULT_CHECKPOINT_CANDIDATES:
        if path.exists():
            return path
    return DEFAULT_CHECKPOINT_CANDIDATES[0]


def load_training_checkpoint(path: Path) -> dict:
    """Load the original train_model.py checkpoint."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain `model_state_dict`")
    if "model_config" not in checkpoint:
        raise KeyError(f"{path} does not contain `model_config`")
    return checkpoint


def sentencepiece_piece_id(processor: spm.SentencePieceProcessor, piece: str) -> int | None:
    """Return a piece id only when the model contains that exact piece."""
    token_id = int(processor.piece_to_id(piece))
    if token_id < 0 or processor.id_to_piece(token_id) != piece:
        return None
    return token_id


def tokenizer_special_ids(tokenizer_path: Path) -> dict[str, int | None]:
    """Read special token ids from the existing SentencePiece model."""
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    ids = {
        "bos_token_id": processor.bos_id(),
        "eos_token_id": processor.eos_id(),
        "pad_token_id": processor.pad_id(),
        "unk_token_id": processor.unk_id(),
        "cls_token_id": sentencepiece_piece_id(processor, "[CLS]"),
        "sep_token_id": sentencepiece_piece_id(processor, "[SEP]"),
        "mask_token_id": sentencepiece_piece_id(processor, "[MASK]"),
    }
    return {key: (value if value is not None and value >= 0 else None) for key, value in ids.items()}


def tokenizer_vocab_size(tokenizer_path: Path) -> int:
    """Return the number of pieces in a SentencePiece model."""
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return processor.get_piece_size()


def build_config(checkpoint: dict, tokenizer_path: Path) -> PinyinCodeConfig:
    """Create the HF config from the original model config and tokenizer ids."""
    model_config = dict(checkpoint["model_config"])
    training_model_type = model_config.pop("model_type", "gpt")
    if training_model_type not in {"gpt", "bert"}:
        raise ValueError(f"Unsupported checkpoint model_type: {training_model_type}")

    checkpoint_vocab_size = int(model_config["vocab_size"])
    actual_vocab_size = tokenizer_vocab_size(tokenizer_path)
    if checkpoint_vocab_size != actual_vocab_size:
        raise ValueError(
            "Tokenizer vocab size does not match checkpoint config: "
            f"tokenizer={actual_vocab_size}, checkpoint={checkpoint_vocab_size}."
        )

    special_ids = tokenizer_special_ids(tokenizer_path)
    if training_model_type == "bert":
        missing = [
            name
            for name in ("pad_token_id", "unk_token_id", "cls_token_id", "sep_token_id", "mask_token_id")
            if special_ids.get(name) is None
        ]
        if missing:
            raise ValueError(
                "BERT/MLM export requires tokenizer pieces [PAD], [UNK], [CLS], [SEP], "
                f"and [MASK]. Missing ids: {', '.join(missing)}."
            )

    config = PinyinCodeConfig(
        **model_config,
        **special_ids,
        training_model_type=training_model_type,
        recommended_score_normalization="mean" if training_model_type == "bert" else "sum",
    )
    if training_model_type == "bert":
        config.architectures = ["PinyinCodeForMaskedLM"]
        config.evaluation_backend = "masked_language_modeling"
    else:
        config.architectures = ["PinyinCodeForCausalLM"]
        config.evaluation_backend = "causal"
    config.patch_pathlib_utf8_open = True
    config.auto_map = {
        "AutoConfig": "configuration_pinyin_code.PinyinCodeConfig",
        "AutoModel": (
            "modeling_pinyin_code.PinyinCodeEncoderModel"
            if training_model_type == "bert"
            else "modeling_pinyin_code.PinyinCodeModel"
        ),
        "AutoTokenizer": ["tokenization_pinyin_code.EncodedMandarinTokenizer", None],
    }
    if training_model_type == "bert":
        config.auto_map["AutoModelForMaskedLM"] = "modeling_pinyin_code.PinyinCodeForMaskedLM"
    else:
        config.auto_map["AutoModelForCausalLM"] = "modeling_pinyin_code.PinyinCodeForCausalLM"
    return config


def convert_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map original train_model.py keys onto the HF module names."""
    mapped = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("module.")
        mapped[new_key] = value
    return mapped


def copy_remote_code(output_dir: Path) -> None:
    """Copy custom modeling code into the saved model folder."""
    package_dir = output_dir / "hf"
    package_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_dir = output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    (preprocessing_dir / "__init__.py").write_text("", encoding="utf-8")
    for filename in (
        "__init__.py",
        "configuration_pinyin_code.py",
        "modeling_pinyin_code.py",
        "tokenization_pinyin_code.py",
    ):
        shutil.copy2(Path("hf") / filename, package_dir / filename)
        if filename != "__init__.py":
            shutil.copy2(Path("hf") / filename, output_dir / filename)
    shutil.copy2(Path("preprocessing") / "preprocess.py", preprocessing_dir / "preprocess.py")


def patch_json(path: Path, updates: dict) -> None:
    """Merge updates into a JSON file written by save_pretrained."""
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_model_readme(
    output_dir: Path,
    transliteration: str,
    use_jieba: bool,
    training_model_type: str,
) -> None:
    """Write a minimal model-card README for external evaluation users."""
    if training_model_type == "bert":
        pipeline_tag = "fill-mask"
        tag = "masked-lm"
        title = "Pinyin-Code Masked LM"
        model_kind = "custom Transformers masked language model"
        loading_import = "AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer"
        loading_model = "model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)"
        backend = "masked_language_modeling"
        evaluation_note = (
            "For BLiMP-style sentence-pair scoring, use pseudo-log-likelihood "
            "rather than left-to-right probability. Mean-normalize token "
            "log-probabilities when candidate lengths can differ; this requires "
            "one forward pass per scored token."
        )
    else:
        pipeline_tag = "text-generation"
        tag = "causal-lm"
        title = "Pinyin-Code Causal LM"
        model_kind = "custom Transformers causal language model"
        loading_import = "AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer"
        loading_model = "model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)"
        backend = "causal"
        evaluation_note = "External evaluation repositories should select the causal backend."

    text = dedent(
        f"""\
        ---
        library_name: transformers
        pipeline_tag: {pipeline_tag}
        tags:
        - {tag}
        - trust-remote-code
        - sentencepiece
        ---

        # {title}

        This repository contains a {model_kind}. Load it with
        `trust_remote_code=True`.

        ## Dependencies

        Install the runtime dependencies before loading the model:

        ```bash
        pip install torch transformers safetensors sentencepiece pypinyin jieba
        ```

        `sentencepiece` is required for `AutoTokenizer`. `pypinyin` is required
        for raw Mandarin-to-pinyin tokenization. `jieba` is required when
        `use_jieba` is true; this export was created with `use_jieba={str(use_jieba).lower()}`.

        ## Loading

        ```python
        from transformers import {loading_import}

        model_path = "PATH_OR_REPO_ID"

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        {loading_model}
        ```

        ## Evaluation

        Configure external evaluators with:

        - model path: this local folder or Hugging Face repo ID
        - backend: `{backend}`
        - trust remote code: enabled

        {evaluation_note}

        The tokenizer accepts raw text through standard calls such as
        `tokenizer(text)`, `tokenizer(text, add_special_tokens=False)`, and
        `tokenizer(texts, padding=True, truncation=True, return_tensors="pt")`.
        It also accepts `return_offsets_mapping=True` for compatibility with
        completion-ranking evaluators that need suffix masks. The model supports
        `output_hidden_states=True` for representation extraction tasks.

        This export sets `patch_pathlib_utf8_open=true` in `config.json`.
        When loaded with `trust_remote_code=True`, the config installs a narrow
        Windows compatibility shim so later text-mode `Path.open("r")` calls
        without an explicit encoding default to UTF-8. Set
        `PINYIN_CODE_DISABLE_UTF8_OPEN_PATCH=1` before loading the model to
        disable that shim.

        Export metadata:

        - transliteration: `{transliteration}`
        - training_model_type: `{training_model_type}`
        - use_jieba: `{str(use_jieba).lower()}`
        """
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def convert(args: argparse.Namespace) -> None:
    """Convert and save the model, tokenizer, config, and remote-code files."""
    checkpoint = load_training_checkpoint(args.checkpoint)
    config = build_config(checkpoint, args.tokenizer)
    training_model_type = config.training_model_type
    model = PinyinCodeForMaskedLM(config) if training_model_type == "bert" else PinyinCodeForCausalLM(config)
    load_result = model.load_state_dict(
        convert_state_dict(checkpoint["model_state_dict"]),
        strict=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint key mismatch: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    model.eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy_remote_code(args.output_dir)
    model.save_pretrained(args.output_dir, safe_serialization=args.safe_serialization)

    tokenizer = EncodedMandarinTokenizer(
        vocab_file=str(args.tokenizer),
        transliteration=args.transliteration,
        use_jieba=args.jieba,
    )
    tokenizer.save_pretrained(args.output_dir)
    tokenizer_vocab = args.tokenizer.with_suffix(".vocab")
    if tokenizer_vocab.exists():
        shutil.copy2(tokenizer_vocab, args.output_dir / tokenizer_vocab.name)

    if training_model_type == "gpt":
        generation_config = GenerationConfig(
            bos_token_id=config.bos_token_id,
            eos_token_id=config.eos_token_id,
            pad_token_id=config.pad_token_id,
        )
        generation_config.save_pretrained(args.output_dir)

    patch_json(
        args.output_dir / "tokenizer_config.json",
        {
            "auto_map": {
                "AutoTokenizer": [
                    "tokenization_pinyin_code.EncodedMandarinTokenizer",
                    None,
                ]
            },
            "model_max_length": config.block_size,
            "pinyin_format": args.transliteration,
            "jieba": args.jieba,
            "tokenizer_class": "EncodedMandarinTokenizer",
            "transliteration": args.transliteration,
            "use_jieba": args.jieba,
        },
    )
    special_tokens_map = {
        key: value
        for key, value in tokenizer.special_tokens_map.items()
        if value is not None
    }
    (args.output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "evaluation_backend": config.evaluation_backend,
        "recommended_score_normalization": config.recommended_score_normalization,
        "source_checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "jieba": args.jieba,
        "model_type": training_model_type,
        "transliteration": args.transliteration,
        "use_jieba": args.jieba,
        "validation_loss": checkpoint.get("validation_loss"),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_model_readme(args.output_dir, args.transliteration, args.jieba, training_model_type)
    print(f"Saved Transformers model to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the pinyin-code checkpoint into a Transformers model folder."
    )
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--transliteration",
        choices=("pinyin-code", "pinyin-initial", "hanzi"),
        default="pinyin-code",
        help="Preprocessing mode used to train this tokenizer/model.",
    )
    parser.add_argument(
        "--jieba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Whether the training corpus used jieba word segmentation. Pass "
            "--no-jieba for character-level Chinese preprocessing."
        ),
    )
    parser.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write model.safetensors by default. Use --no-safe-serialization "
            "to write pytorch_model.bin instead."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
