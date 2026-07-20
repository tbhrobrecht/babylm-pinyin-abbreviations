"""Convert the trained custom PyTorch checkpoint to a Transformers model repo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from textwrap import dedent

import torch
from transformers import GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hf.configuration_pinyin_code import PinyinCodeConfig
from hf.modeling_pinyin_code import PinyinCodeForCausalLM
from hf.tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer
from train_model import (
    build_model,
    checkpoint_architecture,
    model_config_from_checkpoint,
)


DEFAULT_OUTPUT_DIR = Path("hf_pinyin_code_model")
DEFAULT_TOKENIZER = Path("tokenizers/babylm_zho_pinyin_spm.model")
DEFAULT_CHECKPOINT_CANDIDATES = (
    Path("models/pinyin-code-gpt-small/best.pt"),
    Path("models/pinyin-code-gpt-small/best_model.pt"),
)
HYBRID_METADATA_NAME = "hybrid_tokenizer_metadata.json"


def require_sentencepiece():
    """Import SentencePiece only when converting a SentencePiece tokenizer."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for SentencePiece conversion: install it with "
            "`py -m pip install sentencepiece`."
        ) from exc

    return spm


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


def is_hybrid_tokenizer(tokenizer_path: Path) -> bool:
    """Return true when the tokenizer path points at a hybrid tokenizer."""
    if tokenizer_path.name == "vocab.json":
        return True
    return tokenizer_path.is_dir() and (tokenizer_path / "vocab.json").exists()


def hybrid_tokenizer_dir(tokenizer_path: Path) -> Path:
    """Normalize a hybrid tokenizer path to the directory passed to from_pretrained."""
    if tokenizer_path.name == "vocab.json":
        return tokenizer_path.parent
    return tokenizer_path


def tokenizer_special_ids(tokenizer_path: Path) -> dict[str, int | None]:
    """Read special token ids from the existing tokenizer."""
    if is_hybrid_tokenizer(tokenizer_path):
        tokenizer = HybridPinyinCodeTokenizer.from_pretrained(
            hybrid_tokenizer_dir(tokenizer_path)
        )
        return {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        }

    spm = require_sentencepiece()
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    ids = {
        "bos_token_id": processor.bos_id(),
        "eos_token_id": processor.eos_id(),
        "pad_token_id": processor.pad_id(),
        "unk_token_id": processor.unk_id(),
    }
    return {key: (value if value >= 0 else None) for key, value in ids.items()}


def tokenizer_vocab_size(tokenizer_path: Path) -> int:
    """Return the vocabulary size of a SentencePiece or hybrid tokenizer."""
    if is_hybrid_tokenizer(tokenizer_path):
        tokenizer = HybridPinyinCodeTokenizer.from_pretrained(
            hybrid_tokenizer_dir(tokenizer_path)
        )
        return tokenizer.vocab_size

    spm = require_sentencepiece()
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return processor.get_piece_size()


def tokenizer_auto_map(tokenizer_path: Path) -> list[str | None]:
    """Return the AutoTokenizer remote-code target for the tokenizer family."""
    if is_hybrid_tokenizer(tokenizer_path):
        return ["tokenization_hybrid_pinyin_code.HybridPinyinCodeTokenizer", None]
    return ["tokenization_pinyin_code.EncodedMandarinTokenizer", None]


def build_config(checkpoint: dict, tokenizer_path: Path) -> PinyinCodeConfig:
    """Create the HF config from the original model config and tokenizer ids."""
    checkpoint_vocab_size = int(checkpoint["model_config"]["vocab_size"])
    actual_vocab_size = tokenizer_vocab_size(tokenizer_path)
    if checkpoint_vocab_size != actual_vocab_size:
        raise ValueError(
            "Tokenizer vocab size does not match checkpoint config: "
            f"tokenizer={actual_vocab_size}, checkpoint={checkpoint_vocab_size}."
        )

    model_config = {
        key: value
        for key, value in checkpoint["model_config"].items()
        if key
        not in {
            "architecture",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
        }
    }
    config = PinyinCodeConfig(
        **model_config,
        **tokenizer_special_ids(tokenizer_path),
    )
    config.architectures = ["PinyinCodeForCausalLM"]
    config.evaluation_backend = "causal"
    config.patch_pathlib_utf8_open = True
    config.auto_map = {
        "AutoConfig": "configuration_pinyin_code.PinyinCodeConfig",
        "AutoModel": "modeling_pinyin_code.PinyinCodeModel",
        "AutoModelForCausalLM": "modeling_pinyin_code.PinyinCodeForCausalLM",
        "AutoModelForSequenceClassification": (
            "modeling_pinyin_code.PinyinCodeForSequenceClassification"
        ),
        "AutoTokenizer": tokenizer_auto_map(tokenizer_path),
    }
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
        "tokenization_hybrid_pinyin_code.py",
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
    tokenizer_kind: str,
) -> None:
    """Write a minimal model-card README for external evaluation users."""
    tokenizer_tag = "sentencepiece" if tokenizer_kind == "sentencepiece" else "hybrid-tokenizer"
    install_command = (
        "pip install torch transformers safetensors sentencepiece pypinyin jieba"
        if tokenizer_kind == "sentencepiece"
        else "pip install torch transformers safetensors pypinyin jieba"
    )
    dependency_note = (
        "`sentencepiece` is required for the SentencePiece tokenizer. "
        if tokenizer_kind == "sentencepiece"
        else "This export uses the repository's hybrid tokenizer and does not require SentencePiece. "
    )
    tokenizer_note = (
        "The tokenizer accepts raw text through standard calls such as\n"
        "        `tokenizer(text)`, `tokenizer(text, add_special_tokens=False)`, and\n"
        "        `tokenizer(texts, padding=True, truncation=True, return_tensors=\"pt\")`.\n"
        "        It also accepts `return_offsets_mapping=True` for compatibility with\n"
        "        completion-ranking evaluators that need suffix masks."
        if tokenizer_kind == "sentencepiece"
        else "The hybrid tokenizer accepts preprocessed pinyin-code text through standard\n"
        "        calls such as `tokenizer(text)`, `tokenizer(text, add_special_tokens=False)`,\n"
        "        and `tokenizer(texts, padding=True, truncation=True, return_tensors=\"pt\")`."
    )
    text = dedent(
        f"""\
        ---
        library_name: transformers
        pipeline_tag: text-generation
        tags:
        - causal-lm
        - trust-remote-code
        - {tokenizer_tag}
        ---

        # Pinyin-Code Causal LM

        This repository contains a custom Transformers causal language model.
        External evaluation repositories should load it with
        `trust_remote_code=True` and use the `causal` backend.

        ## Dependencies

        Install the runtime dependencies before loading the model:

        ```bash
        {install_command}
        ```

        {dependency_note}`pypinyin` is required
        for raw Mandarin-to-pinyin preprocessing. `jieba` is required when
        `use_jieba` is true; this export was created with `use_jieba={str(use_jieba).lower()}`.

        ## Loading

        ```python
        from transformers import (
            AutoConfig,
            AutoModel,
            AutoModelForCausalLM,
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        model_path = "PATH_OR_REPO_ID"

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        classifier = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            num_labels=3,
        )
        ```

        ## Evaluation

        Configure external evaluators with:

        - model path: this local folder or Hugging Face repo ID
        - backend: `causal`
        - trust remote code: enabled

        {tokenizer_note} The model supports
        `output_hidden_states=True` for representation extraction tasks.

        This export sets `patch_pathlib_utf8_open=true` in `config.json`.
        When loaded with `trust_remote_code=True`, the config installs a narrow
        Windows compatibility shim so later text-mode `Path.open("r")` calls
        without an explicit encoding default to UTF-8. Set
        `PINYIN_CODE_DISABLE_UTF8_OPEN_PATCH=1` before loading the model to
        disable that shim.

        Export metadata:

        - tokenizer_kind: `{tokenizer_kind}`
        - transliteration: `{transliteration}`
        - use_jieba: `{str(use_jieba).lower()}`
        """
    )
    (output_dir / "README.md").write_text(text, encoding="utf-8", newline="\n")


def load_export_tokenizer(args: argparse.Namespace):
    """Load the tokenizer implementation matching the input tokenizer path."""
    if is_hybrid_tokenizer(args.tokenizer):
        return HybridPinyinCodeTokenizer.from_pretrained(
            hybrid_tokenizer_dir(args.tokenizer)
        )
    from hf.tokenization_pinyin_code import EncodedMandarinTokenizer

    return EncodedMandarinTokenizer(
        vocab_file=str(args.tokenizer),
        transliteration=args.transliteration,
        use_jieba=args.jieba,
    )


def copy_tokenizer_sidecars(tokenizer_path: Path, output_dir: Path) -> None:
    """Copy optional tokenizer metadata files that save_pretrained does not own."""
    if is_hybrid_tokenizer(tokenizer_path):
        metadata_path = hybrid_tokenizer_dir(tokenizer_path) / HYBRID_METADATA_NAME
        if metadata_path.exists():
            shutil.copy2(metadata_path, output_dir / metadata_path.name)
        return

    tokenizer_vocab = tokenizer_path.with_suffix(".vocab")
    if tokenizer_vocab.exists():
        shutil.copy2(tokenizer_vocab, output_dir / tokenizer_vocab.name)


def tokenizer_config_updates(args: argparse.Namespace, config) -> dict:
    """Return tokenizer_config.json updates for the exported tokenizer."""
    model_max_length = getattr(config, "block_size", None)
    if model_max_length is None:
        model_max_length = getattr(config, "max_position_embeddings", None)
    updates = {
        "auto_map": {"AutoTokenizer": tokenizer_auto_map(args.tokenizer)},
        "model_max_length": model_max_length,
        "pinyin_format": args.transliteration,
        "jieba": args.jieba,
        "tokenizer_kind": "hybrid" if is_hybrid_tokenizer(args.tokenizer) else "sentencepiece",
        "transliteration": args.transliteration,
        "use_jieba": args.jieba,
    }
    if is_hybrid_tokenizer(args.tokenizer):
        updates["tokenizer_class"] = "HybridPinyinCodeTokenizer"
    else:
        updates["tokenizer_class"] = "EncodedMandarinTokenizer"
    return updates


def convert(args: argparse.Namespace) -> None:
    """Convert and save the model, tokenizer, config, and remote-code files."""
    checkpoint = load_training_checkpoint(args.checkpoint)
    architecture = checkpoint_architecture(checkpoint)
    if architecture == "gpt2":
        config = build_config(checkpoint, args.tokenizer)
        model = PinyinCodeForCausalLM(config)
    elif architecture == "qwen2":
        model_config = model_config_from_checkpoint(checkpoint)
        checkpoint_vocab_size = int(model_config.vocab_size)
        actual_vocab_size = tokenizer_vocab_size(args.tokenizer)
        if checkpoint_vocab_size != actual_vocab_size:
            raise ValueError(
                "Tokenizer vocab size does not match checkpoint config: "
                f"tokenizer={actual_vocab_size}, checkpoint={checkpoint_vocab_size}."
            )
        model = build_model(model_config)
        special_ids = tokenizer_special_ids(args.tokenizer)
        for key, value in special_ids.items():
            setattr(model.config, key, value)
        config = model.config
    else:
        raise AssertionError(f"Unhandled architecture: {architecture}")

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

    tokenizer = load_export_tokenizer(args)
    tokenizer.save_pretrained(args.output_dir)
    copy_tokenizer_sidecars(args.tokenizer, args.output_dir)

    generation_config = GenerationConfig(
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
    )
    generation_config.save_pretrained(args.output_dir)

    patch_json(
        args.output_dir / "tokenizer_config.json",
        {
            **tokenizer_config_updates(args, config),
        },
    )
    if architecture == "qwen2":
        patch_json(
            args.output_dir / "config.json",
            {
                "auto_map": {"AutoTokenizer": tokenizer_auto_map(args.tokenizer)},
                "evaluation_backend": "causal",
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
        "evaluation_backend": "causal",
        "architecture": architecture,
        "source_checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "jieba": args.jieba,
        "tokenizer_kind": "hybrid" if is_hybrid_tokenizer(args.tokenizer) else "sentencepiece",
        "transliteration": args.transliteration,
        "use_jieba": args.jieba,
        "validation_loss": checkpoint.get("validation_loss"),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_model_readme(
        args.output_dir,
        args.transliteration,
        args.jieba,
        "hybrid" if is_hybrid_tokenizer(args.tokenizer) else "sentencepiece",
    )
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
