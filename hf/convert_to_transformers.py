"""Convert the trained custom PyTorch checkpoint to a Transformers model repo."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from transformers import GenerationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hf.configuration_pinyin_code import PinyinCodeConfig
from hf.modeling_pinyin_code import PinyinCodeForCausalLM
from hf.tokenization_pinyin_code import PinyinCodeTokenizer


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


def tokenizer_special_ids(tokenizer_path: Path) -> dict[str, int | None]:
    """Read special token ids from the existing SentencePiece model."""
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    ids = {
        "bos_token_id": processor.bos_id(),
        "eos_token_id": processor.eos_id(),
        "pad_token_id": processor.pad_id(),
        "unk_token_id": processor.unk_id(),
    }
    return {key: (value if value >= 0 else None) for key, value in ids.items()}


def tokenizer_vocab_size(tokenizer_path: Path) -> int:
    """Return the number of pieces in a SentencePiece model."""
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return processor.get_piece_size()


def build_config(checkpoint: dict, tokenizer_path: Path) -> PinyinCodeConfig:
    """Create the HF config from the original model config and tokenizer ids."""
    checkpoint_vocab_size = int(checkpoint["model_config"]["vocab_size"])
    actual_vocab_size = tokenizer_vocab_size(tokenizer_path)
    if checkpoint_vocab_size != actual_vocab_size:
        raise ValueError(
            "Tokenizer vocab size does not match checkpoint config: "
            f"tokenizer={actual_vocab_size}, checkpoint={checkpoint_vocab_size}."
        )

    config = PinyinCodeConfig(
        **checkpoint["model_config"],
        **tokenizer_special_ids(tokenizer_path),
    )
    config.architectures = ["PinyinCodeForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_pinyin_code.PinyinCodeConfig",
        "AutoModelForCausalLM": "modeling_pinyin_code.PinyinCodeForCausalLM",
        "AutoTokenizer": ["tokenization_pinyin_code.PinyinCodeTokenizer", None],
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


def convert(args: argparse.Namespace) -> None:
    """Convert and save the model, tokenizer, config, and remote-code files."""
    checkpoint = load_training_checkpoint(args.checkpoint)
    config = build_config(checkpoint, args.tokenizer)
    model = PinyinCodeForCausalLM(config)
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

    tokenizer = PinyinCodeTokenizer(
        vocab_file=str(args.tokenizer),
        transliteration=args.transliteration,
        use_jieba=args.jieba,
    )
    tokenizer.save_pretrained(args.output_dir)
    tokenizer_vocab = args.tokenizer.with_suffix(".vocab")
    if tokenizer_vocab.exists():
        shutil.copy2(tokenizer_vocab, args.output_dir / tokenizer_vocab.name)

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
                "AutoTokenizer": ["tokenization_pinyin_code.PinyinCodeTokenizer", None]
            },
            "model_max_length": config.block_size,
            "pinyin_format": args.transliteration,
            "jieba": args.jieba,
            "tokenizer_class": "PinyinCodeTokenizer",
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
        "source_checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "global_step": checkpoint.get("global_step"),
        "jieba": args.jieba,
        "transliteration": args.transliteration,
        "use_jieba": args.jieba,
        "validation_loss": checkpoint.get("validation_loss"),
    }
    (args.output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
