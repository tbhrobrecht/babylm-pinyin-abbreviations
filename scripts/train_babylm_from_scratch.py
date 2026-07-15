"""Run the end-to-end BabyLM pinyin-code training pipeline.

This is a thin orchestrator around the commands documented in the README TLDR:
extract, preprocess, train a tokenizer, create binary datasets, train the
PyTorch model, and optionally convert the best checkpoint to a Transformers
folder. Hybrid tokenization is the default; pass --tokenizer-kind bpe for the
older SentencePiece path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def path_arg(path: Path) -> str:
    """Return a stable command argument for repo-relative paths."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def default_paths(model_name: str, args: argparse.Namespace) -> None:
    """Fill derived paths after parsing model_name."""
    dataset_suffix = "hybrid" if args.tokenizer_kind == "hybrid" else "spm"
    if args.raw_output is None:
        args.raw_output = Path("data") / f"{model_name}.jsonl"
    if args.processed_output is None:
        args.processed_output = Path("data/processed") / f"{model_name}.txt"
    if args.tokenizer_name is None:
        args.tokenizer_name = f"{model_name}_hybrid" if args.tokenizer_kind == "hybrid" else model_name
    if args.train_dataset is None:
        args.train_dataset = Path("data/datasets") / f"{model_name}_train_{dataset_suffix}.bin"
    if args.validation_dataset is None:
        args.validation_dataset = Path("data/datasets") / f"{model_name}_valid_{dataset_suffix}.bin"
    if args.model_output_dir is None:
        args.model_output_dir = Path("models") / model_name
    if args.hf_output_dir is None:
        args.hf_output_dir = Path(f"hf_{model_name}")


def tokenizer_path(args: argparse.Namespace) -> Path:
    if args.tokenizer_kind == "hybrid":
        return args.tokenizer_dir / args.tokenizer_name
    return args.tokenizer_dir / f"{args.tokenizer_name}.model"


def tokenizer_done_path(args: argparse.Namespace) -> Path:
    if args.tokenizer_kind == "hybrid":
        return tokenizer_path(args) / "vocab.json"
    return tokenizer_path(args)


def training_vocab_size(args: argparse.Namespace) -> int:
    if args.tokenizer_kind != "hybrid":
        return args.vocab_size
    vocab_path = tokenizer_done_path(args)
    if not vocab_path.exists():
        return args.vocab_size
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if not isinstance(vocab, dict):
        raise ValueError(f"{vocab_path} must contain a JSON object")
    return len(vocab)


def checkpoint_path(args: argparse.Namespace) -> Path:
    return args.model_output_dir / "best.pt"


def converted_config_path(args: argparse.Namespace) -> Path:
    return args.hf_output_dir / "config.json"


def existing_output_for_step(step: str, args: argparse.Namespace) -> Path:
    outputs = {
        "extract": args.raw_output,
        "preprocess": args.processed_output,
        "tokenizer": tokenizer_done_path(args),
        "dataset": args.train_dataset,
        "train": checkpoint_path(args),
        "convert": converted_config_path(args),
    }
    return outputs[step]


def should_skip(step: str, args: argparse.Namespace) -> bool:
    if getattr(args, f"skip_{step}"):
        return True
    output = existing_output_for_step(step, args)
    return args.resume and output.exists()


def run_command(label: str, command: list[str], args: argparse.Namespace) -> None:
    print()
    print(f"==> {label}")
    print(subprocess.list2cmdline(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def python_command(script: str, *items: object) -> list[str]:
    return [sys.executable, script, *(str(item) for item in items)]


def run_extract(args: argparse.Namespace) -> None:
    if should_skip("extract", args):
        print(f"Skipping extract; output is {path_arg(args.raw_output)}")
        return

    command = python_command(
        "preprocessing/extract_babylm_zho.py",
        "--output",
        path_arg(args.raw_output),
        "--split",
        args.split,
    )
    if args.max_docs is not None:
        command += ["--max-docs", str(args.max_docs)]
    for category in args.category:
        command += ["--category", category]
    for script_name in args.script_filter:
        command += ["--script", script_name]
    for language in args.language:
        command += ["--language", language]
    if args.text_only:
        command.append("--text-only")
    if not args.streaming:
        command.append("--no-streaming")
    run_command("Extract BabyLM zho", command, args)


def run_preprocess(args: argparse.Namespace) -> None:
    if should_skip("preprocess", args):
        print(f"Skipping preprocess; output is {path_arg(args.processed_output)}")
        return

    command = python_command(
        "preprocessing/preprocess.py",
        "--input",
        path_arg(args.raw_output),
        "--output",
        path_arg(args.processed_output),
        "--transliteration",
        args.transliteration,
    )
    if not args.jieba:
        command.append("--no-jieba")
    run_command("Preprocess corpus", command, args)


def run_tokenizer(args: argparse.Namespace) -> None:
    if should_skip("tokenizer", args):
        print(f"Skipping tokenizer; output is {path_arg(tokenizer_done_path(args))}")
        return

    if args.tokenizer_kind == "hybrid":
        command = python_command(
            "train_hybrid_tokenizer.py",
            "--input",
            path_arg(args.processed_output),
            "--output-dir",
            path_arg(tokenizer_path(args)),
            "--vocab-size",
            args.vocab_size,
            "--min-word-frequency",
            args.hybrid_min_word_frequency,
        )
        if args.hybrid_atomic_only:
            command.append("--atomic-only")
        if args.hybrid_permissive:
            command.append("--permissive")
        run_command("Train hybrid tokenizer", command, args)
        return

    command = python_command(
        "train_sentencepiece.py",
        "--input",
        path_arg(args.processed_output),
        "--output-dir",
        path_arg(args.tokenizer_dir),
        "--model-name",
        args.tokenizer_name,
        "--vocab-size",
        args.vocab_size,
        "--model-type",
        args.sentencepiece_model_type,
    )
    run_command("Train SentencePiece tokenizer", command, args)


def run_dataset(args: argparse.Namespace) -> None:
    if should_skip("dataset", args):
        print(f"Skipping dataset; output is {path_arg(args.train_dataset)}")
        return

    command = python_command(
        "create_dataset.py",
        "--format",
        "bin",
        "--input",
        path_arg(args.processed_output),
        "--output",
        path_arg(args.train_dataset),
        "--validation-output",
        path_arg(args.validation_dataset),
        "--validation-fraction",
        args.validation_fraction,
        "--tokenizer",
        path_arg(tokenizer_path(args)),
        "--block-size",
        args.block_size,
        "--stride",
        args.stride,
        "--seed",
        args.seed,
    )
    run_command("Create train/validation datasets", command, args)


def run_training(args: argparse.Namespace) -> None:
    if should_skip("train", args):
        print(f"Skipping train; output is {path_arg(checkpoint_path(args))}")
        return

    command = python_command(
        "train_model.py",
        "--dataset",
        path_arg(args.train_dataset),
        "--validation-dataset",
        path_arg(args.validation_dataset),
        "--output-dir",
        path_arg(args.model_output_dir),
        "--vocab-size",
        training_vocab_size(args),
        "--block-size",
        args.block_size,
        "--n-layer",
        args.n_layer,
        "--n-head",
        args.n_head,
        "--n-embd",
        args.n_embd,
        "--epochs",
        args.epochs,
        "--batch-size",
        args.batch_size,
        "--learning-rate",
        args.learning_rate,
        "--gradient-accumulation-steps",
        args.gradient_accumulation_steps,
        "--seed",
        args.seed,
    )
    if args.device is not None:
        command += ["--device", args.device]
    run_command("Train language model", command, args)


def run_convert(args: argparse.Namespace) -> None:
    if should_skip("convert", args):
        print(f"Skipping convert; output is {path_arg(converted_config_path(args))}")
        return

    command = python_command(
        "hf/convert_to_transformers.py",
        "--checkpoint",
        path_arg(checkpoint_path(args)),
        "--tokenizer",
        path_arg(tokenizer_path(args)),
        "--output-dir",
        path_arg(args.hf_output_dir),
        "--transliteration",
        args.transliteration,
    )
    if not args.jieba:
        command.append("--no-jieba")
    run_command("Convert to Transformers folder", command, args)


def run_upload(args: argparse.Namespace) -> None:
    if args.hf_repo is None:
        return
    command = [
        "hf",
        "upload",
        args.hf_repo,
        path_arg(args.hf_output_dir),
        "--repo-type",
        "model",
    ]
    run_command("Upload Transformers folder to Hugging Face", command, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the README TLDR BabyLM pinyin-code pipeline end to end."
    )
    parser.add_argument(
        "--model-name",
        required=True,
        help=(
            "Run name used for tokenizer, datasets, checkpoint directory, and "
            "Transformers export unless those paths are overridden."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip a step when that step's expected output already exists.",
    )

    extract = parser.add_argument_group("extract")
    extract.add_argument("--raw-output", type=Path, default=None)
    extract.add_argument("--split", default="train")
    extract.add_argument("--max-docs", type=int, default=None)
    extract.add_argument("--category", action="append", default=[])
    extract.add_argument("--script-filter", action="append", default=[])
    extract.add_argument("--language", action="append", default=[])
    extract.add_argument("--text-only", action="store_true")
    extract.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forward streaming mode to preprocessing/extract_babylm_zho.py.",
    )

    preprocess = parser.add_argument_group("preprocess")
    preprocess.add_argument("--processed-output", type=Path, default=None)
    preprocess.add_argument(
        "--transliteration",
        choices=("pinyin-code", "pinyin-initial", "hanzi"),
        default="pinyin-code",
    )
    preprocess.add_argument("--jieba", action=argparse.BooleanOptionalAction, default=True)

    tokenizer = parser.add_argument_group("tokenizer")
    tokenizer.add_argument(
        "--tokenizer-kind",
        choices=("hybrid", "bpe"),
        default="hybrid",
        help="Tokenizer family to train. Hybrid is the default; bpe uses SentencePiece.",
    )
    tokenizer.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizers"))
    tokenizer.add_argument("--tokenizer-name", default=None)
    tokenizer.add_argument("--vocab-size", type=int, default=16000)
    tokenizer.add_argument(
        "--sentencepiece-model-type",
        choices=("bpe", "unigram", "char", "word"),
        default="bpe",
    )
    tokenizer.add_argument("--hybrid-min-word-frequency", type=int, default=20)
    tokenizer.add_argument("--hybrid-atomic-only", action="store_true")
    tokenizer.add_argument("--hybrid-permissive", action="store_true")

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--train-dataset", type=Path, default=None)
    dataset.add_argument("--validation-dataset", type=Path, default=None)
    dataset.add_argument("--validation-fraction", type=float, default=0.05)
    dataset.add_argument("--block-size", type=int, default=512)
    dataset.add_argument("--stride", type=int, default=512)

    training = parser.add_argument_group("training")
    training.add_argument("--model-output-dir", type=Path, default=None)
    training.add_argument("--n-layer", type=int, default=8)
    training.add_argument("--n-head", type=int, default=8)
    training.add_argument("--n-embd", type=int, default=512)
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--gradient-accumulation-steps", type=int, default=1)
    training.add_argument("--device", choices=("cpu", "cuda"), default=None)
    training.add_argument("--seed", type=int, default=1337)

    convert = parser.add_argument_group("convert/upload")
    convert.add_argument("--hf-output-dir", type=Path, default=None)
    convert.add_argument(
        "--hf-repo",
        default=None,
        help="Optional Hugging Face repo id to upload after conversion, e.g. user/model.",
    )

    skips = parser.add_argument_group("skip steps")
    skips.add_argument("--skip-extract", action="store_true")
    skips.add_argument("--skip-preprocess", action="store_true")
    skips.add_argument("--skip-tokenizer", action="store_true")
    skips.add_argument("--skip-dataset", action="store_true")
    skips.add_argument("--skip-train", action="store_true")
    skips.add_argument("--skip-convert", action="store_true")

    args = parser.parse_args()
    default_paths(args.model_name, args)
    return args


def main() -> None:
    args = parse_args()
    run_extract(args)
    run_preprocess(args)
    run_tokenizer(args)
    run_dataset(args)
    run_training(args)
    run_convert(args)
    run_upload(args)
    print()
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
