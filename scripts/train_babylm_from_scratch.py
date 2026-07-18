"""Run the end-to-end BabyLM pinyin-code training pipeline.

This is a thin orchestrator around the commands documented in the README TLDR:
extract, preprocess, train a tokenizer, create binary datasets, train the
PyTorch model, and optionally convert the best checkpoint to a Transformers
folder. Hybrid tokenization is the default; pass --tokenizer-kind bpe for the
older SentencePiece path.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_STEPS = (
    "extract",
    "preprocess",
    "tokenizer",
    "dataset",
    "train",
    "convert",
    "upload",
)


def path_arg(path: Path) -> str:
    """Return a stable command argument for repo-relative paths."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def format_command(command: list[str]) -> str:
    """Return a shell-readable command line for logging."""
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def default_tokenizer_name(
    model_name: str,
    tokenizer_kind: str,
    matrix_mode: bool,
) -> str:
    """Return the default tokenizer artifact name for this run."""
    if matrix_mode:
        return f"{model_name}_{tokenizer_kind}"
    return f"{model_name}_hybrid" if tokenizer_kind == "hybrid" else model_name


def default_paths(
    corpus_name: str,
    model_name: str,
    run_name: str,
    matrix_mode: bool,
    args: argparse.Namespace,
) -> None:
    """Fill derived paths after parsing model_name."""
    dataset_suffix = "hybrid" if args.tokenizer_kind == "hybrid" else "spm"
    if args.raw_output is None:
        args.raw_output = Path("data") / f"{corpus_name}.jsonl"
    if args.processed_output is None:
        args.processed_output = Path("data/processed") / f"{corpus_name}.txt"
    if args.tokenizer_name is None:
        args.tokenizer_name = default_tokenizer_name(
            model_name,
            args.tokenizer_kind,
            matrix_mode,
        )
    dataset_name = f"{model_name}_{args.tokenizer_kind}" if matrix_mode else model_name
    if args.train_dataset is None:
        args.train_dataset = Path("data/datasets") / f"{dataset_name}_train_{dataset_suffix}.bin"
    if args.validation_dataset is None:
        args.validation_dataset = Path("data/datasets") / f"{dataset_name}_valid_{dataset_suffix}.bin"
    if args.model_output_dir is None:
        args.model_output_dir = Path("models") / run_name
    if args.hf_output_dir is None:
        args.hf_output_dir = Path(f"hf_{run_name}")


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
    if (step, str(output)) in getattr(args, "completed_outputs", set()):
        return True
    return args.resume and output.exists()


def step_selected(step: str, args: argparse.Namespace) -> bool:
    """Return true when a step falls inside the requested step range."""
    start_index = PIPELINE_STEPS.index(args.start_at)
    stop_index = PIPELINE_STEPS.index(args.stop_after)
    step_index = PIPELINE_STEPS.index(step)
    return start_index <= step_index <= stop_index


def run_command(label: str, command: list[str], args: argparse.Namespace) -> None:
    print()
    print(f"==> {label}")
    print(format_command(command))
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
        "--preview",
        args.preprocess_preview,
        "--workers",
        args.preprocess_workers,
        "--chunksize",
        args.preprocess_chunksize,
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
            "--initial-alphabet",
            args.hybrid_initial_alphabet,
            "--digits",
            args.hybrid_digits,
            "--max-invalid-examples",
            args.hybrid_max_invalid_examples,
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
        "--character-coverage",
        args.sentencepiece_character_coverage,
        "--input-sentence-size",
        args.sentencepiece_input_sentence_size,
        "--max-sentence-length",
        args.sentencepiece_max_sentence_length,
        "--split-max-chars",
        args.sentencepiece_split_max_chars,
    )
    if not args.sentencepiece_split_long_lines:
        command.append("--no-split-long-lines")
    if not args.sentencepiece_shuffle_input_sentence:
        command.append("--no-shuffle-input-sentence")
    if args.sentencepiece_train_extremely_large_corpus:
        command.append("--train-extremely-large-corpus")
    if args.sentencepiece_hard_vocab_limit:
        command.append("--hard-vocab-limit")
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
        "--architecture",
        args.architecture,
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
        "--dropout",
        args.dropout,
        "--epochs",
        args.epochs,
        "--batch-size",
        args.batch_size,
        "--learning-rate",
        args.learning_rate,
        "--gradient-accumulation-steps",
        args.gradient_accumulation_steps,
        "--lr-schedule",
        args.lr_schedule,
        "--warmup-steps",
        args.warmup_steps,
        "--min-learning-rate",
        args.min_learning_rate,
        "--weight-decay",
        args.weight_decay,
        "--grad-clip",
        args.grad_clip,
        "--log-every",
        args.log_every,
        "--num-workers",
        args.num_workers,
        "--amp-dtype",
        args.amp_dtype,
        "--seed",
        args.seed,
    )
    if args.architecture == "qwen2":
        command += [
            "--num-key-value-heads",
            str(args.num_key_value_heads),
            "--intermediate-size",
            str(args.intermediate_size),
            "--rms-norm-eps",
            str(args.rms_norm_eps),
            "--rope-theta",
            str(args.rope_theta),
            "--attention-dropout",
            str(args.attention_dropout),
        ]
        if not args.tie_word_embeddings:
            command.append("--no-tie-word-embeddings")
        if args.attn_implementation is not None:
            command += ["--attn-implementation", args.attn_implementation]
    if args.device is not None:
        command += ["--device", args.device]
    if args.num_threads is not None:
        command += ["--num-threads", str(args.num_threads)]
    if args.metrics_log is not None:
        command += ["--metrics-log", path_arg(args.metrics_log)]
    if args.resume_checkpoint is not None:
        command += ["--resume", path_arg(args.resume_checkpoint)]
    if not args.amp:
        command.append("--no-amp")
    if not args.tf32:
        command.append("--no-tf32")
    if not args.fused_adamw:
        command.append("--no-fused-adamw")
    if args.compile:
        command.append("--compile")
    if args.save_optimizer:
        command.append("--save-optimizer")
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
    if not args.safe_serialization:
        command.append("--no-safe-serialization")
    run_command("Convert to Transformers folder", command, args)


def run_upload(args: argparse.Namespace) -> None:
    if args.skip_upload:
        print("Skipping upload")
        return
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
            "Base run name used for outputs. In matrix mode, architecture and "
            "tokenizer suffixes are added to model/export directories."
        ),
    )
    parser.add_argument(
        "--corpus-name",
        default=None,
        help=(
            "Name used for extracted and preprocessed corpus files. Defaults to "
            "--model-name, which lets matrix runs share one corpus."
        ),
    )
    parser.add_argument(
        "--run-name-template",
        default=None,
        help=(
            "Optional Python format string for per-run names. Available fields: "
            "model_name, architecture, tokenizer_kind."
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
    preprocess.add_argument("--preprocess-preview", type=int, default=3)
    preprocess.add_argument("--preprocess-workers", type=int, default=1)
    preprocess.add_argument("--preprocess-chunksize", type=int, default=32)

    tokenizer = parser.add_argument_group("tokenizer")
    tokenizer.add_argument(
        "--tokenizer-kind",
        choices=("hybrid", "bpe"),
        default="hybrid",
        help="Tokenizer family to train. Hybrid is the default; bpe uses SentencePiece.",
    )
    tokenizer.add_argument(
        "--tokenizer-kinds",
        choices=("hybrid", "bpe"),
        nargs="+",
        default=None,
        help=(
            "Run one or more tokenizer families. Overrides --tokenizer-kind and "
            "can be combined with --architectures for a matrix run."
        ),
    )
    tokenizer.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizers"))
    tokenizer.add_argument("--tokenizer-name", default=None)
    tokenizer.add_argument("--vocab-size", type=int, default=16000)
    tokenizer.add_argument(
        "--sentencepiece-model-type",
        choices=("bpe", "unigram", "char", "word"),
        default="bpe",
    )
    tokenizer.add_argument("--sentencepiece-character-coverage", type=float, default=1.0)
    tokenizer.add_argument("--sentencepiece-input-sentence-size", type=int, default=0)
    tokenizer.add_argument("--sentencepiece-max-sentence-length", type=int, default=100000)
    tokenizer.add_argument(
        "--sentencepiece-split-long-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    tokenizer.add_argument("--sentencepiece-split-max-chars", type=int, default=60000)
    tokenizer.add_argument(
        "--sentencepiece-shuffle-input-sentence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    tokenizer.add_argument(
        "--sentencepiece-train-extremely-large-corpus",
        action="store_true",
    )
    tokenizer.add_argument("--sentencepiece-hard-vocab-limit", action="store_true")
    tokenizer.add_argument("--hybrid-min-word-frequency", type=int, default=20)
    tokenizer.add_argument("--hybrid-atomic-only", action="store_true")
    tokenizer.add_argument("--hybrid-permissive", action="store_true")
    tokenizer.add_argument(
        "--hybrid-initial-alphabet",
        default=None,
        help="Initial alphabet passed to train_hybrid_tokenizer.py.",
    )
    tokenizer.add_argument(
        "--hybrid-digits",
        default=None,
        help="Digit alphabet passed to train_hybrid_tokenizer.py.",
    )
    tokenizer.add_argument("--hybrid-max-invalid-examples", type=int, default=20)

    dataset = parser.add_argument_group("dataset")
    dataset.add_argument("--train-dataset", type=Path, default=None)
    dataset.add_argument("--validation-dataset", type=Path, default=None)
    dataset.add_argument("--validation-fraction", type=float, default=0.05)
    dataset.add_argument("--block-size", type=int, default=512)
    dataset.add_argument("--stride", type=int, default=512)

    training = parser.add_argument_group("training")
    training.add_argument("--model-output-dir", type=Path, default=None)
    training.add_argument("--architecture", choices=("gpt2", "qwen2"), default="gpt2")
    training.add_argument(
        "--architectures",
        choices=("gpt2", "qwen2"),
        nargs="+",
        default=None,
        help=(
            "Run one or more model architectures. Overrides --architecture and "
            "can be combined with --tokenizer-kinds for a matrix run."
        ),
    )
    training.add_argument("--n-layer", type=int, default=8)
    training.add_argument("--n-head", type=int, default=8)
    training.add_argument("--n-embd", type=int, default=512)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--num-key-value-heads", type=int, default=4)
    training.add_argument("--intermediate-size", type=int, default=1376)
    training.add_argument("--rms-norm-eps", type=float, default=1.0e-6)
    training.add_argument("--rope-theta", type=float, default=10000.0)
    training.add_argument("--attention-dropout", type=float, default=0.0)
    training.add_argument("--tie-word-embeddings", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    training.add_argument("--epochs", type=int, default=5)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--gradient-accumulation-steps", type=int, default=1)
    training.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    training.add_argument("--warmup-steps", type=int, default=0)
    training.add_argument("--min-learning-rate", type=float, default=0.0)
    training.add_argument("--weight-decay", type=float, default=0.1)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--num-workers", type=int, default=0)
    training.add_argument("--num-threads", type=int, default=None)
    training.add_argument("--metrics-log", type=Path, default=None)
    training.add_argument("--device", choices=("cpu", "cuda"), default=None)
    training.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Forwarded to train_model.py --resume; distinct from pipeline --resume.",
    )
    training.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default="float16")
    training.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--fused-adamw", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--compile", action="store_true")
    training.add_argument("--save-optimizer", action=argparse.BooleanOptionalAction, default=False)
    training.add_argument("--seed", type=int, default=1337)

    convert = parser.add_argument_group("convert/upload")
    convert.add_argument("--hf-output-dir", type=Path, default=None)
    convert.add_argument(
        "--safe-serialization",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    convert.add_argument(
        "--hf-repo",
        default=None,
        help="Optional Hugging Face repo id to upload after conversion, e.g. user/model.",
    )

    skips = parser.add_argument_group("step control")
    skips.add_argument(
        "--start-at",
        choices=PIPELINE_STEPS,
        default="extract",
        help=(
            "Start at this pipeline step, assuming earlier step outputs already "
            "exist when they are needed."
        ),
    )
    skips.add_argument(
        "--stop-after",
        choices=PIPELINE_STEPS,
        default="upload",
        help="Stop after this pipeline step.",
    )
    skips.add_argument("--skip-extract", action="store_true")
    skips.add_argument("--skip-preprocess", action="store_true")
    skips.add_argument("--skip-tokenizer", action="store_true")
    skips.add_argument("--skip-dataset", action="store_true")
    skips.add_argument("--skip-train", action="store_true")
    skips.add_argument("--skip-convert", action="store_true")
    skips.add_argument("--skip-upload", action="store_true")

    return parser.parse_args()


def selected_values(args: argparse.Namespace, plural_name: str, singular_name: str) -> list[str]:
    """Return de-duplicated values from a plural override or singular default."""
    values = getattr(args, plural_name) or [getattr(args, singular_name)]
    return list(dict.fromkeys(values))


def make_run_name(
    args: argparse.Namespace,
    architecture: str,
    tokenizer_kind: str,
    matrix_mode: bool,
) -> str:
    """Derive the model/export run name for one pipeline leg."""
    if args.run_name_template is not None:
        return args.run_name_template.format(
            model_name=args.model_name,
            architecture=architecture,
            tokenizer_kind=tokenizer_kind,
        )
    if matrix_mode:
        return f"{args.model_name}_{architecture}_{tokenizer_kind}"
    return args.model_name


def validate_matrix_args(
    args: argparse.Namespace,
    architectures: list[str],
    tokenizer_kinds: list[str],
) -> None:
    """Fail early for explicit paths that would collide across matrix runs."""
    if PIPELINE_STEPS.index(args.stop_after) < PIPELINE_STEPS.index(args.start_at):
        raise ValueError("--stop-after must be the same as or later than --start-at")

    run_count = len(architectures) * len(tokenizer_kinds)
    if run_count <= 1:
        return

    per_run_paths = {
        "--model-output-dir": args.model_output_dir,
        "--hf-output-dir": args.hf_output_dir,
        "--metrics-log": args.metrics_log,
    }
    for option, value in per_run_paths.items():
        if value is not None:
            raise ValueError(f"{option} cannot be shared across a matrix run")

    if len(tokenizer_kinds) > 1:
        tokenizer_scoped_paths = {
            "--tokenizer-name": args.tokenizer_name,
            "--train-dataset": args.train_dataset,
            "--validation-dataset": args.validation_dataset,
        }
        for option, value in tokenizer_scoped_paths.items():
            if value is not None:
                raise ValueError(f"{option} cannot be shared across multiple tokenizer kinds")

    if args.resume_checkpoint is not None:
        raise ValueError("--resume-checkpoint is only supported for a single pipeline run")
    if args.hf_repo is not None:
        raise ValueError("--hf-repo is only supported for a single pipeline run")


def pipeline_runs(args: argparse.Namespace) -> list[argparse.Namespace]:
    """Create per-run argument namespaces for single or matrix execution."""
    architectures = selected_values(args, "architectures", "architecture")
    tokenizer_kinds = selected_values(args, "tokenizer_kinds", "tokenizer_kind")
    validate_matrix_args(args, architectures, tokenizer_kinds)

    matrix_mode = len(architectures) * len(tokenizer_kinds) > 1
    corpus_name = args.corpus_name or args.model_name
    runs: list[argparse.Namespace] = []
    for tokenizer_kind in tokenizer_kinds:
        for architecture in architectures:
            run_args = copy.copy(args)
            run_args.architecture = architecture
            run_args.tokenizer_kind = tokenizer_kind
            run_name = make_run_name(run_args, architecture, tokenizer_kind, matrix_mode)
            default_paths(corpus_name, args.model_name, run_name, matrix_mode, run_args)
            if run_args.hybrid_initial_alphabet is None:
                run_args.hybrid_initial_alphabet = (
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                )
            if run_args.hybrid_digits is None:
                run_args.hybrid_digits = "0123456789"
            runs.append(run_args)
    return runs


def main() -> None:
    try:
        runs = pipeline_runs(parse_args())
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None

    completed_outputs: set[tuple[str, str]] = set()
    pipeline = [
        ("extract", run_extract),
        ("preprocess", run_preprocess),
        ("tokenizer", run_tokenizer),
        ("dataset", run_dataset),
        ("train", run_training),
        ("convert", run_convert),
        ("upload", run_upload),
    ]
    for index, args in enumerate(runs, start=1):
        args.completed_outputs = completed_outputs
        if len(runs) > 1:
            print()
            print(
                f"### Pipeline {index}/{len(runs)}: "
                f"architecture={args.architecture} tokenizer={args.tokenizer_kind}"
            )

        for step, runner in pipeline:
            if not step_selected(step, args):
                continue
            runner(args)
            if step in {"extract", "preprocess", "tokenizer", "dataset"}:
                if not getattr(args, f"skip_{step}"):
                    completed_outputs.add(
                        (step, str(existing_output_for_step(step, args)))
                    )
    print()
    print("Pipeline finished.")


if __name__ == "__main__":
    main()
