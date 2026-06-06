"""Train a SentencePiece tokenizer on preprocessed BabyLM pinyin-code text."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from preprocessing.split_long_sentencepiece_lines import SplitStats, split_file


SPECIAL_TOKENS = [
    "<QUESTION>",
    "<OPTIONS>",
    "<ANSWER>",
    "<EXPLANATION>",
    "<MATH>",
    "<URL>",
    "<BLANK>",
    "<YES>",
    "<NO>",
    "<NUM>",
]

BPE_SAFE_MAX_LINE_CHARS = 60_000


def require_sentencepiece():
    """Import SentencePiece or stop with an install hint.

    Keeping the import inside this function lets users run ``--help`` or read
    this file without needing SentencePiece installed first.
    """
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `py -m pip install sentencepiece`."
        ) from exc

    return spm


def should_split_long_lines(args: argparse.Namespace) -> bool:
    """Return true when BPE training needs SentencePiece-safe line lengths."""
    return (
        getattr(args, "model_type", None) == "bpe"
        and getattr(args, "split_long_lines", False)
    )


def split_training_inputs(
    input_paths: list[Path],
    output_dir: Path,
    model_name: str,
    max_chars: int,
) -> tuple[tempfile.TemporaryDirectory[str], list[str], list[tuple[Path, SplitStats]]]:
    """Create temporary SentencePiece input files with bounded line lengths."""
    if max_chars <= 0:
        raise ValueError("--split-max-chars must be greater than zero")
    if max_chars > 65_535:
        raise ValueError("--split-max-chars must be 65,535 or lower for SentencePiece BPE")

    temp_dir = tempfile.TemporaryDirectory(prefix=f"{model_name}_spm_input_", dir=output_dir)
    split_paths: list[str] = []
    stats_by_input: list[tuple[Path, SplitStats]] = []

    for index, input_path in enumerate(input_paths):
        split_path = Path(temp_dir.name) / f"{index}_{input_path.name}"
        stats = split_file(input_path, split_path, max_chars=max_chars, dry_run=False)
        split_paths.append(str(split_path))
        stats_by_input.append((input_path, stats))

    return temp_dir, split_paths, stats_by_input


def train_tokenizer(args: argparse.Namespace) -> None:
    """Train a SentencePiece model from one or more preprocessed text files.

    The input files should contain one already-preprocessed document per line.
    The generated ``.model`` and ``.vocab`` files are written to
    ``args.output_dir`` using ``args.model_name`` as the filename prefix.
    """
    spm = require_sentencepiece()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = args.output_dir / args.model_name
    input_files = [str(path) for path in args.input]
    temp_input_dir: tempfile.TemporaryDirectory[str] | None = None

    if should_split_long_lines(args):
        split_max_chars = min(args.split_max_chars, args.max_sentence_length)
        temp_input_dir, input_files, stats_by_input = split_training_inputs(
            args.input,
            args.output_dir,
            args.model_name,
            split_max_chars,
        )
        for input_path, stats in stats_by_input:
            if stats.split_lines:
                print(
                    "Split long SentencePiece training lines for "
                    f"{input_path}: {stats.split_lines:,} lines split; "
                    f"max line {stats.max_input_chars:,} -> {stats.max_output_chars:,} chars"
                )

    # The corpus starts as whitespace-separated atomic tokens, but BPE is allowed
    # to learn pieces that span adjacent atoms. Pinyin-code digits should still
    # stay attached to their letters instead of becoming separate pieces.
    try:
        spm.SentencePieceTrainer.train(
            input=input_files,
            model_prefix=str(model_prefix),
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            character_coverage=args.character_coverage,
            input_sentence_size=args.input_sentence_size,
            max_sentence_length=args.max_sentence_length,
            shuffle_input_sentence=args.shuffle_input_sentence,
            train_extremely_large_corpus=args.train_extremely_large_corpus,
            hard_vocab_limit=args.hard_vocab_limit,
            # Register preprocessing markers as indivisible pieces so tokens such as
            # <NUM> and <MATH> survive tokenizer training exactly as written.
            user_defined_symbols=",".join(SPECIAL_TOKENS),
            # Fixed ids make the tokenizer easier to use consistently in training
            # code and model configs.
            pad_id=0,
            unk_id=1,
            bos_id=2,
            eos_id=3,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            normalization_rule_name="identity",
            remove_extra_whitespaces=False,
            split_by_whitespace=False,
            split_by_number=False,
            allow_whitespace_only_pieces=False,
        )
    finally:
        if temp_input_dir is not None:
            temp_input_dir.cleanup()

    print(f"Wrote tokenizer model to {model_prefix.with_suffix('.model')}")
    print(f"Wrote tokenizer vocab to {model_prefix.with_suffix('.vocab')}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for tokenizer training.

    Defaults are chosen for the repository's 10k processed BabyLM sample, but
    all important SentencePiece knobs can be overridden from the command line.
    """
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer from preprocessed pinyin-code text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("data/processed/10k_babylm_zho.txt")],
        help="One or more UTF-8 text files, one preprocessed document per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokenizers"),
        help="Directory where the .model and .vocab files will be written.",
    )
    parser.add_argument(
        "--model-name",
        default="babylm_zho_pinyin_spm",
        help="Base filename for the generated tokenizer artifacts.",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument(
        "--model-type",
        choices=["unigram", "bpe", "char", "word"],
        default="bpe",
    )
    parser.add_argument("--character-coverage", type=float, default=1.0)
    parser.add_argument(
        "--input-sentence-size",
        type=int,
        default=0,
        help="Maximum training lines sampled by SentencePiece; 0 uses all lines.",
    )
    parser.add_argument(
        "--max-sentence-length",
        type=int,
        default=100000,
        help=(
            "Maximum line length accepted by SentencePiece before skipping a line. "
            "For BPE, long lines are split before training by default."
        ),
    )
    parser.add_argument(
        "--split-long-lines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For BPE training, split long input lines at whitespace boundaries before "
            "calling SentencePiece. This avoids SentencePiece's 16-bit per-line "
            "position limit while preserving tokens. Disable with --no-split-long-lines."
        ),
    )
    parser.add_argument(
        "--split-max-chars",
        type=int,
        default=BPE_SAFE_MAX_LINE_CHARS,
        help=(
            "Maximum characters per temporary BPE training line when "
            "--split-long-lines is enabled. Must stay at or below 65,535."
        ),
    )
    parser.add_argument(
        "--shuffle-input-sentence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train-extremely-large-corpus",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hard-vocab-limit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require the exact requested vocab size instead of allowing a smaller valid vocab.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    train_tokenizer(args)


if __name__ == "__main__":
    main()
