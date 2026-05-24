"""Train a SentencePiece tokenizer on preprocessed BabyLM pinyin-code text."""

from __future__ import annotations

import argparse
from pathlib import Path


SPECIAL_TOKENS = [
    "<QUESTION>",
    "<OPTIONS>",
    "<ANSWER>",
    "<EXPLANATION>",
    "<MATH>",
    "<BLANK>",
    "<YES>",
    "<NO>",
    "<NUM>",
]


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

    # These settings match the output of preprocess.py: tokens are already
    # whitespace-separated, and pinyin-code digits should stay attached to their
    # letters instead of becoming separate pieces.
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
        help="Maximum line length accepted by SentencePiece before skipping a line.",
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
