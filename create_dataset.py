"""Create a chunked language-modeling dataset from SentencePiece-tokenized text."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path


def require_sentencepiece():
    """Import SentencePiece or stop with an install hint."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `py -m pip install sentencepiece`."
        ) from exc

    return spm


def iter_token_ids(input_paths: Iterable[Path], processor) -> Iterable[int]:
    """Yield token ids from processed text files, adding EOS after each document."""
    eos_id = processor.eos_id()

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                yield from processor.encode(text, out_type=int)
                if eos_id >= 0:
                    yield eos_id


def iter_chunks(token_ids: Iterable[int], block_size: int, stride: int) -> Iterable[list[int]]:
    """Yield fixed-size token chunks using a sliding window."""
    if block_size <= 0:
        raise ValueError("--block-size must be greater than zero")
    if stride <= 0:
        raise ValueError("--stride must be greater than zero")

    buffer: list[int] = []
    for token_id in token_ids:
        buffer.append(token_id)
        if len(buffer) == block_size:
            yield list(buffer)
            del buffer[:stride]


def write_dataset(args: argparse.Namespace) -> int:
    """Tokenize processed text and write one JSON record per training chunk."""
    spm = require_sentencepiece()
    processor = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    chunks = iter_chunks(
        iter_token_ids(args.input, processor),
        block_size=args.block_size,
        stride=args.stride,
    )

    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for chunk in chunks:
            record = {"input_ids": chunk}
            if args.include_labels:
                record["labels"] = chunk
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")
            written += 1

    return written


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset creation."""
    parser = argparse.ArgumentParser(
        description="Build a JSONL language-modeling dataset from processed pinyin-code text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("data/processed/10k_babylm_zho.txt")],
        help="One or more UTF-8 processed text files, one document per line.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizers/babylm_zho_pinyin_spm.model"),
        help="SentencePiece .model file used to encode the dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/10k_babylm_zho_spm.jsonl"),
        help="Output JSONL dataset path.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Number of token ids per training example.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Sliding-window stride. Use a value smaller than block size for overlap.",
    )
    parser.add_argument(
        "--include-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write labels identical to input_ids for causal language modeling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = write_dataset(args)
    print(f"Wrote {count:,} examples to {args.output}")


if __name__ == "__main__":
    main()
