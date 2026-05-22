"""Count BabyLM corpus tokens before SentencePiece/BPE.

For this project, one corpus token is one whitespace-separated pinyin-code token
in the processed text, where each code token corresponds to one Jieba-segmented
Chinese word or preserved special/punctuation token.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


def iter_text_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    yield from sorted(path.glob("*.txt"))


def count_whitespace_tokens(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            total += len(line.split())
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count pre-SentencePiece BabyLM tokens from processed pinyin-code text."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/processed"),
        help="Processed .txt file or directory of processed .txt files.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        help="Optionally report training exposure as tokens multiplied by epochs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list(iter_text_files(args.path))
    if not files:
        raise SystemExit(f"No processed .txt files found at {args.path}")

    total_tokens = sum(count_whitespace_tokens(path) for path in files)
    print(f"Corpus tokens: {total_tokens:,}")

    if args.epochs is not None:
        exposure = total_tokens * args.epochs
        print(f"Tokens x epochs ({args.epochs:g}): {exposure:,.0f}")


if __name__ == "__main__":
    main()
