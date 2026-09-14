"""Split long processed-text lines for SentencePiece training.

SentencePiece BPE can fail on very long lines when whitespace splitting is
disabled. This utility writes a tokenizer-training copy of a processed corpus
where long documents are split at whitespace boundaries while preserving all
tokens. Because encoded Mandarin words carry their own boundaries and are
written without whitespace, a single whitespace-delimited item can be longer
than the limit; such runs are split at encoded-word boundaries instead.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from preprocessing.encoding import is_encoded_run, split_encoded_words
except ImportError:  # Executed directly from the preprocessing directory.
    from encoding import is_encoded_run, split_encoded_words


@dataclass
class SplitStats:
    input_lines: int = 0
    output_lines: int = 0
    split_lines: int = 0
    max_input_chars: int = 0
    max_output_chars: int = 0


def split_encoded_run(run: str, max_chars: int) -> list[str]:
    """Split one long encoded run into pieces of at most max_chars characters.

    Splitting happens only between complete encoded words, so no word and no
    syllable atom is ever cut in half.
    """
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in split_encoded_words(run):
        word_len = len(word)
        if word_len > max_chars:
            raise ValueError(
                f"Encoded word longer than --max-chars ({word_len} > {max_chars}): "
                f"{word[:80]!r}"
            )
        if current and current_len + word_len > max_chars:
            pieces.append("".join(current))
            current = [word]
            current_len = word_len
        else:
            current.append(word)
            current_len += word_len

    if current:
        pieces.append("".join(current))

    return pieces


def split_line_at_whitespace(line: str, max_chars: int) -> list[str]:
    """Return chunks no longer than max_chars, splitting only between tokens."""
    line = line.strip()
    if not line:
        return []
    if len(line) <= max_chars:
        return [line]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    for token in line.split():
        token_len = len(token)
        if token_len > max_chars:
            if not is_encoded_run(token):
                raise ValueError(
                    f"Token longer than --max-chars ({token_len} > {max_chars}): "
                    f"{token[:80]!r}"
                )
            # Each piece is already near the limit, so emit them as their own
            # lines rather than re-joining them with spaces a tokenizer would
            # then treat as word boundaries.
            flush()
            chunks.extend(split_encoded_run(token, max_chars))
            continue

        next_len = token_len if not current else current_len + 1 + token_len
        if current and next_len > max_chars:
            flush()
        current.append(token)
        current_len = token_len if len(current) == 1 else next_len

    flush()

    return chunks


def iter_split_lines(input_path: Path, max_chars: int, stats: SplitStats) -> Iterable[str]:
    """Yield split lines and update stats."""
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n\r")
            stats.input_lines += 1
            stats.max_input_chars = max(stats.max_input_chars, len(line))

            chunks = split_line_at_whitespace(line, max_chars)
            if len(chunks) > 1:
                stats.split_lines += 1

            for chunk in chunks:
                stats.output_lines += 1
                stats.max_output_chars = max(stats.max_output_chars, len(chunk))
                yield chunk


def split_file(input_path: Path, output_path: Path, max_chars: int, dry_run: bool) -> SplitStats:
    """Split input_path into output_path and return summary statistics."""
    if max_chars <= 0:
        raise ValueError("--max-chars must be greater than zero")

    stats = SplitStats()
    split_lines = iter_split_lines(input_path, max_chars, stats)

    if dry_run:
        for _ in split_lines:
            pass
        return stats

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for line in split_lines:
            out.write(line)
            out.write("\n")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a SentencePiece-training copy of processed text by splitting "
            "long lines at whitespace boundaries."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Processed .txt file with one document per line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .txt file to use for SentencePiece training.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=60000,
        help="Maximum characters per output line. Keep below 65535 for SentencePiece BPE.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be written without creating the output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = split_file(args.input, args.output, args.max_chars, args.dry_run)

    action = "Would write" if args.dry_run else "Wrote"
    print(f"{action} {stats.output_lines:,} lines from {stats.input_lines:,} input lines")
    print(f"Split input lines: {stats.split_lines:,}")
    print(f"Max input chars: {stats.max_input_chars:,}")
    print(f"Max output chars: {stats.max_output_chars:,}")
    if not args.dry_run:
        print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
