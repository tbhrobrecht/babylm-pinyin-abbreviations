"""Count token IDs in JSONL dataset files.

By default this scans every ``*.jsonl`` file in ``data/datasets`` and counts
the number of items in each row's ``input_ids`` list.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def iter_jsonl_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    yield from sorted(path.glob("*.jsonl"))


def count_tokens(path: Path, field: str) -> tuple[int, int, int, int]:
    rows = 0
    total_tokens = 0
    min_tokens: int | None = None
    max_tokens = 0

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_number}: {exc}") from exc

            tokens = row.get(field)
            if not isinstance(tokens, list):
                raise ValueError(
                    f"{path}: line {line_number} does not contain a list field named {field!r}"
                )

            token_count = len(tokens)
            rows += 1
            total_tokens += token_count
            min_tokens = token_count if min_tokens is None else min(min_tokens, token_count)
            max_tokens = max(max_tokens, token_count)

    return rows, total_tokens, min_tokens or 0, max_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count token IDs in JSONL files, defaulting to data/datasets/*.jsonl."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("data/datasets"),
        help="JSONL file or directory containing JSONL files.",
    )
    parser.add_argument(
        "--field",
        default="input_ids",
        help="JSON field containing the token ID list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list(iter_jsonl_files(args.path))
    if not files:
        raise SystemExit(f"No JSONL files found at {args.path}")

    grand_rows = 0
    grand_tokens = 0

    for path in files:
        rows, tokens, min_tokens, max_tokens = count_tokens(path, args.field)
        grand_rows += rows
        grand_tokens += tokens
        average = tokens / rows if rows else 0
        print(path)
        print(f"  rows: {rows:,}")
        print(f"  tokens: {tokens:,}")
        print(f"  min/avg/max tokens per row: {min_tokens:,} / {average:,.2f} / {max_tokens:,}")

    if len(files) > 1:
        print("TOTAL")
        print(f"  rows: {grand_rows:,}")
        print(f"  tokens: {grand_tokens:,}")


if __name__ == "__main__":
    main()
