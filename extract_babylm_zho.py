"""Extract BabyLM Chinese data from Hugging Face into JSONL.

The dataset is gated. Accept the terms at:
https://huggingface.co/datasets/BabyLM-community/babylm-zho

Then log in with `huggingface-cli login` or set `HF_TOKEN`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable


DATASET_ID = "BabyLM-community/babylm-zho"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_rows(split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `py -m pip install datasets`."
        ) from exc

    kwargs: dict[str, Any] = {
        "path": DATASET_ID,
        "split": split,
        "streaming": streaming,
    }
    token = hf_token()
    if token:
        kwargs["token"] = token

    return load_dataset(**kwargs)


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.category and row.get("category") not in args.category:
        return False
    if args.script and row.get("script") not in args.script:
        return False
    if args.language and row.get("language") not in args.language:
        return False
    return True


def clean_row(row: dict[str, Any], text_only: bool) -> dict[str, Any]:
    if text_only:
        return {
            "doc_id": row.get("doc-id"),
            "text": row.get("text", ""),
        }
    return dict(row)


def extract(args: argparse.Namespace) -> int:
    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for row in load_rows(args.split, args.streaming):
            if args.max_docs is not None and written >= args.max_docs:
                break
            if not keep_row(row, args):
                continue

            out.write(json.dumps(clean_row(row, args.text_only), ensure_ascii=False))
            out.write("\n")
            written += 1

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Extract {DATASET_ID} from Hugging Face into JSONL."
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=Path, default=Path("data/babylm_zho.jsonl"))
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--category", action="append")
    parser.add_argument("--script", action="append")
    parser.add_argument("--language", action="append")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stream by default so the full dataset is not downloaded first.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract(args)
    print(f"Wrote {count:,} records to {args.output}")


if __name__ == "__main__":
    main()
