"""Preprocess BabyLM Mandarin JSONL into word-preserving pinyin initials."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import jieba
from pypinyin import Style, pinyin


LABELS = {
    "题干": "<QUESTION>",
    "选项": "<OPTIONS>",
    "答案": "<ANSWER>",
    "解析": "<EXPLANATION>",
}

PUNCTUATION = set("。，、？！：；.,?!:;()[]“”\"'")
CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")

# Match protected markers before ordinary words so tokens like <ANSWER> survive
# the later English/punctuation handling as a single vocabulary item.
TOKEN_RE = re.compile(
    r"<[A-Z_]+>|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|"
    r"[A-Za-z]+|"
    r"[。，、？！：；.,?!:;()\[\]“”\"']|"
    r"\S"
)


def require_dependencies() -> None:
    """Fail early with a concise install hint and quiet jieba startup logging."""
    if jieba is None or Style is None or pinyin is None:
        raise SystemExit(
            "Missing dependency: install with `py -m pip install jieba pypinyin`."
        )
    jieba.setLogLevel(logging.WARNING)


def normalize_text(text: str) -> str:
    """Replace task-specific surface forms with stable special tokens.

    This happens before tokenization so multi-character patterns such as
    ``$$...$$`` and empty brackets cannot be split into punctuation pieces.
    """
    text = re.sub(r"\$\$.*?\$\$", " <MATH> ", text, flags=re.DOTALL)
    text = re.sub(r"[（(]\s*[）)]", " <BLANK> ", text)

    for label, marker in LABELS.items():
        text = re.sub(rf"{label}\s*[:：]", f" {marker} ", text)

    text = re.sub(r"(?<![A-Za-z])yes(?![A-Za-z])", " <YES> ", text, flags=re.I)
    text = re.sub(r"(?<![A-Za-z])no(?![A-Za-z])", " <NO> ", text, flags=re.I)
    text = re.sub(r"(?<![A-Za-z])[ABCD](?=\s*[:：.．、\)])", r" \g<0> ", text)
    text = re.sub(r"(?<![A-Za-z0-9])[-+]?\d+(?:[.,]\d+)*(?:%|％)?", " <NUM> ", text)

    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chinese_word_to_initials(word: str) -> str:
    """Convert one already-segmented Chinese word to one initials token.

    The important representation choice is here: ``今天`` becomes ``jt`` rather
    than ``j t``, preserving jieba's word boundary for later BPE training.
    """
    syllables = pinyin(word, style=Style.NORMAL, heteronym=False, errors="ignore")
    initials = [item[0][0].lower() for item in syllables if item and item[0]]
    return "".join(initials)


def tokenize_chinese_span(text: str) -> Iterable[str]:
    """Segment a contiguous Chinese span and emit one abbreviation per jieba word."""
    for word in jieba.cut(text, cut_all=False):
        word = word.strip()
        if not word:
            continue
        if CHINESE_RE.search(word):
            token = chinese_word_to_initials(word)
            if token:
                yield token


def process_text(text: str) -> str:
    """Convert one raw document string into the final space-separated token line."""
    tokens: list[str] = []
    for part in TOKEN_RE.findall(normalize_text(text)):
        if part.startswith("<") and part.endswith(">"):
            tokens.append(part)
        elif CHINESE_RE.fullmatch(part):
            tokens.extend(tokenize_chinese_span(part))
        elif part in PUNCTUATION:
            tokens.append(part)
        elif re.fullmatch(r"[A-Za-z]+", part):
            upper = part.upper()
            tokens.append(upper if upper in {"A", "B", "C", "D"} else part.lower())

    return " ".join(tokens)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield JSON objects from UTF-8 JSONL, tolerating a leading BOM if present."""
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc


def preprocess_file(input_path: Path, output_path: Path, preview_count: int) -> int:
    """Stream input documents to output while retaining a small preview buffer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    previews: list[tuple[str, str]] = []

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for obj in read_jsonl(input_path):
            text = str(obj.get("text", ""))
            processed = process_text(text)
            out.write(processed)
            out.write("\n")
            written += 1

            if len(previews) < preview_count:
                previews.append((text, processed))

    for original, processed in previews:
        print("ORIGINAL:")
        print(original)
        print("PROCESSED:")
        print(processed)
        print()

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert BabyLM Mandarin JSONL text fields to pinyin-initial tokens."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    require_dependencies()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    count = preprocess_file(args.input, args.output, args.preview)
    print(f"Wrote {count:,} processed documents to {args.output}")


if __name__ == "__main__":
    main()
