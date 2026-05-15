"""Count unique Mandarin character-to-pinyin tone cases in BabyLM JSONL."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import jieba
from pypinyin import Style, pinyin


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
TONE_RE = re.compile(r"[1-4]$")
TONE_ORDER = ("1", "2", "3", "4", "blank")


def require_dependencies() -> None:
    """Fail early with a concise install hint and quiet jieba startup logging."""
    if jieba is None or Style is None or pinyin is None:
        raise SystemExit(
            "Missing dependency: install with `py -m pip install jieba pypinyin`."
        )
    jieba.setLogLevel(logging.WARNING)


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


def chinese_spans(text: str) -> Iterable[str]:
    """Yield contiguous Chinese spans from mixed text."""
    yield from CHINESE_RE.findall(text)


def pinyin_tone_case(character: str, pinyin_with_tone: str) -> tuple[str, str, str]:
    """Return the unique mapping key plus the tone bucket for one character."""
    tone_match = TONE_RE.search(pinyin_with_tone)
    tone = tone_match.group(0) if tone_match else "blank"
    return character, pinyin_with_tone.lower(), tone


def iter_contextual_cases(text: str) -> Iterable[tuple[str, str, str]]:
    """Segment Chinese text with jieba and yield contextual pinyin cases.

    pypinyin receives whole jieba words instead of isolated characters, which lets
    it choose readings for common contextual forms such as polyphonic characters.
    """
    for span in chinese_spans(text):
        for word in jieba.cut(span, cut_all=False):
            word = word.strip()
            if not word:
                continue

            pronunciations = pinyin(
                word,
                style=Style.TONE3,
                heteronym=False,
                neutral_tone_with_five=False,
                errors="ignore",
            )
            chinese_chars = [char for char in word if CHINESE_RE.fullmatch(char)]

            for character, syllable in zip(chinese_chars, pronunciations):
                if not syllable:
                    continue
                yield pinyin_tone_case(character, syllable[0])


def collect_unique_cases(input_path: Path, text_field: str) -> set[tuple[str, str, str]]:
    """Collect unique character/pinyin/tone cases from a JSONL file."""
    unique_cases: set[tuple[str, str, str]] = set()
    for obj in read_jsonl(input_path):
        text = str(obj.get(text_field, ""))
        unique_cases.update(iter_contextual_cases(text))
    return unique_cases


def summarize(unique_cases: set[tuple[str, str, str]]) -> dict[str, Any]:
    """Build counts and percentages for unique cases grouped by tone."""
    counts = Counter(tone for _, _, tone in unique_cases)
    total = sum(counts.values())

    tones = {}
    for tone in TONE_ORDER:
        count = counts[tone]
        percentage = (count / total * 100) if total else 0.0
        tones[tone] = {
            "count": count,
            "percentage": round(percentage, 4),
        }

    return {
        "total_unique_cases": total,
        "tones": tones,
    }


def write_case_list(
    output_path: Path,
    unique_cases: set[tuple[str, str, str]],
    summary: dict[str, Any],
) -> None:
    """Write summary plus all unique mappings for later inspection."""
    payload = {
        **summary,
        "unique_cases": [
            {"character": char, "pinyin": syllable, "tone": tone}
            for char, syllable, tone in sorted(unique_cases)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment JSONL text with jieba, convert to contextual pinyin, and "
            "count unique Chinese character/pinyin cases by tone."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--show-cases",
        action="store_true",
        help="Print every unique character/pinyin/tone case after the summary.",
    )
    return parser.parse_args()


def main() -> None:
    require_dependencies()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()
    unique_cases = collect_unique_cases(args.input, args.text_field)
    summary = summarize(unique_cases)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.show_cases:
        for char, syllable, tone in sorted(unique_cases):
            print(f"{char}\t{syllable}\t{tone}")

    if args.output:
        write_case_list(args.output, unique_cases, summary)
        print(f"Wrote tone statistics to {args.output}")


if __name__ == "__main__":
    main()

"""
{
  "total_unique_cases": 8231,
  "tones": {
    "1": {
      "count": 2092,
      "percentage": 25.4161
    },
    "2": {
      "count": 2095,
      "percentage": 25.4526
    },
    "3": {
      "count": 1350,
      "percentage": 16.4014
    },
    "4": {
      "count": 2629,
      "percentage": 31.9402
    },
    "blank": {
      "count": 65,
      "percentage": 0.7897
    }
  }
}
"""