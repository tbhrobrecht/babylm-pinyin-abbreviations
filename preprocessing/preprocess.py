"""Preprocess BabyLM Mandarin JSONL into word-preserving pinyin initial codes."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Literal

import jieba
from pypinyin import Style, pinyin


LABELS = {
    "题干": "<QUESTION>",
    "选项": "<OPTIONS>",
    "答案": "<ANSWER>",
    "解析": "<EXPLANATION>",
}

PUNCTUATION = set("。，、？！：；.,?!:;()[]{}<>《》【】“”\"'‘’「」『』—-~…/\\")
CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
Transliteration = Literal["pinyin-code", "pinyin-initial", "hanzi"]
LATIN_LETTER = r"A-Za-zÀ-ÖØ-öø-ÿĀ-ſƀ-ɏɐ-ʯ"
LATIN_ALNUM_PATTERN = (
    rf"(?:[{LATIN_LETTER}][{LATIN_LETTER}0-9]*"
    rf"(?:[-_][{LATIN_LETTER}0-9]+)*|"
    rf"[0-9]+[{LATIN_LETTER}][{LATIN_LETTER}0-9]*"
    rf"(?:[-_][{LATIN_LETTER}0-9]+)*)"
)
LATIN_ALNUM_RE = re.compile(LATIN_ALNUM_PATTERN)
URL_RE = re.compile(r"\b(?:https?://\S*|www\.\S+)", flags=re.I)
DISCARDED_UNICODE_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn"}

# Match protected markers before ordinary words so tokens like <ANSWER> survive
# the later English/punctuation handling as a single vocabulary item.
TOKEN_RE = re.compile(
    r"<[A-Z_]+>|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|"
    rf"{LATIN_ALNUM_PATTERN}|"
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
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" <URL> ", text)
    text = re.sub(r"\$\$.*?\$\$", " <MATH> ", text, flags=re.DOTALL)
    text = re.sub(r"[（(]\s*[）)]", " <BLANK> ", text)

    for label, marker in LABELS.items():
        text = re.sub(rf"{label}\s*[:：]", f" {marker} ", text)

    text = re.sub(
        rf"(?<![{LATIN_LETTER}])yes(?![{LATIN_LETTER}])",
        " <YES> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"(?<![{LATIN_LETTER}])no(?![{LATIN_LETTER}])",
        " <NO> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        rf"(?<![{LATIN_LETTER}])[ABCD](?=\s*[:：.．、\)])",
        r" \g<0> ",
        text,
    )
    text = re.sub(
        rf"(?<![{LATIN_LETTER}0-9])[-+]?\d+(?:[.,]\d+)*(?:%|％)?"
        rf"(?![{LATIN_LETTER}0-9])",
        " <NUM> ",
        text,
    )

    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def latin_token_to_model_token(token: str) -> str:
    """Normalize non-Mandarin alphanumeric tokens without losing option labels."""
    upper = token.upper()
    return upper if upper in {"A", "B", "C", "D"} else token.lower()


def should_preserve_fallback_token(token: str) -> bool:
    """Return true for visible non-Hanzi letters, punctuation, and symbols."""
    if token == "\ufffd":
        return False
    for char in token:
        category = unicodedata.category(char)
        if category in DISCARDED_UNICODE_CATEGORIES:
            return False
        if category[0] not in {"L", "P", "S"}:
            return False
    return True


def split_tone3_syllable(syllable: str) -> tuple[str, int]:
    """Return the plain pinyin syllable and its tone number.

    ``Style.TONE3`` writes tones as final digits, but neutral tone syllables have
    no digit. Treat those digitless cases as fifth tone.
    """
    match = re.fullmatch(r"([a-züv]+)([1-5]?)", syllable.lower())
    if not match:
        return syllable, 5

    plain, tone = match.groups()
    return plain, int(tone or "5")


def length_digit_offset(syllable: str) -> int:
    """Map pinyin syllable length to the requested 0-4 digit offset."""
    return min(max(len(syllable), 1), 5) - 1


def syllable_to_initial_code(syllable: str) -> str:
    """Convert one pinyin syllable with tone into ``initial + digit``.

    Tone controls initial casing and whether the digit starts from 0 or 5:
    tones 1/3/5 use uppercase initials, while tones 2/4 use lowercase initials.
    Syllable length then adds the 0-4 offset that makes the final digit.
    """
    plain, tone = split_tone3_syllable(syllable)
    if not plain:
        return ""

    tone_offset = 5 if tone in {3, 4, 5} else 0
    digit = tone_offset + length_digit_offset(plain)
    initial = plain[0].upper() if tone in {1, 3, 5} else plain[0].lower()
    return f"{initial}{digit}"


def syllable_to_initial_letter(syllable: str) -> str:
    """Convert one pinyin syllable with tone into its lowercase first letter."""
    plain, _ = split_tone3_syllable(syllable)
    return plain[:1].lower()


def chinese_word_to_initial_codes(word: str) -> str:
    """Convert one already-segmented Chinese word to one compact code token.

    The important representation choice is preserved: jieba decides the word
    boundary, and all syllable codes inside that word are concatenated. For
    example, ``我们`` becomes ``W6M7`` rather than ``W6 M7``.
    """
    syllables = pinyin(word, style=Style.TONE3, heteronym=False, errors="ignore")
    codes = [syllable_to_initial_code(item[0]) for item in syllables if item and item[0]]
    return "".join(code for code in codes if code)


def chinese_word_to_initial_letters(word: str) -> str:
    """Convert one already-segmented Chinese word to lowercase pinyin initials."""
    syllables = pinyin(word, style=Style.TONE3, heteronym=False, errors="ignore")
    initials = [
        syllable_to_initial_letter(item[0]) for item in syllables if item and item[0]
    ]
    return "".join(initial for initial in initials if initial)


def chinese_word_to_transliteration(word: str, transliteration: Transliteration) -> str:
    """Convert one segmented Chinese word using the requested transliteration."""
    if transliteration == "pinyin-code":
        return chinese_word_to_initial_codes(word)
    if transliteration == "pinyin-initial":
        return chinese_word_to_initial_letters(word)
    if transliteration == "hanzi":
        return word
    raise ValueError(f"Unsupported transliteration: {transliteration}")


def tokenize_chinese_span(
    text: str,
    transliteration: Transliteration = "pinyin-code",
    use_jieba: bool = True,
) -> Iterable[str]:
    """Emit one token per jieba word or per Hanzi character."""
    words = jieba.cut(text, cut_all=False) if use_jieba else text
    for word in words:
        word = word.strip()
        if not word:
            continue
        if CHINESE_RE.search(word):
            token = chinese_word_to_transliteration(word, transliteration)
            if token:
                yield token


def process_text(
    text: str,
    transliteration: Transliteration = "pinyin-code",
    use_jieba: bool = True,
) -> str:
    """Convert one raw document string into the final space-separated token line."""
    tokens: list[str] = []
    for part in TOKEN_RE.findall(normalize_text(text)):
        if part.startswith("<") and part.endswith(">"):
            tokens.append(part)
        elif CHINESE_RE.fullmatch(part):
            tokens.extend(tokenize_chinese_span(part, transliteration, use_jieba))
        elif part in PUNCTUATION:
            tokens.append(part)
        elif LATIN_ALNUM_RE.fullmatch(part):
            tokens.append(latin_token_to_model_token(part))
        elif part.isdigit():
            tokens.append("<NUM>")
        elif should_preserve_fallback_token(part):
            tokens.append(part.lower())

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


def preprocess_file(
    input_path: Path,
    output_path: Path,
    preview_count: int,
    transliteration: Transliteration = "pinyin-code",
    use_jieba: bool = True,
) -> int:
    """Stream input documents to output while retaining a small preview buffer."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    previews: list[tuple[str, str]] = []

    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for obj in read_jsonl(input_path):
            text = str(obj.get("text", ""))
            processed = process_text(text, transliteration, use_jieba)
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
        description="Convert BabyLM Mandarin JSONL text fields to model-ready tokens."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=int, default=3)
    parser.add_argument(
        "--transliteration",
        choices=("pinyin-code", "pinyin-initial", "hanzi"),
        default="pinyin-code",
        help=(
            "Mandarin transliteration to emit: 'pinyin-code' keeps the original "
            "tone/length code, while 'pinyin-initial' emits lowercase pinyin "
            "first letters only, and 'hanzi' keeps segmented Mandarin as Hanzi."
        ),
    )
    parser.add_argument(
        "--jieba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use jieba word segmentation for Chinese spans. Disable with "
            "--no-jieba to emit one token per Hanzi character before "
            "transliteration."
        ),
    )
    return parser.parse_args()


def main() -> None:
    require_dependencies()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    count = preprocess_file(
        args.input,
        args.output,
        args.preview,
        args.transliteration,
        args.jieba,
    )
    print(f"Wrote {count:,} processed documents to {args.output}")


if __name__ == "__main__":
    main()
