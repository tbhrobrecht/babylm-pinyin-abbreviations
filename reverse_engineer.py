"""Reverse-engineer likely Mandarin text from BabyLM pinyin-initial codes.

The preprocessing format is lossy: each Chinese word becomes a compact token
that keeps pinyin initials, a syllable-length bucket, and partial tone
information. This script therefore reports likely candidates learned from a
source corpus instead of claiming a unique decryption.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import jieba
from pypinyin import Style, pinyin

from preprocessing.preprocess import (
    CHINESE_RE,
    PUNCTUATION,
    TOKEN_RE,
    chinese_word_to_initial_codes,
    normalize_text,
    read_jsonl,
    require_dependencies,
)


DEFAULT_MESSAGE = (
    "W6M7 Y6 j2D6 z6J6 X2L6 s7 g6 H7 D6 r2 N6 Y6 s7 j2D6 z6J6 H7"
)

Codebook = dict[str, Counter[str]]


def iter_code_word_pairs(text: str) -> list[tuple[str, str]]:
    """Yield ``(code, source_word)`` pairs using the project's preprocessor rules."""
    pairs: list[tuple[str, str]] = []

    for part in TOKEN_RE.findall(normalize_text(text)):
        if part.startswith("<") and part.endswith(">"):
            pairs.append((part, part))
        elif CHINESE_RE.fullmatch(part):
            for word in jieba.cut(part, cut_all=False):
                word = word.strip()
                if not word or not CHINESE_RE.search(word):
                    continue
                code = chinese_word_to_initial_codes(word)
                if code:
                    pairs.append((code, word))
        elif part in PUNCTUATION:
            pairs.append((part, part))
        elif part.isalpha():
            word = part.upper() if part.upper() in {"A", "B", "C", "D"} else part.lower()
            pairs.append((word, word))

    return pairs


def build_codebook(
    corpus_path: Path,
    max_docs: int | None,
    target_codes: set[str] | None,
) -> Codebook:
    """Build a frequency table from pinyin-code tokens to observed source words."""
    codebook: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for index, obj in enumerate(read_jsonl(corpus_path), start=1):
        if max_docs is not None and index > max_docs:
            break

        text = str(obj.get("text", ""))
        for code, word in iter_code_word_pairs(text):
            if target_codes is not None and code not in target_codes:
                continue
            codebook[code][word] += 1

    return dict(codebook)


def decode_message(
    encrypted: str,
    codebook: Codebook,
    max_candidates: int,
) -> tuple[str, list[tuple[str, list[tuple[str, int]]]]]:
    """Return a greedy best guess plus per-token candidate lists."""
    decoded_words: list[str] = []
    token_candidates: list[tuple[str, list[tuple[str, int]]]] = []

    for token in encrypted.split():
        candidates = codebook.get(token, Counter()).most_common(max_candidates)
        token_candidates.append((token, candidates))
        decoded_words.append(candidates[0][0] if candidates else f"<{token}?>")

    return "".join(decoded_words), token_candidates


def hanzi_to_pinyin(text: str) -> str:
    """Convert decoded Hanzi to readable tone-mark pinyin where possible."""
    syllables = pinyin(text, style=Style.TONE, heteronym=False, errors="default")
    return " ".join(item[0] for item in syllables if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find likely Mandarin source words for BabyLM pinyin-code text."
    )
    parser.add_argument(
        "message",
        nargs="?",
        default=None,
        help="Space-separated BabyLM pinyin-code message to decode.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/10k_babylm_zho.jsonl"),
        help="Original BabyLM JSONL file with raw text fields.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=2_000,
        help="Only scan the first N documents while building the codebook.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=8,
        help="Number of candidate source words to show for each encrypted token.",
    )
    parser.add_argument(
        "--full-codebook",
        action="store_true",
        help="Collect every observed code instead of only codes from the message.",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Decode the built-in example message instead of prompting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_dependencies()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not args.corpus.exists():
        raise SystemExit(f"Corpus file not found: {args.corpus}")

    message = args.message
    if args.example:
        message = DEFAULT_MESSAGE
    if message is None:
        message = input("Enter BabyLM pinyin-code text: ").strip()
    if not message:
        raise SystemExit("No input provided.")

    target_codes = None if args.full_codebook else set(message.split())
    codebook = build_codebook(args.corpus, args.max_docs, target_codes)
    best_guess, token_candidates = decode_message(
        message,
        codebook,
        max_candidates=args.max_candidates,
    )

    print()
    print("Best Hanzi guess:")
    print(best_guess)
    print()
    print("Pinyin for that guess:")
    print(hanzi_to_pinyin(best_guess))
    print()
    print("Token candidates:")
    for token, candidates in token_candidates:
        if not candidates:
            print(f"{token}: no candidates found")
            continue
        options = ", ".join(f"{word} ({count})" for word, count in candidates)
        print(f"{token}: {options}")


if __name__ == "__main__":
    main()
