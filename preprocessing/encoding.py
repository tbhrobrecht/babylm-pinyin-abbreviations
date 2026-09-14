"""Grammar of the whitespace-free pinyin-code encoding.

Every Jieba word is written as one run of two-character syllable atoms. The
first syllable keeps the ``initial + digit`` order, and every later syllable of
the same word is written ``digit + initial``::

    中国  ->  Z4 + g2  ->  "Z42g"

Because a word can only start with a letter and can only continue with a digit,
words are self-delimiting and the corpus needs no whitespace between them::

    我 爱 北京  ->  "W6" "a6" "B7" + "3J"  ->  "W6a6B73J"

This module is intentionally dependency-free so tokenizer-training and
corpus-counting utilities can use it without importing jieba or pypinyin. The
same grammar is duplicated in ``hf/tokenization_hybrid_pinyin_code.py`` and
``hf/tokenization_pinyin_code.py``, which must stay importable as standalone
Transformers remote-code modules.
"""

from __future__ import annotations

import re
from typing import Iterable

HEAD_ATOM_PATTERN = r"[A-Za-z][0-9]"
TAIL_ATOM_PATTERN = r"[0-9][A-Za-z]"
ENCODED_WORD_PATTERN = rf"{HEAD_ATOM_PATTERN}(?:{TAIL_ATOM_PATTERN})*"

# One complete encoded word.
ENCODED_WORD_RE = re.compile(rf"^{ENCODED_WORD_PATTERN}$")
# One or more encoded words written without any separator between them.
ENCODED_RUN_RE = re.compile(rf"^(?:{ENCODED_WORD_PATTERN})+$")
# Unanchored form used to scan a run for its successive words.
ENCODED_WORD_SCAN_RE = re.compile(ENCODED_WORD_PATTERN)
# One syllable atom in either word position.
ATOM_RE = re.compile(rf"^(?:{HEAD_ATOM_PATTERN}|{TAIL_ATOM_PATTERN})$")
# Any sequence of atoms, including a word-internal piece that starts with a
# word-continuing atom such as ``3J``.
ENCODED_PIECE_RE = re.compile(
    rf"^(?:{HEAD_ATOM_PATTERN}|{TAIL_ATOM_PATTERN})(?:{TAIL_ATOM_PATTERN})*$"
)


def continuation_form(code: str) -> str:
    """Return the word-internal form of a syllable code (digit before initial)."""
    if len(code) != 2:
        raise ValueError(f"Expected a two-character syllable code, got {code!r}")
    return f"{code[1]}{code[0]}"


def is_encoded_word(text: str) -> bool:
    """Return true for exactly one encoded word."""
    return bool(ENCODED_WORD_RE.fullmatch(text))


def is_encoded_run(text: str) -> bool:
    """Return true for one or more encoded words written without separators."""
    return bool(ENCODED_RUN_RE.fullmatch(text))


def split_encoded_words(run: str) -> list[str]:
    """Split a whitespace-free encoded run back into its Jieba words.

    The scan is unambiguous: a word starts at every letter and absorbs every
    following ``digit + initial`` atom.
    """
    words = ENCODED_WORD_SCAN_RE.findall(run)
    if "".join(words) != run:
        raise ValueError(f"Not a valid encoded word run: {run!r}")
    return words


def encoded_word_atoms(word: str) -> list[str]:
    """Split one encoded word into its two-character syllable atoms."""
    return [word[index : index + 2] for index in range(0, len(word), 2)]


def join_corpus_tokens(tokens: Iterable[tuple[str, bool]]) -> str:
    """Join ``(token, is_encoded)`` pairs into one corpus line.

    Adjacent encoded words are concatenated because the encoding already marks
    the boundary; every other neighbouring pair keeps a single space.
    """
    pieces: list[str] = []
    previous_encoded = False
    for token, is_encoded in tokens:
        if pieces and not (is_encoded and previous_encoded):
            pieces.append(" ")
        pieces.append(token)
        previous_encoded = is_encoded
    return "".join(pieces)


def iter_corpus_words(line: str) -> Iterable[str]:
    """Yield corpus items from a processed line, expanding encoded runs.

    Whitespace-separated items are returned as-is unless they are an encoded
    run, in which case each contained word is yielded separately.
    """
    for item in line.split():
        if is_encoded_run(item):
            yield from split_encoded_words(item)
        else:
            yield item
