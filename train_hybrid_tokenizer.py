"""Train a Jieba-aware hybrid tokenizer from preprocessed pinyin-code text."""

from __future__ import annotations

import argparse
import json
import shutil
import string
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf.tokenization_hybrid_pinyin_code import ENCODED_WORD_RE
from preprocessing.preprocess import LATIN_ALNUM_RE, PUNCTUATION, should_preserve_fallback_token
from train_sentencepiece import SPECIAL_TOKENS


FIXED_SPECIAL_TOKENS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]
DEFAULT_INITIAL_ALPHABET = string.ascii_uppercase + string.ascii_lowercase
DEFAULT_DIGITS = string.digits
METADATA_NAME = "hybrid_tokenizer_metadata.json"


def is_encoded_word(token: str) -> bool:
    return bool(ENCODED_WORD_RE.fullmatch(token))


def atom_count(token: str) -> int:
    return len(token) // 2


def is_supported_preserved_token(token: str) -> bool:
    """Return true for non-encoded items produced by the current preprocessor."""
    if token in PUNCTUATION:
        return True
    if LATIN_ALNUM_RE.fullmatch(token):
        return True
    return should_preserve_fallback_token(token)


def atomic_tokens(initial_alphabet: str, digits: str) -> list[str]:
    atoms: list[str] = []
    seen: set[str] = set()
    for initial in initial_alphabet:
        for digit in digits:
            token = f"{initial}{digit}"
            if token not in seen:
                atoms.append(token)
                seen.add(token)
    return atoms


def iter_corpus_items(input_paths: Iterable[Path]) -> Iterable[tuple[Path, int, str]]:
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                for item in line.strip().split():
                    yield input_path, line_number, item


def ordered_vocab(tokens: Iterable[str]) -> dict[str, int]:
    vocab: dict[str, int] = {}
    for token in tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect_counts(
    input_paths: list[Path],
    permissive: bool,
    max_invalid_examples: int,
) -> tuple[Counter[str], Counter[str], Counter[str], dict[str, object]]:
    encoded_counts: Counter[str] = Counter()
    special_counts: Counter[str] = Counter()
    preserved_counts: Counter[str] = Counter()
    invalid_counts: Counter[str] = Counter()
    invalid_examples: list[dict[str, object]] = []
    total_items = 0
    valid_encoded_occurrences = 0

    for input_path, line_number, item in iter_corpus_items(input_paths):
        total_items += 1
        if item in SPECIAL_TOKENS or item in FIXED_SPECIAL_TOKENS:
            special_counts[item] += 1
        elif is_encoded_word(item):
            encoded_counts[item] += 1
            valid_encoded_occurrences += 1
        elif is_supported_preserved_token(item):
            preserved_counts[item] += 1
        else:
            invalid_counts[item] += 1
            if len(invalid_examples) < max_invalid_examples:
                invalid_examples.append(
                    {
                        "path": str(input_path),
                        "line": line_number,
                        "token": item,
                    }
                )

    if invalid_counts and not permissive:
        examples = ", ".join(
            f"{example['path']}:{example['line']} {example['token']!r}"
            for example in invalid_examples
        )
        raise ValueError(
            "Invalid or unsupported corpus items detected. "
            f"Examples: {examples}. Re-run with --permissive to map them to <unk>."
        )

    multi_atom_unique = sum(1 for token in encoded_counts if atom_count(token) > 1)
    stats = {
        "total_whitespace_items": total_items,
        "valid_encoded_word_occurrences": valid_encoded_occurrences,
        "unique_encoded_words": len(encoded_counts),
        "unique_multi_atom_encoded_words": multi_atom_unique,
        "special_token_occurrences": sum(special_counts.values()),
        "preserved_token_occurrences": sum(preserved_counts.values()),
        "invalid_or_unsupported_items": sum(invalid_counts.values()),
        "invalid_or_unsupported_unique_items": len(invalid_counts),
        "invalid_or_unsupported_examples": invalid_examples,
    }
    return encoded_counts, special_counts, preserved_counts, stats


def build_vocab(args: argparse.Namespace) -> tuple[dict[str, int], dict[str, object]]:
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be greater than zero")
    if args.min_word_frequency <= 0:
        raise ValueError("--min-word-frequency must be greater than zero")

    atoms = atomic_tokens(args.initial_alphabet, args.digits)
    encoded_counts, special_counts, preserved_counts, stats = collect_counts(
        args.input,
        permissive=args.permissive,
        max_invalid_examples=args.max_invalid_examples,
    )

    preserved_tokens = sorted(preserved_counts)
    base_tokens = FIXED_SPECIAL_TOKENS + SPECIAL_TOKENS + preserved_tokens + atoms
    base_vocab = ordered_vocab(base_tokens)
    if len(base_vocab) > args.vocab_size and not args.atomic_only:
        raise ValueError(
            "--vocab-size is smaller than the required special/preserved/atomic "
            f"base vocabulary ({len(base_vocab)} tokens)"
        )

    whole_word_budget = 0 if args.atomic_only else max(args.vocab_size - len(base_vocab), 0)
    candidates = [
        (token, count)
        for token, count in encoded_counts.items()
        if atom_count(token) > 1 and count >= args.min_word_frequency
    ]
    candidates.sort(key=lambda item: (-item[1], item[0]))
    selected_words = candidates[:whole_word_budget]

    vocab = ordered_vocab([*base_vocab, *(token for token, _ in selected_words)])
    least_selected_frequency = selected_words[-1][1] if selected_words else None
    metadata = {
        "format": "babylm-pinyin-code-hybrid-tokenizer-v1",
        "target_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "minimum_word_frequency": args.min_word_frequency,
        "permissive": args.permissive,
        "corpus_paths": [str(path) for path in args.input],
        "atomic_alphabet": args.initial_alphabet,
        "atomic_digits": args.digits,
        "number_of_atomic_tokens": len(atoms),
        "number_of_preserved_tokens": len(preserved_tokens),
        "number_of_word_tokens": len(selected_words),
        "creation_timestamp": datetime.now(timezone.utc).isoformat(),
        "least_frequent_selected_whole_word_frequency": least_selected_frequency,
        "selected_word_frequencies": [
            {"token": token, "frequency": count} for token, count in selected_words
        ],
        "top_preserved_tokens": [
            {"token": token, "frequency": count}
            for token, count in preserved_counts.most_common(50)
        ],
        "statistics": {
            **stats,
            "atomic_vocabulary_entries": len(atoms),
            "selected_whole_word_entries": len(selected_words),
            "final_vocabulary_size": len(vocab),
            "least_frequent_selected_whole_word_frequency": least_selected_frequency,
        },
    }
    return vocab, metadata


def token_scores_from_metadata(metadata: dict[str, object]) -> dict[str, float]:
    """Return a token->frequency map for softmax scoring from build metadata.

    Only whole-word entries carry a corpus frequency; atomic and preserved
    tokens have no stored frequency and are left out (they score 0.0, which the
    softmax treats as a constant that cancels out)."""
    scores: dict[str, float] = {}
    for entry in metadata.get("selected_word_frequencies", []) or []:
        if not isinstance(entry, dict):
            continue
        token = entry.get("token")
        frequency = entry.get("frequency")
        if isinstance(token, str) and isinstance(frequency, (int, float)):
            scores[token] = float(frequency)
    return scores


def write_tokenizer_files(output_dir: Path, vocab: dict[str, int], metadata: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "vocab.json", vocab)
    write_json(output_dir / METADATA_NAME, metadata)
    token_scores = token_scores_from_metadata(metadata)
    if token_scores:
        write_json(output_dir / "token_scores.json", token_scores)
    write_json(
        output_dir / "special_tokens_map.json",
        {
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
        },
    )
    write_json(
        output_dir / "tokenizer_config.json",
        {
            "tokenizer_class": "HybridPinyinCodeTokenizer",
            "auto_map": {
                "AutoTokenizer": [
                    "tokenization_hybrid_pinyin_code.HybridPinyinCodeTokenizer",
                    None,
                ]
            },
            "model_max_length": 1000000000000000019884624838656,
            "add_bos_token": False,
            "add_eos_token": False,
            "strict_validation": not bool(metadata.get("permissive", False)),
            "readable_decode": False,
            # Deterministic greedy longest-match is the default segmentation
            # policy; softmax sampling is opt-in. These fields are optional and
            # older tokenizer directories without them still load as greedy.
            "tokenization_mode": "greedy",
            "sampling_temperature": 1.0,
            "sampling_alpha": 1.0,
            "sampling_beta": 1.0,
            "sampling_epsilon": 1e-8,
            "sampling_seed": None,
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "mask_token": "<mask>",
        },
    )
    shutil.copy2(
        Path(__file__).resolve().parent / "hf" / "tokenization_hybrid_pinyin_code.py",
        output_dir / "tokenization_hybrid_pinyin_code.py",
    )


def print_summary(metadata: dict[str, object]) -> None:
    stats = metadata["statistics"]
    assert isinstance(stats, dict)
    print("Hybrid tokenizer build statistics:")
    for key in [
        "total_whitespace_items",
        "valid_encoded_word_occurrences",
        "unique_encoded_words",
        "unique_multi_atom_encoded_words",
        "special_token_occurrences",
        "preserved_token_occurrences",
        "invalid_or_unsupported_items",
        "atomic_vocabulary_entries",
        "selected_whole_word_entries",
        "final_vocabulary_size",
        "least_frequent_selected_whole_word_frequency",
    ]:
        print(f"  {key}: {stats.get(key)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hybrid pinyin-code tokenizer from preprocessed BabyLM text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more UTF-8 processed corpus files.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--min-word-frequency", type=int, default=20)
    parser.add_argument(
        "--atomic-only",
        action="store_true",
        help="Train only special/preserved/atomic tokens and no whole-word entries.",
    )
    parser.add_argument("--initial-alphabet", default=DEFAULT_INITIAL_ALPHABET)
    parser.add_argument("--digits", default=DEFAULT_DIGITS)
    parser.add_argument(
        "--permissive",
        action="store_true",
        help="Allow invalid corpus items and record examples instead of failing.",
    )
    parser.add_argument("--max-invalid-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vocab, metadata = build_vocab(args)
    write_tokenizer_files(args.output_dir, vocab, metadata)
    print_summary(metadata)
    print(f"Wrote trained hybrid tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
