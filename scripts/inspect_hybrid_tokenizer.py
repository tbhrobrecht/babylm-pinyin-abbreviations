"""Inspect a hybrid pinyin-code tokenizer and optional corpus coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf.tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer
from train_hybrid_tokenizer import METADATA_NAME, atom_count, is_encoded_word


def load_metadata(tokenizer_dir: Path) -> dict:
    path = tokenizer_dir / METADATA_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_vocab(tokenizer: HybridPinyinCodeTokenizer, metadata: dict, top_n: int) -> None:
    vocab = tokenizer.get_vocab()
    selected_words = metadata.get("selected_word_frequencies", [])
    number_of_atomic_tokens = metadata.get("number_of_atomic_tokens", "unknown")
    number_of_preserved_tokens = metadata.get("number_of_preserved_tokens", "unknown")

    print("Vocabulary composition:")
    print(f"  final vocabulary size: {len(vocab)}")
    print(f"  atomic tokens: {number_of_atomic_tokens}")
    print(f"  preserved corpus tokens: {number_of_preserved_tokens}")
    print(f"  whole-word tokens: {len(selected_words)}")

    print("\nTop whole-word tokens:")
    for item in selected_words[:top_n]:
        print(f"  {item['token']}\t{item['frequency']}")

    print("\nLowest-frequency selected whole-word tokens:")
    for item in selected_words[-top_n:]:
        print(f"  {item['token']}\t{item['frequency']}")


def inspect_examples(tokenizer: HybridPinyinCodeTokenizer, examples: list[str]) -> None:
    if not examples:
        return
    print("\nTokenization examples:")
    for example in examples:
        tokens = tokenizer.tokenize(example)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(f"  input:  {example}")
        print(f"  tokens: {' '.join(tokens)}")
        print(f"  ids:    {ids}")


def inspect_corpus(tokenizer: HybridPinyinCodeTokenizer, input_paths: list[Path]) -> None:
    if not input_paths:
        return

    whole_word_vocab = {
        token
        for token in tokenizer.vocab
        if is_encoded_word(token) and atom_count(token) > 1
    }
    encoded_occurrences = 0
    whole_word_hits = 0
    fallback_occurrences = 0
    total_encoded_piece_tokens = 0
    total_lines = 0
    total_line_tokens = 0

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                total_lines += 1
                line_tokens = tokenizer.tokenize(text)
                total_line_tokens += len(line_tokens)
                for item in text.split():
                    if not is_encoded_word(item):
                        continue
                    encoded_occurrences += 1
                    if item in whole_word_vocab or atom_count(item) == 1:
                        whole_word_hits += 1
                        total_encoded_piece_tokens += 1
                    else:
                        fallback_occurrences += 1
                        total_encoded_piece_tokens += atom_count(item)

    coverage = whole_word_hits / encoded_occurrences if encoded_occurrences else 0.0
    fallback_rate = fallback_occurrences / encoded_occurrences if encoded_occurrences else 0.0
    avg_tokens_per_word = (
        total_encoded_piece_tokens / encoded_occurrences if encoded_occurrences else 0.0
    )
    avg_tokens_per_line = total_line_tokens / total_lines if total_lines else 0.0

    print("\nCorpus coverage:")
    print(f"  encoded word occurrences: {encoded_occurrences}")
    print(f"  whole-word occurrence coverage: {coverage:.4f}")
    print(f"  atomic-fallback occurrence rate: {fallback_rate:.4f}")
    print(f"  average tokens per encoded word: {avg_tokens_per_word:.4f}")
    print(f"  average tokens per line: {avg_tokens_per_line:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a hybrid pinyin-code tokenizer.")
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="*", default=[])
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument(
        "--example",
        action="append",
        default=["Y0J7 H2 X4Q3", "<QUESTION> Y0J7"],
        help="Example encoded text to tokenize. Can be passed more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = HybridPinyinCodeTokenizer.from_pretrained(args.tokenizer_dir)
    metadata = load_metadata(args.tokenizer_dir)
    inspect_vocab(tokenizer, metadata, args.top_n)
    inspect_examples(tokenizer, args.example)
    inspect_corpus(tokenizer, args.input)


if __name__ == "__main__":
    main()
