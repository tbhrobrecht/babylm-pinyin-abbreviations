"""Inspect greedy/softmax segmentation of the hybrid pinyin-code tokenizer.

Prints the atomic units, Jieba boundaries, valid candidates, softmax scores and
probabilities, and the selected tokens for one encoded example. This script is
read-only and never modifies tokenizer files.

Example::

    python scripts/inspect_tokenizer.py \
        --tokenizer tokenizers/babylm_zho_hybrid_16k \
        --text "Y0J7 H2 X4Q3" \
        --mode softmax --temperature 1.0 --samples 20 --seed 42
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hf.tokenization_hybrid_pinyin_code import (  # noqa: E402
    ENCODED_WORD_RE,
    HybridPinyinCodeTokenizer,
)


def softmax_probabilities(scores: list[float], temperature: float) -> list[float]:
    """Return the numerically stable softmax over ``scores / temperature``."""
    if not scores:
        return []
    logits = [score / temperature for score in scores]
    max_logit = max(logits)
    weights = [math.exp(logit - max_logit) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def describe_word(tokenizer: HybridPinyinCodeTokenizer, word: str) -> None:
    """Print the per-position candidate breakdown for one encoded word."""
    atomic_units = tokenizer._atomize(word)
    print(f"  word: {word}")
    print(f"    atomic units: {atomic_units}")
    for position in range(len(atomic_units)):
        candidates = tokenizer.get_valid_matches(atomic_units, position)
        if not candidates:
            print(f"    position {position}: <no candidates>")
            continue
        scores = tokenizer._score_candidates(candidates)
        probabilities = softmax_probabilities(scores, tokenizer.sampling_temperature)
        print(f"    position {position} (atom {atomic_units[position]}):")
        for candidate, score, probability in zip(candidates, scores, probabilities):
            frequency = tokenizer._token_scores.get(candidate.token, 0.0)
            print(
                f"      {candidate.token:<12} id={candidate.token_id:<6} "
                f"len={candidate.atomic_length} freq={frequency:g} "
                f"score={score:.4f} p={probability:.4f}"
            )


def reconstruct_atomic_sequence(tokens: list[str]) -> list[str]:
    """Return the atomic units implied by the selected encoded tokens."""
    atoms: list[str] = []
    for token in tokens:
        if ENCODED_WORD_RE.fullmatch(token):
            atoms.extend(token[index : index + 2] for index in range(0, len(token), 2))
        else:
            atoms.append(token)
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect hybrid tokenizer segmentation.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Hybrid tokenizer directory.")
    parser.add_argument("--text", required=True, help="Encoded (preprocessed) example text.")
    parser.add_argument("--mode", choices=("greedy", "softmax"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1, help="Repeated samples for softmax mode.")
    args = parser.parse_args()

    load_path = args.tokenizer
    if load_path.name == "vocab.json":
        load_path = load_path.parent

    tokenizer = HybridPinyinCodeTokenizer.from_pretrained(
        load_path,
        tokenization_mode=args.mode,
        sampling_temperature=args.temperature,
        sampling_alpha=args.alpha,
        sampling_beta=args.beta,
        sampling_epsilon=args.epsilon,
        sampling_seed=args.seed,
    )

    words = args.text.split()
    print(f"Input encoded text: {args.text}")
    print(f"Jieba boundaries (whitespace-separated words): {words}")
    print(f"Active mode: {tokenizer.tokenization_mode}")
    print(f"Sampling temperature: {tokenizer.sampling_temperature:g}")
    print(f"Sampling alpha/beta/epsilon: {tokenizer.sampling_alpha:g}/"
          f"{tokenizer.sampling_beta:g}/{tokenizer.sampling_epsilon:g}")
    print(f"Sampling seed: {tokenizer.sampling_seed}")

    print("\nValid candidates at each position:")
    for word in words:
        if ENCODED_WORD_RE.fullmatch(word):
            describe_word(tokenizer, word)
        else:
            in_vocab = word in tokenizer.vocab
            print(f"  word: {word} (special/preserved token, in_vocab={in_vocab})")

    samples = max(1, args.samples) if tokenizer.tokenization_mode == "softmax" else 1
    print(f"\nSelected segmentation ({samples} sample(s)):")
    for sample_index in range(samples):
        tokens = tokenizer.tokenize(args.text)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        reconstructed = reconstruct_atomic_sequence(tokens)
        label = f"  sample {sample_index + 1}" if samples > 1 else "  result"
        print(f"{label}:")
        print(f"    selected tokens:      {tokens}")
        print(f"    selected token ids:   {ids}")
        print(f"    reconstructed atoms:  {reconstructed}")


if __name__ == "__main__":
    main()
