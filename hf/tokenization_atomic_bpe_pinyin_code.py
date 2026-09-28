"""Syllable-atomic BPE tokenizer for the compact Mandarin encoding.

Unlike character-level SentencePiece BPE, every BPE symbol here is one complete
two-character pinyin-code atom.  Learned merges therefore cannot cut a syllable
in half.  ``within_word`` tokenizers reset BPE at every encoded Jieba boundary;
``cross_word`` tokenizers apply the same learned merge ranks to an entire
contiguous encoded run.
"""

from __future__ import annotations

import heapq
import json
import string
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

# Keep this direct import even though the class is reached through the hybrid
# module.  Transformers' dynamic-module loader copies direct relative imports
# but does not reliably discover this transitive dependency in local folders.
from .tokenization_pinyin_code import PinyinCodeTokenizer as _PinyinCodeTokenizer
from .tokenization_hybrid_pinyin_code import (
    HybridPinyinCodeTokenizer,
    is_encoded_run,
    split_encoded_words,
)


VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "atomic_bpe_merges.json",
}
WITHIN_WORD = "within_word"
CROSS_WORD = "cross_word"
BOUNDARY_POLICIES = (WITHIN_WORD, CROSS_WORD)


class AtomicBPEPinyinCodeTokenizer(HybridPinyinCodeTokenizer):
    """Apply ranked BPE merges over indivisible two-character syllable atoms."""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        merges_file: str,
        boundary_policy: str | None = None,
        atomic_alphabet: str = string.ascii_uppercase + string.ascii_lowercase,
        atomic_digits: str = string.digits,
        **kwargs: Any,
    ) -> None:
        self.merges_file = merges_file
        payload = json.loads(Path(merges_file).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{merges_file} must contain a JSON object")

        stored_policy = payload.get("boundary_policy")
        policy = boundary_policy or stored_policy
        if policy not in BOUNDARY_POLICIES:
            raise ValueError(
                f"boundary_policy must be one of {BOUNDARY_POLICIES}, got {policy!r}"
            )
        if stored_policy is not None and stored_policy != policy:
            raise ValueError(
                "Tokenizer boundary policy disagrees with merge artifact: "
                f"requested={policy!r}, stored={stored_policy!r}"
            )
        self.boundary_policy = policy
        self.atomic_alphabet = atomic_alphabet
        self.atomic_digits = atomic_digits

        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError(f"{merges_file} must contain a `merges` list")
        merges: list[tuple[str, str]] = []
        for index, pair in enumerate(raw_merges):
            if not (
                isinstance(pair, list)
                and len(pair) == 2
                and all(isinstance(piece, str) and piece for piece in pair)
            ):
                raise ValueError(f"Invalid merge at rank {index}: {pair!r}")
            merges.append((pair[0], pair[1]))
        self._merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        # Atomic BPE is deterministic.  Suppress hybrid-only segmentation
        # settings if a generic pipeline happens to pass them through.
        for key in (
            "tokenization_mode",
            "sampling_temperature",
            "sampling_alpha",
            "sampling_beta",
            "sampling_epsilon",
            "sampling_seed",
        ):
            kwargs.pop(key, None)
        kwargs.setdefault("atomic_alphabet", atomic_alphabet)
        kwargs.setdefault("atomic_digits", atomic_digits)
        kwargs.setdefault("boundary_policy", policy)
        super().__init__(vocab_file=vocab_file, tokenization_mode="greedy", **kwargs)
        self._atomic_token_set = {
            f"{initial}{digit}"
            for initial in self.atomic_alphabet
            for digit in self.atomic_digits
        } | {
            f"{digit}{initial}"
            for initial in self.atomic_alphabet
            for digit in self.atomic_digits
        }
        self._validate_merges(merges)

    def _validate_merges(self, merges: Sequence[tuple[str, str]]) -> None:
        available = set(self._atomic_token_set)
        missing_atoms = sorted(available.difference(self.vocab))
        if missing_atoms:
            preview = ", ".join(repr(atom) for atom in missing_atoms[:10])
            raise ValueError(
                "Atomic BPE vocabulary is missing guaranteed fallback atoms: "
                f"{preview}"
            )
        for rank, (left, right) in enumerate(merges):
            if left not in available or right not in available:
                raise ValueError(
                    f"Merge rank {rank} references an unavailable piece: "
                    f"{left!r} + {right!r}"
                )
            merged = left + right
            if merged not in self.vocab:
                raise ValueError(
                    f"Merge rank {rank} produces {merged!r}, which is absent from vocab"
                )
            available.add(merged)

    @lru_cache(maxsize=262_144)
    def _apply_bpe(self, atomic_units: tuple[str, ...]) -> tuple[str, ...]:
        """Apply merge ranks in ``O((n + merges) log n)`` using linked nodes."""
        if len(atomic_units) < 2 or not self._merge_ranks:
            return atomic_units

        count = len(atomic_units)
        values = list(atomic_units)
        previous = [index - 1 for index in range(count)]
        following = [index + 1 for index in range(count)]
        following[-1] = -1
        alive = [True] * count
        versions = [0] * count
        queue: list[tuple[int, int, int, int, int]] = []

        def offer(left: int) -> None:
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            rank = self._merge_ranks.get((values[left], values[right]))
            if rank is not None:
                heapq.heappush(
                    queue,
                    (rank, left, right, versions[left], versions[right]),
                )

        for index in range(count - 1):
            offer(index)

        while queue:
            rank, left, right, left_version, right_version = heapq.heappop(queue)
            if (
                not alive[left]
                or not alive[right]
                or following[left] != right
                or versions[left] != left_version
                or versions[right] != right_version
                or self._merge_ranks.get((values[left], values[right])) != rank
            ):
                continue

            values[left] += values[right]
            versions[left] += 1
            alive[right] = False
            versions[right] += 1
            next_index = following[right]
            following[left] = next_index
            if next_index >= 0:
                previous[next_index] = left
            offer(previous[left])
            offer(left)

        output: list[str] = []
        index = 0
        while index >= 0:
            if alive[index]:
                output.append(values[index])
            index = following[index]
        return tuple(output)

    @staticmethod
    def _atomize_run(run: str) -> tuple[str, ...]:
        return tuple(run[index : index + 2] for index in range(0, len(run), 2))

    def _tokenize(self, text: str) -> list[str]:
        output: list[str] = []
        for item in text.split():
            if is_encoded_run(item):
                sequences = (
                    split_encoded_words(item)
                    if self.boundary_policy == WITHIN_WORD
                    else [item]
                )
                for sequence in sequences:
                    output.extend(self._apply_bpe(self._atomize_run(sequence)))
            elif item in self.vocab:
                output.append(item)
            else:
                char_tokens = self._surface_char_fallback(item)
                if char_tokens is not None:
                    output.extend(char_tokens)
                else:
                    output.extend(self._handle_unsupported_token(item))
        return output

    def _token_is_encoded_piece(self, token: str) -> bool:
        return bool(token) and len(token) % 2 == 0 and all(
            token[index : index + 2] in self._atomic_token_set
            for index in range(0, len(token), 2)
        )

    def set_tokenization_mode(self, mode: str) -> str:
        """Reject stochastic hybrid modes; atomic BPE has one ranked encoding."""
        if mode not in {"greedy", "bpe", None}:
            raise ValueError("Atomic BPE is deterministic and does not support softmax mode")
        return "bpe"

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_path = directory / f"{prefix}vocab.json"
        merges_path = directory / f"{prefix}atomic_bpe_merges.json"
        ordered_vocab = dict(sorted(self.vocab.items(), key=lambda item: item[1]))
        vocab_path.write_text(
            json.dumps(ordered_vocab, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        ordered_merges = [None] * len(self._merge_ranks)
        for pair, rank in self._merge_ranks.items():
            ordered_merges[rank] = list(pair)
        merges_path.write_text(
            json.dumps(
                {
                    "format": "babylm-pinyin-code-atomic-bpe-v1",
                    "boundary_policy": self.boundary_policy,
                    "merges": ordered_merges,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return str(vocab_path), str(merges_path)
