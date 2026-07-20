"""Hybrid Jieba-word tokenizer for preprocessed pinyin-code text.

The tokenizer keeps a single shared segmentation implementation and exposes two
selectable *tokenization modes* over the same vocabulary and boundary policy:

* ``greedy``  – deterministic left-to-right longest match (default).
* ``softmax`` – stochastic left-to-right sampling among the vocabulary tokens
  that are valid at the current atomic position.

Both modes enumerate candidates with the same :meth:`get_valid_matches` method,
never split an atomic Initial+Digit unit, never cross a whitespace-separated
Jieba boundary, and rely on the same atomic fallback. Only the *selection*
strategy differs, so the experiment isolates the segmentation policy.

The softmax mode is *local autoregressive segmentation sampling*: at each
position it samples one token from the valid next tokens and advances. It does
not enumerate every complete segmentation path of the word.
"""

from __future__ import annotations

import contextlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}
ENCODED_WORD_RE = re.compile(r"^(?:[A-Za-z][0-9])+$")
ATOM_RE = re.compile(r"^[A-Za-z][0-9]$")

# Optional sidecar files used to recover per-token frequency/score signal.
TOKEN_SCORES_FILE = "token_scores.json"
HYBRID_METADATA_FILE = "hybrid_tokenizer_metadata.json"

GREEDY_MODE = "greedy"
SOFTMAX_MODE = "softmax"
SUPPORTED_TOKENIZATION_MODES = (GREEDY_MODE, SOFTMAX_MODE)
# Default preserves the historical deterministic longest-match behavior.
DEFAULT_TOKENIZATION_MODE = GREEDY_MODE


class HybridTokenizerError(ValueError):
    """Raised for malformed encoded input, missing atomic fallback, or bad config.

    Subclasses :class:`ValueError` so existing ``assertRaises(ValueError)`` call
    sites and error handling keep working.
    """


@dataclass(frozen=True)
class TokenMatch:
    """One vocabulary token that matches the atomic sequence at a position.

    ``atomic_length`` is the number of atomic Initial+Digit units the token
    covers; ``end`` is the exclusive atomic index after the match.
    """

    token: str
    token_id: int
    start: int
    end: int
    atomic_length: int


class HybridPinyinCodeTokenizer(PreTrainedTokenizer):
    """Tokenize encoded Jieba words with greedy or softmax segmentation.

    The tokenizer expects text that has already passed through the repository's
    pinyin-code preprocessing. Whitespace is used as word-boundary metadata and
    never becomes a token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        strict_validation: bool = True,
        readable_decode: bool = False,
        tokenization_mode: str = DEFAULT_TOKENIZATION_MODE,
        sampling_temperature: float = 1.0,
        sampling_alpha: float = 1.0,
        sampling_beta: float = 1.0,
        sampling_epsilon: float = 1e-8,
        sampling_seed: int | None = None,
        token_scores: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        self.vocab_file = vocab_file
        self.vocab = self._load_vocab(vocab_file)
        self.ids_to_tokens = {token_id: token for token, token_id in self.vocab.items()}
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.strict_validation = strict_validation
        self.readable_decode = readable_decode

        # --- Tokenization mode configuration (shared by both modes) ----------
        self.tokenization_mode = self._normalize_mode(tokenization_mode)
        self.sampling_temperature = float(sampling_temperature)
        self.sampling_alpha = float(sampling_alpha)
        self.sampling_beta = float(sampling_beta)
        self.sampling_epsilon = float(sampling_epsilon)
        self.sampling_seed = sampling_seed
        self._validate_sampling_params()

        # Maximum whole-word span in atomic units. Bounds candidate enumeration
        # so lookup is O(n * L) rather than a full vocabulary scan per position.
        self._max_atomic_span = self._compute_max_atomic_span()

        # Per-token frequency/score signal used only by softmax scoring. Tokens
        # without a stored value score 0.0; with the alpha*log(f+eps) term this
        # is a constant that cancels in the softmax, i.e. length-only scoring.
        self._token_scores = self._resolve_token_scores(vocab_file, token_scores)

        # Tokenizer-local RNG. Keeps softmax reproducible without mutating global
        # random state. Greedy tokenization never consumes it.
        self._rng = random.Random(sampling_seed)

        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("bos_token", "<s>")
        kwargs.setdefault("eos_token", "</s>")
        kwargs.setdefault("mask_token", "<mask>")
        kwargs.setdefault("add_bos_token", add_bos_token)
        kwargs.setdefault("add_eos_token", add_eos_token)
        kwargs.setdefault("strict_validation", strict_validation)
        kwargs.setdefault("readable_decode", readable_decode)
        # Persist the mode configuration through tokenizer_config.json so that a
        # saved/reloaded tokenizer keeps its selection strategy. token_scores is
        # intentionally NOT serialized inline (it can be large); it round-trips
        # through the token_scores.json sidecar written by save_vocabulary.
        kwargs.setdefault("tokenization_mode", self.tokenization_mode)
        kwargs.setdefault("sampling_temperature", self.sampling_temperature)
        kwargs.setdefault("sampling_alpha", self.sampling_alpha)
        kwargs.setdefault("sampling_beta", self.sampling_beta)
        kwargs.setdefault("sampling_epsilon", self.sampling_epsilon)
        kwargs.setdefault("sampling_seed", self.sampling_seed)
        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # Vocabulary loading / configuration
    # ------------------------------------------------------------------ #
    @staticmethod
    def _load_vocab(vocab_file: str) -> dict[str, int]:
        with Path(vocab_file).open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        if not isinstance(vocab, dict):
            raise ValueError(f"{vocab_file} must contain a JSON object")

        normalized: dict[str, int] = {}
        seen_ids: set[int] = set()
        for token, token_id in vocab.items():
            if not isinstance(token, str) or not isinstance(token_id, int):
                raise ValueError("vocab.json must map string tokens to integer ids")
            if token_id in seen_ids:
                raise ValueError(f"Duplicate token id in vocab.json: {token_id}")
            normalized[token] = token_id
            seen_ids.add(token_id)

        expected_ids = set(range(len(normalized)))
        if seen_ids != expected_ids:
            raise ValueError("vocab.json ids must be contiguous from 0")
        return normalized

    @staticmethod
    def _normalize_mode(mode: str | None) -> str:
        """Lowercase, validate, and default a tokenization mode name."""
        if mode is None:
            return DEFAULT_TOKENIZATION_MODE
        normalized = str(mode).strip().lower()
        if normalized not in SUPPORTED_TOKENIZATION_MODES:
            raise HybridTokenizerError(
                f"Unsupported tokenization_mode {mode!r}. Supported modes: "
                f"{', '.join(SUPPORTED_TOKENIZATION_MODES)}."
            )
        return normalized

    def _validate_sampling_params(self) -> None:
        """Validate softmax parameters. Greedy never depends on them."""
        if not math.isfinite(self.sampling_temperature) or self.sampling_temperature <= 0.0:
            raise HybridTokenizerError(
                "sampling_temperature must be a finite value greater than zero, "
                f"got {self.sampling_temperature!r}."
            )
        if not math.isfinite(self.sampling_alpha):
            raise HybridTokenizerError(f"sampling_alpha must be finite, got {self.sampling_alpha!r}.")
        if not math.isfinite(self.sampling_beta):
            raise HybridTokenizerError(f"sampling_beta must be finite, got {self.sampling_beta!r}.")
        if not math.isfinite(self.sampling_epsilon) or self.sampling_epsilon <= 0.0:
            raise HybridTokenizerError(
                "sampling_epsilon must be a finite value greater than zero, "
                f"got {self.sampling_epsilon!r}."
            )

    def _compute_max_atomic_span(self) -> int:
        max_span = 1
        for token in self.vocab:
            if ENCODED_WORD_RE.fullmatch(token):
                span = len(token) // 2
                if span > max_span:
                    max_span = span
        return max_span

    def _resolve_token_scores(
        self,
        vocab_file: str,
        explicit: dict[str, float] | None,
    ) -> dict[str, float]:
        """Return the strongest available per-token frequency/score signal.

        Priority: explicit argument > token_scores.json sidecar > the
        ``selected_word_frequencies`` field of the hybrid metadata file. When
        nothing is found an empty mapping is returned and softmax degenerates to
        length-only scoring (see :meth:`_score_candidates`).
        """
        if explicit is not None:
            return {str(token): float(score) for token, score in explicit.items()}

        directory = Path(vocab_file).resolve().parent

        scores_path = directory / TOKEN_SCORES_FILE
        if scores_path.exists():
            try:
                data = json.loads(scores_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = None
            if isinstance(data, dict):
                return {
                    str(token): float(score)
                    for token, score in data.items()
                    if isinstance(score, (int, float))
                }

        metadata_path = directory / HYBRID_METADATA_FILE
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = None
            if isinstance(metadata, dict):
                scores: dict[str, float] = {}
                for entry in metadata.get("selected_word_frequencies", []) or []:
                    if not isinstance(entry, dict):
                        continue
                    token = entry.get("token")
                    frequency = entry.get("frequency")
                    if isinstance(token, str) and isinstance(frequency, (int, float)):
                        scores[token] = float(frequency)
                if scores:
                    return scores

        return {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self.vocab)
        vocab.update(self.added_tokens_encoder)
        return vocab

    # ------------------------------------------------------------------ #
    # Tokenization-mode public API
    # ------------------------------------------------------------------ #
    def set_tokenization_mode(self, mode: str) -> str:
        """Switch the active tokenization mode. Vocabulary and ids are unchanged."""
        self.tokenization_mode = self._normalize_mode(mode)
        return self.tokenization_mode

    @contextlib.contextmanager
    def use_mode(self, mode: str):
        """Temporarily use ``mode`` (e.g. greedy for evaluation), then restore.

        Useful for forcing deterministic greedy tokenization during evaluation
        even when the tokenizer is otherwise configured for softmax training::

            with tokenizer.use_mode("greedy"):
                eval_ids = tokenizer(eval_text)
        """
        previous = self.tokenization_mode
        self.set_tokenization_mode(mode)
        try:
            yield self
        finally:
            self.tokenization_mode = previous

    def reseed(self, sampling_seed: int | None) -> None:
        """Reset the tokenizer-local softmax RNG to a new seed.

        Use this to derive worker-local streams (e.g. base seed + worker id +
        rank) before tokenizing in a separate process. Greedy is unaffected.
        """
        self.sampling_seed = sampling_seed
        self._rng = random.Random(sampling_seed)

    # ------------------------------------------------------------------ #
    # Shared candidate enumeration (used by both modes)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _atomize(word: str) -> list[str]:
        """Split a validated encoded word into its atomic Initial+Digit units."""
        return [word[index : index + 2] for index in range(0, len(word), 2)]

    def get_valid_matches(
        self,
        atomic_units: Sequence[str],
        start: int,
    ) -> list[TokenMatch]:
        """Return every vocabulary token that matches ``atomic_units`` at ``start``.

        Each candidate covers one or more complete atomic units, so an atomic
        Initial+Digit unit can never be split. Enumeration is bounded by the
        longest whole-word span in the vocabulary. Because every atom is present
        in the vocabulary, the length-1 candidate is always available for valid
        encoded input, so the returned list is non-empty in the normal case.

        Note: candidates are keyed by covered length, and the token for a given
        length is uniquely determined by the atomic units it spans, so no two
        returned candidates share the same ``atomic_length``.
        """
        matches: list[TokenMatch] = []
        limit = min(self._max_atomic_span, len(atomic_units) - start)
        for length in range(1, limit + 1):
            candidate = "".join(atomic_units[start : start + length])
            token_id = self.vocab.get(candidate)
            if token_id is not None:
                matches.append(
                    TokenMatch(
                        token=candidate,
                        token_id=token_id,
                        start=start,
                        end=start + length,
                        atomic_length=length,
                    )
                )
        return matches

    # ------------------------------------------------------------------ #
    # Greedy longest-match selection
    # ------------------------------------------------------------------ #
    def _choose_longest_candidate(self, candidates: list[TokenMatch]) -> TokenMatch:
        """Deterministic greedy selection.

        Tie-breaking order (documented and stable):
          1. greater atomic length (more units covered);
          2. higher stored token score/frequency;
          3. lower token id;
          4. lexicographically smaller token as a final fallback.

        Rules 2-4 exist for completeness and determinism: at a single position
        the candidate token for each covered length is uniquely determined by
        the atomic units it spans, so distinct candidates always have distinct
        atomic lengths and rule 1 already resolves the choice. We never rely on
        set/dict iteration order.
        """
        return max(
            candidates,
            key=lambda match: (
                match.atomic_length,
                self._token_scores.get(match.token, 0.0),
                -match.token_id,
                # Negate ordinals so that, under max(), the lexicographically
                # smaller token wins the final tie-break.
                tuple(-ord(character) for character in match.token),
            ),
        )

    # ------------------------------------------------------------------ #
    # Softmax stochastic selection
    # ------------------------------------------------------------------ #
    def _score_candidates(self, candidates: list[TokenMatch]) -> list[float]:
        r"""Score candidates as ``s(t) = alpha*log(f(t)+eps) + beta*|t|``.

        ``f(t)`` is the stored token frequency/score (0.0 when unavailable),
        ``|t|`` is the atomic length, ``eps`` avoids ``log(0)``. When no
        frequency metadata exists every ``f(t)`` is 0.0, so the frequency term
        is a shared constant that cancels in the softmax; the result is
        equivalent to ``alpha = 0`` (length-only scoring driven by ``beta``).
        """
        scores: list[float] = []
        for match in candidates:
            frequency = self._token_scores.get(match.token, 0.0)
            score = (
                self.sampling_alpha * math.log(frequency + self.sampling_epsilon)
                + self.sampling_beta * match.atomic_length
            )
            scores.append(score)
        return scores

    def _sample_softmax(
        self,
        candidates: list[TokenMatch],
        scores: list[float],
    ) -> TokenMatch:
        r"""Sample one candidate using a numerically stable softmax.

        With logits ``z_j = s(t_j) / tau`` the probabilities are
        ``p_j = exp(z_j - max_k z_k) / sum_m exp(z_m - max_k z_k)``. Subtracting
        the max keeps ``exp`` from overflowing. A single uniform draw from the
        tokenizer-local RNG then selects a candidate by cumulative weight.
        """
        if len(candidates) == 1:
            # Only the atomic fallback is valid here; no random draw needed, so
            # the RNG stream stays aligned across inputs that differ only in
            # positions with a single candidate.
            return candidates[0]

        temperature = self.sampling_temperature
        logits = [score / temperature for score in scores]
        max_logit = max(logits)
        weights = [math.exp(logit - max_logit) for logit in logits]
        total = sum(weights)

        threshold = self._rng.random() * total
        cumulative = 0.0
        for match, weight in zip(candidates, weights):
            cumulative += weight
            if threshold < cumulative:
                return match
        # Fall back to the last candidate to guard against floating-point drift.
        return candidates[-1]

    # ------------------------------------------------------------------ #
    # Segmentation dispatch
    # ------------------------------------------------------------------ #
    def _segment_atomic_units(
        self,
        atomic_units: Sequence[str],
        encoded_word: str,
    ) -> list[str]:
        """Segment one Jieba word's atomic units with the active mode.

        Both modes share candidate enumeration and only differ in selection.
        Matching is confined to this single word, so no token crosses a
        whitespace-separated Jieba boundary.
        """
        tokens: list[str] = []
        position = 0
        total = len(atomic_units)
        while position < total:
            candidates = self.get_valid_matches(atomic_units, position)
            if not candidates:
                if not self.strict_validation:
                    # Permissive fallback preserves the historical behavior of
                    # emitting the raw atom (mapped to <unk> downstream) when the
                    # vocabulary lacks an atom, instead of failing.
                    tokens.append(atomic_units[position])
                    position += 1
                    continue
                local = list(atomic_units[position : position + self._max_atomic_span])
                raise HybridTokenizerError(
                    "No valid vocabulary match and no atomic fallback available "
                    f"for encoded word {encoded_word!r} at atomic position "
                    f"{position} (local atomic units {local}); "
                    f"tokenization_mode={self.tokenization_mode!r}."
                )
            if self.tokenization_mode == SOFTMAX_MODE:
                scores = self._score_candidates(candidates)
                selected = self._sample_softmax(candidates, scores)
            else:
                selected = self._choose_longest_candidate(candidates)
            tokens.append(selected.token)
            position = selected.end
        return tokens

    def _handle_unsupported_token(self, token: str) -> list[str]:
        if self.strict_validation:
            raise ValueError(
                f"Unsupported or malformed pinyin-code token {token!r}. "
                "Expected a known special/preserved token or text matching "
                "^(?:[A-Za-z][0-9])+$."
            )
        return [self.unk_token]

    def _tokenize(self, text: str) -> list[str]:
        output: list[str] = []
        for item in text.split():
            if ENCODED_WORD_RE.fullmatch(item):
                # Encoded Jieba word: segment its atoms with the active mode.
                # A whole word that is itself in the vocabulary is selected as
                # the longest match under greedy, matching the previous
                # whole-word-lookup behavior.
                output.extend(self._segment_atomic_units(self._atomize(item), item))
            elif item in self.vocab:
                # Special/preserved marker preserved verbatim.
                output.append(item)
            else:
                output.extend(self._handle_unsupported_token(item))
        return output

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def _token_is_atom(self, token: str) -> bool:
        return bool(ATOM_RE.fullmatch(token))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        if self.readable_decode:
            return " ".join(tokens)

        pieces: list[str] = []
        atom_buffer: list[str] = []
        for token in tokens:
            if self._token_is_atom(token):
                atom_buffer.append(token)
                continue
            if atom_buffer:
                pieces.append("".join(atom_buffer))
                atom_buffer.clear()
            pieces.append(token)
        if atom_buffer:
            pieces.append("".join(atom_buffer))
        return " ".join(pieces)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        readable: bool | None = None,
        **kwargs: Any,
    ) -> str:
        if readable is None:
            return super().decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )

        previous = self.readable_decode
        self.readable_decode = readable
        try:
            return super().decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
        finally:
            self.readable_decode = previous

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        output = list(token_ids_0)
        if self.add_bos_token and self.bos_token_id is not None:
            output = [self.bos_token_id] + output
        if self.add_eos_token and self.eos_token_id is not None:
            output = output + [self.eos_token_id]
        if token_ids_1 is not None:
            output += list(token_ids_1)
            if self.add_eos_token and self.eos_token_id is not None:
                output.append(self.eos_token_id)
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            special_ids = set(self.all_special_ids)
            return [1 if token_id in special_ids else 0 for token_id in token_ids_0]

        mask = [0] * len(token_ids_0)
        if self.add_bos_token and self.bos_token_id is not None:
            mask = [1] + mask
        if self.add_eos_token and self.eos_token_id is not None:
            mask = mask + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1)
            if self.add_eos_token and self.eos_token_id is not None:
                mask.append(1)
        return mask

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str, ...]:
        output_name = "vocab.json"
        if filename_prefix:
            output_name = f"{filename_prefix}-{output_name}"
        output_path = Path(save_directory) / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = {
            token: token_id
            for token, token_id in sorted(self.vocab.items(), key=lambda item: item[1])
        }
        output_path.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        outputs: tuple[str, ...] = (str(output_path),)

        # Persist the frequency/score signal alongside the vocabulary so that a
        # save/reload round trip keeps softmax scoring identical. vocab.json is
        # never modified by this.
        if self._token_scores:
            scores_name = TOKEN_SCORES_FILE
            if filename_prefix:
                scores_name = f"{filename_prefix}-{scores_name}"
            scores_path = Path(save_directory) / scores_name
            ordered_scores = {
                token: self._token_scores[token] for token in sorted(self._token_scores)
            }
            scores_path.write_text(
                json.dumps(ordered_scores, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs = (str(output_path), str(scores_path))

        return outputs
