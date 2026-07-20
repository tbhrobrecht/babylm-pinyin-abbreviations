"""Tests for greedy and softmax tokenization modes of the hybrid tokenizer."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from hf.tokenization_hybrid_pinyin_code import (
    ENCODED_WORD_RE,
    HybridPinyinCodeTokenizer,
    HybridTokenizerError,
    TokenMatch,
)
from train_hybrid_tokenizer import build_vocab, write_tokenizer_files

FIXED_SPECIALS = ["<pad>", "<unk>", "<s>", "</s>", "<mask>"]


class ControlledVocabMixin(unittest.TestCase):
    """Helpers to build tokenizers with a hand-crafted vocabulary."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self._counter = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_tokenizer(
        self,
        tokens: list[str],
        scores: dict[str, float] | None = None,
        **kwargs,
    ) -> HybridPinyinCodeTokenizer:
        vocab: dict[str, int] = {}
        for token in FIXED_SPECIALS + tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        self._counter += 1
        directory = self.root / f"controlled-{self._counter}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "vocab.json").write_text(
            json.dumps(vocab, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return HybridPinyinCodeTokenizer(
            vocab_file=str(directory / "vocab.json"),
            token_scores=scores,
            **kwargs,
        )

    def reconstruct(self, tokens: list[str]) -> list[str]:
        atoms: list[str] = []
        for token in tokens:
            if ENCODED_WORD_RE.fullmatch(token):
                atoms.extend(token[i : i + 2] for i in range(0, len(token), 2))
            else:
                atoms.append(token)
        return atoms


class GreedyModeTests(ControlledVocabMixin):
    def test_greedy_prefers_longest_full_word(self) -> None:
        tokenizer = self.make_tokenizer(
            ["H7", "W7", "L6", "H7W7", "W7L6", "H7W7L6"]
        )
        self.assertEqual(tokenizer.tokenization_mode, "greedy")
        self.assertEqual(tokenizer.tokenize("H7W7L6"), ["H7W7L6"])

    def test_greedy_falls_back_to_shorter_prefix(self) -> None:
        tokenizer = self.make_tokenizer(["H7", "W7", "L6", "H7W7", "W7L6"])
        self.assertEqual(tokenizer.tokenize("H7W7L6"), ["H7W7", "L6"])

    def test_greedy_atomic_only(self) -> None:
        tokenizer = self.make_tokenizer(["H7", "W7", "L6"])
        self.assertEqual(tokenizer.tokenize("H7W7L6"), ["H7", "W7", "L6"])

    def test_atomic_units_are_never_split(self) -> None:
        tokenizer = self.make_tokenizer(["Y6", "J3", "H7"])
        tokens = tokenizer.tokenize("Y6J3H7")
        self.assertEqual(tokens, ["Y6", "J3", "H7"])
        for token in tokens:
            self.assertEqual(len(token), 2)
            self.assertNotIn(token, {"Y", "6", "J", "3", "H", "7"})

    def test_matching_never_crosses_jieba_boundary(self) -> None:
        # "J3H7" is a valid vocabulary token but must never be produced because
        # J3 and H7 belong to different whitespace-separated words.
        tokenizer = self.make_tokenizer(
            ["Y6", "J3", "H7", "W7", "L6", "J3H7", "H7W7L6"]
        )
        tokens = tokenizer.tokenize("Y6J3 H7W7L6")
        self.assertNotIn("J3H7", tokens)
        self.assertEqual(tokens, ["Y6", "J3", "H7W7L6"])
        self.assertEqual(self.reconstruct(tokens), ["Y6", "J3", "H7", "W7", "L6"])

    def test_greedy_is_deterministic_and_seed_independent(self) -> None:
        tokens = ["H7", "W7", "L6", "H7W7", "W7L6"]
        first = self.make_tokenizer(tokens, sampling_seed=1)
        second = self.make_tokenizer(tokens, sampling_seed=999)
        text = "H7W7L6 H7W7 L6"
        outputs = {tuple(first.tokenize(text)) for _ in range(20)}
        self.assertEqual(len(outputs), 1)
        self.assertEqual(first.tokenize(text), second.tokenize(text))

    def test_get_valid_matches_returns_distinct_length_candidates(self) -> None:
        tokenizer = self.make_tokenizer(["H7", "W7", "L6", "H7W7", "H7W7L6"])
        atoms = tokenizer._atomize("H7W7L6")
        matches = tokenizer.get_valid_matches(atoms, 0)
        self.assertTrue(all(isinstance(match, TokenMatch) for match in matches))
        lengths = [match.atomic_length for match in matches]
        self.assertEqual(lengths, sorted(set(lengths)))
        self.assertEqual({match.token for match in matches}, {"H7", "H7W7", "H7W7L6"})


class SoftmaxModeTests(ControlledVocabMixin):
    def test_softmax_output_is_valid_and_complete(self) -> None:
        tokenizer = self.make_tokenizer(
            ["A1", "B2", "C3", "A1B2", "B2C3", "A1B2C3"],
            tokenization_mode="softmax",
            sampling_seed=7,
        )
        vocab = tokenizer.get_vocab()
        for _ in range(200):
            tokens = tokenizer.tokenize("A1B2C3")
            for token in tokens:
                self.assertIn(token, vocab)
            self.assertEqual(self.reconstruct(tokens), ["A1", "B2", "C3"])

    def test_softmax_reproducible_with_same_seed(self) -> None:
        tokens = ["A1", "B2", "A1B2"]
        inputs = ["A1B2"] * 50
        first = self.make_tokenizer(
            tokens, tokenization_mode="softmax", sampling_seed=42, sampling_temperature=2.0
        )
        second = self.make_tokenizer(
            tokens, tokenization_mode="softmax", sampling_seed=42, sampling_temperature=2.0
        )
        first_outputs = [first.tokenize(text) for text in inputs]
        second_outputs = [second.tokenize(text) for text in inputs]
        self.assertEqual(first_outputs, second_outputs)

    def test_softmax_different_seeds_differ_over_large_sample(self) -> None:
        tokens = ["A1", "B2", "A1B2"]
        inputs = ["A1B2"] * 100
        seed_a = self.make_tokenizer(
            tokens, tokenization_mode="softmax", sampling_seed=1, sampling_temperature=2.0
        )
        seed_b = self.make_tokenizer(
            tokens, tokenization_mode="softmax", sampling_seed=2, sampling_temperature=2.0
        )
        outputs_a = [tuple(seed_a.tokenize(text)) for text in inputs]
        outputs_b = [tuple(seed_b.tokenize(text)) for text in inputs]
        self.assertNotEqual(outputs_a, outputs_b)

    def test_temperature_controls_diversity(self) -> None:
        tokens = ["A1", "B2", "A1B2"]
        samples = 400

        low = self.make_tokenizer(
            tokens, tokenization_mode="softmax", sampling_temperature=0.2, sampling_seed=3
        )
        high = self.make_tokenizer(
            tokens, tokenization_mode="softmax", sampling_temperature=5.0, sampling_seed=3
        )
        low_whole = sum(low.tokenize("A1B2") == ["A1B2"] for _ in range(samples)) / samples
        high_whole = sum(high.tokenize("A1B2") == ["A1B2"] for _ in range(samples)) / samples

        # Low temperature should strongly favor the longest candidate; high
        # temperature should be closer to uniform (more diversity).
        self.assertGreater(low_whole, 0.9)
        self.assertGreater(low_whole, high_whole + 0.15)

    def test_near_greedy_limit(self) -> None:
        tokens = ["A1", "B2", "A1B2"]
        tokenizer = self.make_tokenizer(
            tokens,
            tokenization_mode="softmax",
            sampling_temperature=0.01,
            sampling_beta=5.0,
            sampling_seed=5,
        )
        whole = sum(tokenizer.tokenize("A1B2") == ["A1B2"] for _ in range(200)) / 200
        self.assertGreater(whole, 0.98)

    def test_frequency_scoring_prefers_high_frequency_with_alpha(self) -> None:
        tokens = ["A1", "B2", "A1B2"]
        # A1 (shorter) is far more frequent than A1B2 (longer). With beta=0 the
        # length term is removed so only frequency drives selection.
        scores = {"A1": 10.0, "A1B2": 1.0, "B2": 1.0}
        samples = 600

        def fraction_starts_with_atom(alpha: float) -> float:
            tokenizer = self.make_tokenizer(
                tokens,
                scores=scores,
                tokenization_mode="softmax",
                sampling_alpha=alpha,
                sampling_beta=0.0,
                sampling_seed=11,
            )
            return sum(tokenizer.tokenize("A1B2") == ["A1", "B2"] for _ in range(samples)) / samples

        low_alpha = fraction_starts_with_atom(0.1)
        high_alpha = fraction_starts_with_atom(1.0)
        self.assertGreater(high_alpha, low_alpha + 0.1)

    def test_length_only_scoring_without_frequency_metadata(self) -> None:
        tokenizer = self.make_tokenizer(
            ["A1", "B2", "A1B2"],
            tokenization_mode="softmax",
            sampling_temperature=0.1,
            sampling_beta=1.0,
            sampling_seed=13,
        )
        self.assertFalse(tokenizer._token_scores)
        whole = sum(tokenizer.tokenize("A1B2") == ["A1B2"] for _ in range(200)) / 200
        self.assertGreater(whole, 0.95)


class InvalidConfigurationTests(ControlledVocabMixin):
    def test_unsupported_mode(self) -> None:
        with self.assertRaises(HybridTokenizerError):
            self.make_tokenizer(["A1"], tokenization_mode="beam")

    def test_mode_names_are_normalized(self) -> None:
        tokenizer = self.make_tokenizer(["A1"], tokenization_mode="SOFTMAX")
        self.assertEqual(tokenizer.tokenization_mode, "softmax")

    def test_zero_temperature(self) -> None:
        with self.assertRaises(HybridTokenizerError):
            self.make_tokenizer(["A1"], sampling_temperature=0.0)

    def test_negative_temperature(self) -> None:
        with self.assertRaises(HybridTokenizerError):
            self.make_tokenizer(["A1"], sampling_temperature=-1.0)

    def test_non_finite_alpha_and_beta(self) -> None:
        with self.assertRaises(HybridTokenizerError):
            self.make_tokenizer(["A1"], sampling_alpha=float("nan"))
        with self.assertRaises(HybridTokenizerError):
            self.make_tokenizer(["A1"], sampling_beta=float("inf"))

    def test_malformed_atomic_input(self) -> None:
        tokenizer = self.make_tokenizer(["A1", "B2"])
        with self.assertRaises(ValueError):
            tokenizer.tokenize("A1B")

    def test_missing_atomic_fallback_raises_clear_error(self) -> None:
        tokenizer = self.make_tokenizer(["A1", "B2"])  # no C3 atom present
        with self.assertRaises(HybridTokenizerError) as context:
            tokenizer.tokenize("C3")
        message = str(context.exception)
        self.assertIn("C3", message)
        self.assertIn("atomic position", message)
        self.assertIn("greedy", message)


class SetModeTests(ControlledVocabMixin):
    def test_set_tokenization_mode_switch(self) -> None:
        tokenizer = self.make_tokenizer(["A1", "B2", "A1B2"])
        self.assertEqual(tokenizer.tokenize("A1B2"), ["A1B2"])
        tokenizer.set_tokenization_mode("softmax")
        self.assertEqual(tokenizer.tokenization_mode, "softmax")
        tokenizer.set_tokenization_mode("greedy")
        self.assertEqual(tokenizer.tokenize("A1B2"), ["A1B2"])

    def test_use_mode_context_manager_restores_mode(self) -> None:
        tokenizer = self.make_tokenizer(["A1", "B2", "A1B2"], tokenization_mode="softmax")
        with tokenizer.use_mode("greedy"):
            self.assertEqual(tokenizer.tokenization_mode, "greedy")
            self.assertEqual(tokenizer.tokenize("A1B2"), ["A1B2"])
        self.assertEqual(tokenizer.tokenization_mode, "softmax")


class TrainedTokenizerModeTests(unittest.TestCase):
    """Tests over a tokenizer built by train_hybrid_tokenizer."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self._counter = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build(self, corpus: str, vocab_size: int = 600) -> Path:
        corpus_path = self.root / f"corpus-{self._counter}.txt"
        corpus_path.write_text(corpus, encoding="utf-8")
        self._counter += 1
        output_dir = self.root / f"tokenizer-{self._counter}"
        args = argparse.Namespace(
            input=[corpus_path],
            output_dir=output_dir,
            vocab_size=vocab_size,
            min_word_frequency=1,
            atomic_only=False,
            initial_alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            digits="0123456789",
            permissive=False,
            max_invalid_examples=20,
        )
        vocab, metadata = build_vocab(args)
        write_tokenizer_files(output_dir, vocab, metadata)
        return output_dir

    def test_backward_compatible_default_is_greedy(self) -> None:
        output_dir = self.build("Y0J7 Y0J7 H2\n")
        # Simulate a pre-change tokenizer directory that predates the mode fields.
        config_path = output_dir / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        for key in (
            "tokenization_mode",
            "sampling_temperature",
            "sampling_alpha",
            "sampling_beta",
            "sampling_epsilon",
            "sampling_seed",
        ):
            config.pop(key, None)
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

        tokenizer = HybridPinyinCodeTokenizer.from_pretrained(output_dir)
        self.assertEqual(tokenizer.tokenization_mode, "greedy")
        # Golden output identical to the pre-change hybrid tokenizer.
        tokens = tokenizer.tokenize("Y0J7 H2 X4Q3")
        self.assertEqual(tokens, ["Y0J7", "H2", "X4", "Q3"])
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(ids, [tokenizer.vocab[token] for token in tokens])

    def test_save_and_load_preserves_mode_config_and_behavior(self) -> None:
        output_dir = self.build("Y0J7 Y0J7 H2 Y0J7\n")

        greedy = HybridPinyinCodeTokenizer.from_pretrained(output_dir)
        softmax = HybridPinyinCodeTokenizer.from_pretrained(
            output_dir,
            tokenization_mode="softmax",
            sampling_temperature=1.5,
            sampling_alpha=0.5,
            sampling_beta=2.0,
            sampling_epsilon=1e-6,
            sampling_seed=123,
        )

        greedy_dir = self.root / "saved-greedy"
        softmax_dir = self.root / "saved-softmax"
        greedy.save_pretrained(greedy_dir)
        softmax.save_pretrained(softmax_dir)

        reloaded_greedy = HybridPinyinCodeTokenizer.from_pretrained(greedy_dir)
        reloaded_softmax = HybridPinyinCodeTokenizer.from_pretrained(softmax_dir)

        self.assertEqual(reloaded_greedy.tokenization_mode, "greedy")
        self.assertEqual(reloaded_softmax.tokenization_mode, "softmax")
        self.assertEqual(reloaded_softmax.sampling_temperature, 1.5)
        self.assertEqual(reloaded_softmax.sampling_alpha, 0.5)
        self.assertEqual(reloaded_softmax.sampling_beta, 2.0)
        self.assertEqual(reloaded_softmax.sampling_epsilon, 1e-6)
        self.assertEqual(reloaded_softmax.sampling_seed, 123)

        self.assertEqual(greedy.get_vocab(), reloaded_greedy.get_vocab())
        self.assertEqual(softmax.get_vocab(), reloaded_softmax.get_vocab())
        self.assertEqual(
            greedy.tokenize("Y0J7 H2 X4Q3"),
            reloaded_greedy.tokenize("Y0J7 H2 X4Q3"),
        )

        # A freshly reloaded softmax tokenizer with the same seed reproduces the
        # same sampled sequence (scores restored from token_scores.json).
        reloaded_softmax_again = HybridPinyinCodeTokenizer.from_pretrained(softmax_dir)
        inputs = ["Y0J7"] * 30
        self.assertEqual(
            [reloaded_softmax.tokenize(text) for text in inputs],
            [reloaded_softmax_again.tokenize(text) for text in inputs],
        )

    def test_token_scores_loaded_from_metadata_for_softmax(self) -> None:
        output_dir = self.build("Y0J7 Y0J7 Y0J7 H2\n")
        tokenizer = HybridPinyinCodeTokenizer.from_pretrained(
            output_dir, tokenization_mode="softmax"
        )
        # Y0J7 was frequent enough to be a whole-word token with a stored score.
        self.assertIn("Y0J7", tokenizer._token_scores)
        self.assertGreater(tokenizer._token_scores["Y0J7"], 0.0)


if __name__ == "__main__":
    unittest.main()
