"""Tests for syllable-atomic BPE and its two boundary policies."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from create_dataset import load_tokenizer_processor
from hf.tokenization_atomic_bpe_pinyin_code import AtomicBPEPinyinCodeTokenizer
from train_atomic_bpe_tokenizer import build_atomic_bpe, write_tokenizer_files
from train_hybrid_tokenizer import FIXED_SPECIAL_TOKENS, atomic_tokens, ordered_vocab, write_json
from train_sentencepiece import SPECIAL_TOKENS


class AtomicBPETokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus.txt"
        # A0 and C2 are separate one-syllable Jieba words in a contiguous run.
        # A01B is a two-syllable word and exercises legal within-word merging.
        self.corpus.write_text(("A0C2 A01B , hello\n" * 200), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def train(self, policy: str) -> tuple[Path, AtomicBPEPinyinCodeTokenizer]:
        output = self.root / policy
        args = argparse.Namespace(
            input=[self.corpus],
            output_dir=output,
            boundary_policy=policy,
            vocab_size=256,
            min_frequency=2,
            min_preserved_frequency=2,
            initial_alphabet="ABC",
            digits="012",
            permissive=False,
            max_invalid_examples=10,
        )
        vocab, metadata, merges = build_atomic_bpe(args)
        write_tokenizer_files(output, vocab, metadata, merges)
        return output, AtomicBPEPinyinCodeTokenizer.from_pretrained(output)

    def test_all_configured_atoms_are_guaranteed_fallbacks(self) -> None:
        _output, tokenizer = self.train("within_word")
        expected = set(atomic_tokens("ABC", "012"))
        self.assertTrue(expected.issubset(tokenizer.get_vocab()))

    def test_within_word_never_merges_across_recovered_words(self) -> None:
        _output, tokenizer = self.train("within_word")
        self.assertEqual(tokenizer.tokenize("A0C2"), ["A0", "C2"])
        self.assertIn("A01B", tokenizer.tokenize("A01B"))

    def test_cross_word_can_merge_but_stops_at_surface_boundaries(self) -> None:
        _output, tokenizer = self.train("cross_word")
        self.assertEqual(tokenizer.tokenize("A0C2"), ["A0C2"])
        tokens = tokenizer.tokenize("A0 , C2")
        self.assertEqual(tokens, ["A0", ",", "C2"])

    def test_every_encoded_piece_is_even_and_round_trips(self) -> None:
        _output, tokenizer = self.train("cross_word")
        text = "A0C2A01B"
        tokens = tokenizer.tokenize(text)
        self.assertTrue(all(len(token) % 2 == 0 for token in tokens))
        ids = tokenizer.convert_tokens_to_ids(tokens)
        self.assertEqual(tokenizer.decode(ids), text)

    def test_dataset_loader_detects_atomic_bpe_directory(self) -> None:
        output, _tokenizer = self.train("within_word")
        processor = load_tokenizer_processor(output)
        self.assertIsInstance(processor.tokenizer, AtomicBPEPinyinCodeTokenizer)
        self.assertEqual(
            processor.encode("A0C2", out_type=str),
            ["A0", "C2"],
        )

    def test_merge_artifact_records_policy_and_only_even_pieces(self) -> None:
        output, _tokenizer = self.train("cross_word")
        payload = json.loads((output / "atomic_bpe_merges.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["boundary_policy"], "cross_word")
        for left, right in payload["merges"]:
            self.assertEqual(len(left) % 2, 0)
            self.assertEqual(len(right) % 2, 0)

    def test_ranked_bpe_keeps_independent_pending_merges(self) -> None:
        output = self.root / "manual"
        output.mkdir()
        atoms = atomic_tokens("ACEG", "0246")
        merges = [
            ["A0", "C2"],
            ["E4", "G6"],
            ["A0C2", "E4G6"],
        ]
        vocab = ordered_vocab(
            [
                *FIXED_SPECIAL_TOKENS,
                *SPECIAL_TOKENS,
                *atoms,
                "A0C2",
                "E4G6",
                "A0C2E4G6",
            ]
        )
        write_json(output / "vocab.json", vocab)
        write_json(
            output / "atomic_bpe_merges.json",
            {
                "format": "babylm-pinyin-code-atomic-bpe-v1",
                "boundary_policy": "cross_word",
                "merges": merges,
            },
        )
        tokenizer = AtomicBPEPinyinCodeTokenizer(
            vocab_file=str(output / "vocab.json"),
            merges_file=str(output / "atomic_bpe_merges.json"),
            atomic_alphabet="ACEG",
            atomic_digits="0246",
        )
        self.assertEqual(tokenizer.tokenize("A0C2E4G6"), ["A0C2E4G6"])

    def test_auto_tokenizer_remote_code_round_trip(self) -> None:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            self.skipTest("transformers is not installed")
        output, _tokenizer = self.train("within_word")
        modules_cache = self.root / "modules-cache"
        with patch(
            "transformers.dynamic_module_utils.HF_MODULES_CACHE",
            str(modules_cache),
        ):
            loaded = AutoTokenizer.from_pretrained(output, trust_remote_code=True)
        self.assertEqual(type(loaded).__name__, "AtomicBPEPinyinCodeTokenizer")
        self.assertEqual(loaded.tokenize("A0C2"), ["A0", "C2"])


if __name__ == "__main__":
    unittest.main()
