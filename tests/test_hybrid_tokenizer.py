from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from hf.tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer
from train_hybrid_tokenizer import build_vocab, write_tokenizer_files

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    AutoTokenizer = None


class HybridTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.build_count = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build_tokenizer(
        self,
        corpus: str,
        vocab_size: int = 1200,
        min_word_frequency: int = 1,
    ) -> tuple[HybridPinyinCodeTokenizer, Path]:
        corpus_path = self.root / "corpus.txt"
        corpus_path.write_text(corpus, encoding="utf-8")
        self.build_count += 1
        output_dir = self.root / f"tokenizer-{self.build_count}"
        args = argparse.Namespace(
            input=[corpus_path],
            output_dir=output_dir,
            vocab_size=vocab_size,
            min_word_frequency=min_word_frequency,
            atomic_only=False,
            initial_alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            digits="0123456789",
            permissive=False,
            max_invalid_examples=20,
        )
        vocab, metadata = build_vocab(args)
        write_tokenizer_files(output_dir, vocab, metadata)
        return HybridPinyinCodeTokenizer.from_pretrained(output_dir), output_dir

    def test_whole_word_lookup(self) -> None:
        tokenizer, _ = self.build_tokenizer("Y07JY07J\n")

        self.assertEqual(tokenizer.tokenize("Y07J"), ["Y07J"])

    def test_atomic_fallback(self) -> None:
        tokenizer, _ = self.build_tokenizer("Y07JY07J\n")

        self.assertEqual(tokenizer.tokenize("X43Q"), ["X4", "3Q"])

    def test_mixed_sentence(self) -> None:
        tokenizer, _ = self.build_tokenizer("Y07JY07JH2\n")

        self.assertEqual(
            tokenizer.tokenize("Y07JH2X43Q"),
            ["Y07J", "H2", "X4", "3Q"],
        )

    def test_word_boundaries_come_from_the_encoding(self) -> None:
        tokenizer, _ = self.build_tokenizer("Y07JY07JH2\n")

        # The whitespace-free run splits into the same words as the spaced form.
        self.assertEqual(
            tokenizer.tokenize("Y07JH2X43Q"),
            tokenizer.tokenize("Y07J H2 X43Q"),
        )

    def test_no_whitespace_tokens(self) -> None:
        tokenizer, _ = self.build_tokenizer("Y07JY07JH2\n")

        tokens = tokenizer.tokenize("Y07J   H2\nX43Q")
        self.assertNotIn(" ", tokens)
        self.assertNotIn(" ", tokenizer.get_vocab())
        self.assertNotIn("\n", tokenizer.get_vocab())

    def test_special_markers(self) -> None:
        tokenizer, _ = self.build_tokenizer("<QUESTION> Y07JY07J\n")

        self.assertEqual(tokenizer.tokenize("<QUESTION> Y07J"), ["<QUESTION>", "Y07J"])

    def test_malformed_input_rejects_or_maps_to_unk(self) -> None:
        tokenizer, output_dir = self.build_tokenizer("Y07JY07J\n")

        for value in ["Y0J", "Y00", "123", "ni3"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    tokenizer.tokenize(value)

        permissive = HybridPinyinCodeTokenizer(
            vocab_file=str(output_dir / "vocab.json"),
            strict_validation=False,
        )
        self.assertEqual(permissive.tokenize("ni3"), ["<unk>"])

    def test_save_and_reload_preserves_ids_and_tokenization(self) -> None:
        tokenizer, output_dir = self.build_tokenizer("<QUESTION> Y07JY07JH2\n")
        reloaded = HybridPinyinCodeTokenizer.from_pretrained(output_dir)

        self.assertEqual(tokenizer.get_vocab(), reloaded.get_vocab())
        self.assertEqual(reloaded.pad_token_id, 0)
        self.assertEqual(reloaded.unk_token_id, 1)
        self.assertEqual(reloaded.bos_token_id, 2)
        self.assertEqual(reloaded.eos_token_id, 3)
        self.assertEqual(reloaded.mask_token_id, 4)
        self.assertEqual(tokenizer.tokenize("Y07JH2X43Q"), reloaded.tokenize("Y07JH2X43Q"))

    def test_auto_tokenizer_loads_with_remote_code(self) -> None:
        if AutoTokenizer is None:
            self.skipTest("transformers is not installed")
        _, output_dir = self.build_tokenizer("Y07JY07JH2\n")

        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

        self.assertEqual(tokenizer.tokenize("Y07JH2X43Q"), ["Y07J", "H2", "X4", "3Q"])

    def test_deterministic_vocabulary_ids(self) -> None:
        corpus = "Y07JH2X43Q\nY07JX43Q\n"
        first, first_dir = self.build_tokenizer(corpus)
        second, second_dir = self.build_tokenizer(corpus)

        self.assertEqual(first.get_vocab(), second.get_vocab())
        first_vocab = json.loads((first_dir / "vocab.json").read_text(encoding="utf-8"))
        second_vocab = json.loads((second_dir / "vocab.json").read_text(encoding="utf-8"))
        self.assertEqual(first_vocab, second_vocab)

    def test_decode_default_and_readable_modes(self) -> None:
        tokenizer, _ = self.build_tokenizer("Y07JY07JH2\n")
        ids = tokenizer.encode("Y07JX43Q", add_special_tokens=False)

        self.assertEqual(tokenizer.decode(ids), "Y07JX43Q")
        self.assertEqual(tokenizer.decode(ids, readable=True), "Y07J X4 3Q")

    def test_decode_keeps_markers_spaced_and_encoded_words_joined(self) -> None:
        tokenizer, _ = self.build_tokenizer("<QUESTION> Y07JY07JH2\n")
        ids = tokenizer.encode("<QUESTION> Y07JH2", add_special_tokens=False)

        self.assertEqual(tokenizer.decode(ids), "<QUESTION> Y07JH2")


if __name__ == "__main__":
    unittest.main()
