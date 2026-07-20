"""Raw-Hanzi compatibility tests for the hybrid pinyin-code tokenizer.

These tests verify that ``HybridPinyinCodeTokenizer`` accepts both raw Hanzi
(e.g. ``"已经很晚了"``) and already-preprocessed pinyin-code text
(e.g. ``"Y6J3 H7W7 L6"``) while preserving the greedy/softmax tokenization
modes. They also confirm the behavior survives a ``trust_remote_code`` reload
from an exported tokenizer directory.
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from hf.tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer
from train_hybrid_tokenizer import build_vocab, write_tokenizer_files

try:
    from transformers import AutoTokenizer
except ModuleNotFoundError:
    AutoTokenizer = None

RAW_HANZI = "已经很晚了"
PREPROCESSED = "Y6J3 H7W7 L6"


def _preprocessing_available() -> bool:
    """Return True when raw Hanzi can be converted to pinyin-code."""
    try:
        import jieba  # noqa: F401
        import pypinyin  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


class HybridRawTextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.build_count = 0

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def build_dir(self, corpus: str, vocab_size: int = 600) -> Path:
        corpus_path = self.root / f"corpus-{self.build_count}.txt"
        corpus_path.write_text(corpus, encoding="utf-8")
        self.build_count += 1
        output_dir = self.root / f"tokenizer-{self.build_count}"
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

    def build_tokenizer(self, corpus: str, **kwargs) -> HybridPinyinCodeTokenizer:
        output_dir = self.build_dir(corpus)
        return HybridPinyinCodeTokenizer.from_pretrained(output_dir, **kwargs)

    def encode_ids(self, tokenizer, text: str) -> list[int]:
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    # -- Greedy raw vs preprocessed ------------------------------------------
    def test_greedy_accepts_raw_hanzi(self) -> None:
        if not _preprocessing_available():
            self.skipTest("jieba/pypinyin not installed")
        tokenizer = self.build_tokenizer(f"{PREPROCESSED}\n{PREPROCESSED}\n")
        self.assertEqual(tokenizer.tokenization_mode, "greedy")
        ids = self.encode_ids(tokenizer, RAW_HANZI)
        self.assertTrue(ids)
        self.assertNotIn(tokenizer.unk_token_id, ids)

    def test_greedy_accepts_preprocessed_pinyin_code(self) -> None:
        tokenizer = self.build_tokenizer(f"{PREPROCESSED}\n{PREPROCESSED}\n")
        ids = self.encode_ids(tokenizer, PREPROCESSED)
        self.assertTrue(ids)
        self.assertNotIn(tokenizer.unk_token_id, ids)

    def test_raw_and_preprocessed_match_in_greedy(self) -> None:
        if not _preprocessing_available():
            self.skipTest("jieba/pypinyin not installed")
        tokenizer = self.build_tokenizer(f"{PREPROCESSED}\n{PREPROCESSED}\n")
        raw_ids = self.encode_ids(tokenizer, RAW_HANZI)
        pre_ids = self.encode_ids(tokenizer, PREPROCESSED)
        self.assertEqual(raw_ids, pre_ids)

    def test_encode_and_call_agree_on_raw_hanzi(self) -> None:
        if not _preprocessing_available():
            self.skipTest("jieba/pypinyin not installed")
        tokenizer = self.build_tokenizer(f"{PREPROCESSED}\n{PREPROCESSED}\n")
        call_ids = tokenizer(RAW_HANZI, add_special_tokens=False)["input_ids"]
        encode_ids = tokenizer.encode(RAW_HANZI, add_special_tokens=False)
        self.assertEqual(call_ids, encode_ids)

    # -- Softmax still works, seed/reseed ------------------------------------
    def test_softmax_accepts_raw_hanzi_and_is_reproducible(self) -> None:
        if not _preprocessing_available():
            self.skipTest("jieba/pypinyin not installed")
        output_dir = self.build_dir(f"{PREPROCESSED}\n{PREPROCESSED}\n{PREPROCESSED}\n")
        first = HybridPinyinCodeTokenizer.from_pretrained(
            output_dir, tokenization_mode="softmax", sampling_seed=17
        )
        second = HybridPinyinCodeTokenizer.from_pretrained(
            output_dir, tokenization_mode="softmax", sampling_seed=17
        )
        self.assertEqual(first.tokenization_mode, "softmax")
        first_runs = [self.encode_ids(first, RAW_HANZI) for _ in range(20)]
        second_runs = [self.encode_ids(second, RAW_HANZI) for _ in range(20)]
        self.assertEqual(first_runs, second_runs)
        for ids in first_runs:
            self.assertTrue(ids)
            self.assertNotIn(first.unk_token_id, ids)

    def test_softmax_reseed_reproduces_stream(self) -> None:
        tokenizer = self.build_tokenizer(
            f"{PREPROCESSED}\n{PREPROCESSED}\n{PREPROCESSED}\n",
            tokenization_mode="softmax",
            sampling_seed=3,
            sampling_temperature=2.0,
        )
        tokenizer.reseed(99)
        first = [self.encode_ids(tokenizer, PREPROCESSED) for _ in range(30)]
        tokenizer.reseed(99)
        second = [self.encode_ids(tokenizer, PREPROCESSED) for _ in range(30)]
        self.assertEqual(first, second)

    def test_use_mode_forces_greedy_determinism_on_raw_hanzi(self) -> None:
        if not _preprocessing_available():
            self.skipTest("jieba/pypinyin not installed")
        tokenizer = self.build_tokenizer(
            f"{PREPROCESSED}\n{PREPROCESSED}\n",
            tokenization_mode="softmax",
            sampling_seed=5,
        )
        with tokenizer.use_mode("greedy"):
            greedy_raw = self.encode_ids(tokenizer, RAW_HANZI)
            greedy_pre = self.encode_ids(tokenizer, PREPROCESSED)
        self.assertEqual(greedy_raw, greedy_pre)
        self.assertEqual(tokenizer.tokenization_mode, "softmax")

    def test_config_round_trips_preprocessing_fields(self) -> None:
        output_dir = self.build_dir(f"{PREPROCESSED}\n")
        tokenizer = HybridPinyinCodeTokenizer.from_pretrained(output_dir)
        self.assertEqual(tokenizer.transliteration, "pinyin-code")
        self.assertTrue(tokenizer.use_jieba)

        saved_dir = self.root / "saved-config"
        tokenizer.save_pretrained(saved_dir)
        reloaded = HybridPinyinCodeTokenizer.from_pretrained(saved_dir)
        self.assertEqual(reloaded.transliteration, "pinyin-code")
        self.assertEqual(reloaded.use_jieba, tokenizer.use_jieba)


class HybridRawTextRemoteCodeTests(unittest.TestCase):
    """Reload the exported tokenizer folder through AutoTokenizer."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def export_dir(self) -> Path:
        corpus_path = self.root / "corpus.txt"
        corpus_path.write_text(f"{PREPROCESSED}\n{PREPROCESSED}\n", encoding="utf-8")
        output_dir = self.root / "exported"
        args = argparse.Namespace(
            input=[corpus_path],
            output_dir=output_dir,
            vocab_size=600,
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

    def test_auto_tokenizer_reload_handles_raw_and_preprocessed(self) -> None:
        if AutoTokenizer is None:
            self.skipTest("transformers is not installed")
        output_dir = self.export_dir()

        # The export must ship both remote-code modules for the relative import
        # ``from .tokenization_pinyin_code import PinyinCodeTokenizer`` to resolve.
        self.assertTrue((output_dir / "tokenization_hybrid_pinyin_code.py").exists())
        self.assertTrue((output_dir / "tokenization_pinyin_code.py").exists())

        reference = HybridPinyinCodeTokenizer.from_pretrained(output_dir)
        reference_pre = reference("Y6J3 H7W7 L6", add_special_tokens=False)["input_ids"]

        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)

        # Preprocessed pinyin-code still works after HF reload.
        reload_pre = tokenizer("Y6J3 H7W7 L6", add_special_tokens=False)["input_ids"]
        self.assertEqual(reload_pre, reference_pre)

        if _preprocessing_available():
            # Raw Hanzi works after HF reload and matches preprocessed ids.
            reload_raw = tokenizer(RAW_HANZI, add_special_tokens=False)["input_ids"]
            self.assertEqual(reload_raw, reference_pre)


if __name__ == "__main__":
    unittest.main()
