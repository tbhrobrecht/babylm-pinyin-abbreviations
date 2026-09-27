"""Regression tests for raw-text offsets from the fast BPE tokenizer."""

from __future__ import annotations

import unittest
from pathlib import Path

from hf.tokenization_pinyin_code import EncodedMandarinTokenizerFast


RAW_HANZI = "已经很晚了"


class PinyinCodeFastOffsetTests(unittest.TestCase):
    def test_offsets_use_raw_coordinates_without_changing_ids(self) -> None:
        model = Path("tokenizers/babylm_zho_pinyin_spm.model")
        if not model.is_file():
            self.skipTest(f"missing test tokenizer: {model}")
        tokenizer = EncodedMandarinTokenizerFast(vocab_file=str(model))
        plain = tokenizer(RAW_HANZI, add_special_tokens=False)
        mapped = tokenizer(
            RAW_HANZI,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        self.assertEqual(mapped["input_ids"], plain["input_ids"])
        self.assertEqual(len(mapped["offset_mapping"]), len(plain["input_ids"]))
        self.assertTrue(all(0 <= start <= end <= len(RAW_HANZI)
                            for start, end in mapped["offset_mapping"]))
        self.assertEqual(max(end for _, end in mapped["offset_mapping"]), len(RAW_HANZI))


if __name__ == "__main__":
    unittest.main()
