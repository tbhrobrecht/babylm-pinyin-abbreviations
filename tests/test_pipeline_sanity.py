from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import TensorDataset

from create_dataset import iter_chunks
from generate import prepare_prompt
from preprocessing.preprocess import process_text
from train_model import (
    JsonlTokenDataset,
    ModelConfig,
    PinyinCodeLanguageModel,
    split_dataset,
    validate_dataset_compatibility,
)


class PipelineSanityTests(unittest.TestCase):
    def write_jsonl_dataset(self, records: list[list[int]]) -> Path:
        path = Path(self.temp_dir.name) / "dataset.jsonl"
        lines = [json.dumps({"input_ids": record}) for record in records]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_process_text_normalizes_task_markers(self) -> None:
        text = "题干：A. yes 3"
        self.assertEqual(process_text(text), "<QUESTION> A . <YES> <NUM>")

    def test_prepare_prompt_processes_non_chinese_raw_text(self) -> None:
        prompt = prepare_prompt(
            "A. yes 3",
            raw_prompt=False,
            code_prompt=False,
            transliteration="pinyin-code",
            use_jieba=True,
        )
        self.assertEqual(prompt, "A . <YES> <NUM>")

    def test_prepare_prompt_preserves_pinyin_code_text(self) -> None:
        prompt = prepare_prompt(
            "W6M7 Y6",
            raw_prompt=False,
            code_prompt=False,
            transliteration="pinyin-code",
            use_jieba=True,
        )
        self.assertEqual(prompt, "W6M7 Y6")

    def test_iter_chunks_rejects_stride_larger_than_block_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "--stride"):
            list(iter_chunks([1, 2, 3], block_size=2, stride=3))

    def test_dataset_validation_rejects_vocab_mismatch(self) -> None:
        dataset = JsonlTokenDataset(self.write_jsonl_dataset([[0, 1, 4]]))
        config = ModelConfig(vocab_size=4, block_size=3)
        with self.assertRaisesRegex(ValueError, "vocab-size"):
            validate_dataset_compatibility(dataset, config)

    def test_dataset_validation_rejects_block_size_mismatch(self) -> None:
        dataset = JsonlTokenDataset(self.write_jsonl_dataset([[0, 1, 2, 3]]))
        config = ModelConfig(vocab_size=8, block_size=3)
        with self.assertRaisesRegex(ValueError, "block-size"):
            validate_dataset_compatibility(dataset, config)

    def test_split_dataset_rejects_too_few_examples(self) -> None:
        dataset = TensorDataset(torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, "At least two"):
            split_dataset(dataset, validation_fraction=0.5, seed=1337)

    def test_tiny_model_forward_has_finite_loss(self) -> None:
        model = PinyinCodeLanguageModel(
            ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
        )
        input_ids = torch.randint(0, 32, (2, 8))
        logits, loss = model(input_ids, labels=input_ids)
        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
