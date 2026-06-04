from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from create_dataset import iter_chunks, write_dataset
from generate import prepare_prompt
from preprocessing.preprocess import process_text
from train_sentencepiece import train_tokenizer
from train_model import (
    BinaryTokenDataset,
    JsonlTokenDataset,
    ModelConfig,
    PinyinCodeLanguageModel,
    load_token_dataset,
    learning_rate_for_step,
    split_dataset,
    train,
    validate_dataset_compatibility,
)

try:
    from hf.configuration_pinyin_code import PinyinCodeConfig
    from hf.modeling_pinyin_code import PinyinCodeForCausalLM
except ModuleNotFoundError:
    PinyinCodeConfig = None
    PinyinCodeForCausalLM = None


class PipelineSanityTests(unittest.TestCase):
    def write_jsonl_dataset(self, records: list[list[int]], name: str = "dataset.jsonl") -> Path:
        path = Path(self.temp_dir.name) / name
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

    def test_write_dataset_can_create_document_level_validation_split(self) -> None:
        input_path = Path(self.temp_dir.name) / "processed.txt"
        input_path.write_text(
            "\n".join(
                [
                    "1 2 3 4",
                    "5 6 7 8",
                    "9 10 11 12",
                    "13 14 15 16",
                    "17 18 19 20",
                    "21 22 23 24",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeProcessor:
            def __init__(self, model_file: str) -> None:
                self.model_file = model_file

            def eos_id(self) -> int:
                return 0

            def encode(self, text: str, out_type=int) -> list[int]:
                return [int(token) for token in text.split()]

        fake_sentencepiece = SimpleNamespace(SentencePieceProcessor=FakeProcessor)
        args = SimpleNamespace(
            input=[input_path],
            tokenizer=Path("toy.model"),
            output=Path(self.temp_dir.name) / "train.jsonl",
            validation_output=Path(self.temp_dir.name) / "valid.jsonl",
            validation_fraction=0.5,
            format="jsonl",
            block_size=4,
            stride=4,
            include_labels=False,
            seed=1337,
        )

        with patch("create_dataset.require_sentencepiece", return_value=fake_sentencepiece):
            stats = write_dataset(args)

        self.assertGreater(stats.train_examples, 0)
        self.assertGreater(stats.validation_examples, 0)
        self.assertTrue(args.output.exists())
        self.assertTrue(args.validation_output.exists())

    def test_write_dataset_can_create_binary_chunks(self) -> None:
        input_path = Path(self.temp_dir.name) / "processed-bin.txt"
        input_path.write_text("1 2 3 4\n5 6 7 8\n", encoding="utf-8")

        class FakeProcessor:
            def __init__(self, model_file: str) -> None:
                self.model_file = model_file

            def eos_id(self) -> int:
                return 0

            def encode(self, text: str, out_type=int) -> list[int]:
                return [int(token) for token in text.split()]

        fake_sentencepiece = SimpleNamespace(SentencePieceProcessor=FakeProcessor)
        output_path = Path(self.temp_dir.name) / "chunks.bin"
        args = SimpleNamespace(
            input=[input_path],
            tokenizer=Path("toy.model"),
            output=output_path,
            validation_output=None,
            validation_fraction=None,
            format="bin",
            block_size=4,
            stride=4,
            include_labels=False,
            seed=1337,
        )

        with patch("create_dataset.require_sentencepiece", return_value=fake_sentencepiece):
            stats = write_dataset(args)

        self.assertEqual(stats.train_examples, 2)
        dataset = BinaryTokenDataset(output_path)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0].tolist(), [1, 2, 3, 4])
        self.assertIsInstance(load_token_dataset(output_path), BinaryTokenDataset)

    def test_sentencepiece_training_allows_cross_token_bpe_merges(self) -> None:
        captured = {}

        class FakeTrainer:
            @staticmethod
            def train(**kwargs):
                captured.update(kwargs)

        fake_sentencepiece = SimpleNamespace(SentencePieceTrainer=FakeTrainer)
        args = SimpleNamespace(
            output_dir=Path(self.temp_dir.name),
            model_name="toy",
            input=[Path("toy.txt")],
            vocab_size=32,
            model_type="bpe",
            character_coverage=1.0,
            input_sentence_size=0,
            max_sentence_length=1024,
            shuffle_input_sentence=True,
            train_extremely_large_corpus=False,
            hard_vocab_limit=False,
        )

        with (
            patch("train_sentencepiece.require_sentencepiece", return_value=fake_sentencepiece),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_tokenizer(args)

        self.assertFalse(captured["split_by_whitespace"])
        self.assertFalse(captured["split_by_number"])

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

    def test_learning_rate_schedule_warmup_and_cosine_decay(self) -> None:
        self.assertAlmostEqual(
            learning_rate_for_step(
                step=1,
                total_steps=6,
                base_lr=1.0,
                min_lr=0.1,
                warmup_steps=2,
                schedule="cosine",
            ),
            0.5,
        )
        self.assertAlmostEqual(
            learning_rate_for_step(
                step=2,
                total_steps=6,
                base_lr=1.0,
                min_lr=0.1,
                warmup_steps=2,
                schedule="cosine",
            ),
            1.0,
        )
        self.assertAlmostEqual(
            learning_rate_for_step(
                step=6,
                total_steps=6,
                base_lr=1.0,
                min_lr=0.1,
                warmup_steps=2,
                schedule="cosine",
            ),
            0.1,
        )

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

    def test_tiny_train_loop_writes_compact_checkpoint(self) -> None:
        dataset_path = self.write_jsonl_dataset([[0, 1, 2, 3], [1, 2, 3, 4]])
        output_dir = Path(self.temp_dir.name) / "model"
        args = SimpleNamespace(
            dataset=dataset_path,
            validation_dataset=None,
            output_dir=output_dir,
            vocab_size=8,
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            gradient_accumulation_steps=2,
            lr_schedule="cosine",
            warmup_steps=1,
            min_learning_rate=1e-5,
            weight_decay=0.0,
            grad_clip=1.0,
            validation_fraction=0.5,
            log_every=100,
            num_workers=0,
            num_threads=None,
            seed=1337,
            device="cpu",
            resume=None,
            amp=True,
            amp_dtype="float16",
            tf32=True,
            fused_adamw=True,
            compile=False,
            save_optimizer=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            train(args)

        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", checkpoint)
        self.assertNotIn("optimizer_state_dict", checkpoint)

    def test_train_loop_can_resume_from_compact_checkpoint(self) -> None:
        dataset_path = self.write_jsonl_dataset(
            [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]
        )
        validation_path = self.write_jsonl_dataset(
            [[0, 1, 2, 3], [1, 2, 3, 4]],
            name="validation.jsonl",
        )
        output_dir = Path(self.temp_dir.name) / "resume-model"
        args = SimpleNamespace(
            dataset=dataset_path,
            validation_dataset=validation_path,
            output_dir=output_dir,
            vocab_size=8,
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            gradient_accumulation_steps=2,
            lr_schedule="cosine",
            warmup_steps=1,
            min_learning_rate=1e-5,
            weight_decay=0.0,
            grad_clip=1.0,
            validation_fraction=0.5,
            log_every=100,
            num_workers=0,
            num_threads=None,
            seed=1337,
            device="cpu",
            resume=None,
            amp=True,
            amp_dtype="float16",
            tf32=True,
            fused_adamw=True,
            compile=False,
            save_optimizer=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            train(args)
        args.resume = output_dir / "last.pt"
        args.epochs = 2
        with contextlib.redirect_stdout(io.StringIO()):
            train(args)

        checkpoint = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(checkpoint["global_step"], 4)

    def test_hf_model_derives_position_ids_from_attention_mask(self) -> None:
        if PinyinCodeConfig is None or PinyinCodeForCausalLM is None:
            self.skipTest("transformers is not installed")

        config = PinyinCodeConfig(vocab_size=16, block_size=8, n_layer=1, n_head=1, n_embd=8)
        model = PinyinCodeForCausalLM(config)
        attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
        prepared = model.prepare_inputs_for_generation(
            torch.tensor([[0, 0, 4, 5, 6]]),
            attention_mask=attention_mask,
        )
        self.assertEqual(prepared["position_ids"].tolist(), [[0, 0, 0, 1, 2]])


if __name__ == "__main__":
    unittest.main()
