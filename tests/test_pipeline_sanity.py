from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from create_dataset import iter_chunks, write_dataset
from generate import prepare_prompt
from preprocessing.preprocess import hanzi_to_encoded, process_text
from train_hybrid_tokenizer import build_vocab, write_tokenizer_files
from train_sentencepiece import train_tokenizer
from train_model import (
    BinaryTokenDataset,
    JsonlTokenDataset,
    ModelConfig,
    PinyinCodeLanguageModel,
    build_model,
    mask_padding_labels,
    model_config_from_mapping,
    load_token_dataset,
    learning_rate_for_step,
    output_logits,
    output_loss,
    split_dataset,
    train,
    validate_dataset_compatibility,
)
from dpo_utils import DPODataCollator, dpo_loss_from_logps, sequence_log_probs

try:
    from transformers import Qwen2ForCausalLM
except ImportError:
    Qwen2ForCausalLM = None

try:
    from hf.configuration_pinyin_code import PinyinCodeConfig
    from hf.modeling_pinyin_code import PinyinCodeForCausalLM, PinyinCodeModel
    from hf.tokenization_pinyin_code import EncodedMandarinTokenizer
except ModuleNotFoundError:
    EncodedMandarinTokenizer = None
    PinyinCodeConfig = None
    PinyinCodeForCausalLM = None
    PinyinCodeModel = None


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

    def test_process_text_preserves_full_corpus_surface_tokens(self) -> None:
        text = (
            "hello iPhone12 5G SM-T29 Café mā ɡe 《》~ ^_^ "
            "1280*800 http://example.com \ue5e5\u200b"
        )

        tokens = process_text(text).split()

        self.assertEqual(
            tokens,
            [
                "hello",
                "iphone12",
                "5g",
                "sm-t29",
                "café",
                "mā",
                "ɡe",
                "《",
                "》",
                "~",
                "^",
                "_",
                "^",
                "<NUM>",
                "*",
                "<NUM>",
                "<URL>",
            ],
        )

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

    def test_encoded_mandarin_tokenizer_wraps_hanzi_input(self) -> None:
        if EncodedMandarinTokenizer is None:
            self.skipTest("transformers or sentencepiece is not installed")
        model_path = Path("hf_pinyin_code_model")
        if not (model_path / "tokenizer.model").exists():
            self.skipTest("exported tokenizer model is not available")

        tokenizer = EncodedMandarinTokenizer.from_pretrained(model_path)
        encoded_text = hanzi_to_encoded("已经很晚了")
        hanzi_ids = tokenizer.encode("已经很晚了", add_special_tokens=False)
        direct_ids = tokenizer.sp_model.encode(encoded_text, out_type=int)

        self.assertEqual(hanzi_ids, direct_ids)
        self.assertEqual(
            tokenizer(["已经很晚了"], add_special_tokens=False)["input_ids"][0],
            direct_ids,
        )

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
        self.assertIn("<URL>", captured["user_defined_symbols"].split(","))

    def test_sentencepiece_training_splits_long_bpe_lines(self) -> None:
        input_path = Path(self.temp_dir.name) / "long.txt"
        input_path.write_text("aa bb cc dd ee ff\nshort\n", encoding="utf-8")
        captured = {}

        class FakeTrainer:
            @staticmethod
            def train(**kwargs):
                captured["input"] = kwargs["input"]
                captured["lines"] = [
                    line
                    for path in kwargs["input"]
                    for line in Path(path).read_text(encoding="utf-8").splitlines()
                ]

        fake_sentencepiece = SimpleNamespace(SentencePieceTrainer=FakeTrainer)
        args = SimpleNamespace(
            output_dir=Path(self.temp_dir.name),
            model_name="toy",
            input=[input_path],
            vocab_size=32,
            model_type="bpe",
            character_coverage=1.0,
            input_sentence_size=0,
            max_sentence_length=100,
            shuffle_input_sentence=True,
            train_extremely_large_corpus=False,
            hard_vocab_limit=False,
            split_long_lines=True,
            split_max_chars=10,
        )

        with (
            patch("train_sentencepiece.require_sentencepiece", return_value=fake_sentencepiece),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_tokenizer(args)

        self.assertNotEqual(captured["input"], [str(input_path)])
        self.assertTrue(all(len(line) <= 10 for line in captured["lines"]))
        self.assertEqual(captured["lines"], ["aa bb cc", "dd ee ff", "short"])

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

    def test_model_factory_defaults_to_gpt2(self) -> None:
        model = build_model(ModelConfig(vocab_size=16, block_size=8, n_layer=1, n_head=1, n_embd=8))
        self.assertIsInstance(model, PinyinCodeLanguageModel)

        explicit = build_model(
            ModelConfig(
                architecture="gpt2",
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=1,
                n_embd=8,
            )
        )
        self.assertIsInstance(explicit, PinyinCodeLanguageModel)

        if Qwen2ForCausalLM is None:
            self.skipTest("transformers Qwen2 support is not installed")
        qwen = build_model(
            ModelConfig(
                architecture="qwen2",
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                num_key_value_heads=1,
                intermediate_size=32,
            )
        )
        self.assertIsInstance(qwen, Qwen2ForCausalLM)
        with self.assertRaisesRegex(ValueError, "Unsupported model architecture"):
            build_model(ModelConfig(architecture="llama"))

    def test_qwen2_tiny_forward_has_finite_loss_and_backward(self) -> None:
        if Qwen2ForCausalLM is None:
            self.skipTest("transformers Qwen2 support is not installed")

        model = build_model(
            ModelConfig(
                architecture="qwen2",
                vocab_size=32,
                block_size=16,
                n_layer=2,
                n_head=4,
                n_embd=64,
                num_key_value_heads=2,
                intermediate_size=176,
                attention_dropout=0.0,
            )
        )
        input_ids = torch.randint(0, 32, (2, 8))
        outputs = model(input_ids=input_ids, labels=input_ids)
        logits = output_logits(outputs)
        loss = output_loss(outputs)

        self.assertEqual(tuple(logits.shape), (2, 8, 32))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_factory_rejects_invalid_qwen2_attention_shape(self) -> None:
        if Qwen2ForCausalLM is None:
            self.skipTest("transformers Qwen2 support is not installed")
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            build_model(
                ModelConfig(
                    architecture="qwen2",
                    vocab_size=16,
                    block_size=8,
                    n_layer=1,
                    n_head=3,
                    n_embd=16,
                    num_key_value_heads=1,
                )
            )
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            build_model(
                ModelConfig(
                    architecture="qwen2",
                    vocab_size=16,
                    block_size=8,
                    n_layer=1,
                    n_head=4,
                    n_embd=16,
                    num_key_value_heads=3,
                )
            )

    def test_model_embeddings_match_tokenizer_length(self) -> None:
        class FakeTokenizer:
            def __len__(self) -> int:
                return 21

        configs = [
            ModelConfig(
                architecture="gpt2",
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=1,
                n_embd=8,
            ),
            ModelConfig(
                architecture="qwen2",
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                num_key_value_heads=1,
                intermediate_size=32,
            ),
        ]
        for config in configs:
            if config.architecture == "qwen2" and Qwen2ForCausalLM is None:
                continue
            model = build_model(config, tokenizer=FakeTokenizer())
            self.assertEqual(model.get_input_embeddings().num_embeddings, 21)
            self.assertEqual(model.get_output_embeddings().out_features, 21)

    def test_padding_labels_are_masked_for_loss(self) -> None:
        labels = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        masked = mask_padding_labels(labels, attention_mask)

        self.assertEqual(masked.tolist(), [[1, 2, -100], [4, -100, -100]])

    def test_save_and_load_tiny_model_families(self) -> None:
        from dpo_utils import load_checkpoint_model

        configs = [
            ModelConfig(
                architecture="gpt2",
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=1,
                n_embd=8,
                dropout=0.0,
            ),
            ModelConfig(
                architecture="qwen2",
                vocab_size=16,
                block_size=8,
                n_layer=1,
                n_head=2,
                n_embd=16,
                num_key_value_heads=1,
                intermediate_size=32,
            ),
        ]
        for config in configs:
            if config.architecture == "qwen2" and Qwen2ForCausalLM is None:
                continue
            model = build_model(config)
            checkpoint_path = Path(self.temp_dir.name) / f"{config.architecture}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config),
                    "architecture": config.architecture,
                },
                checkpoint_path,
            )

            loaded, loaded_config, _, _ = load_checkpoint_model(checkpoint_path, torch.device("cpu"))
            self.assertEqual(loaded_config.architecture, config.architecture)
            outputs = loaded(input_ids=torch.randint(0, 16, (1, 4)), labels=torch.randint(0, 16, (1, 4)))
            self.assertEqual(tuple(output_logits(outputs).shape), (1, 4, 16))

    def test_legacy_gpt2_config_without_architecture_loads_as_gpt2(self) -> None:
        config = model_config_from_mapping(
            {
                "vocab_size": 16,
                "block_size": 8,
                "n_layer": 1,
                "n_head": 1,
                "n_embd": 8,
                "dropout": 0.0,
            }
        )
        self.assertEqual(config.architecture, "gpt2")
        self.assertIsInstance(build_model(config), PinyinCodeLanguageModel)

    def test_dpo_sequence_log_probs_scores_only_completion_tokens(self) -> None:
        class UniformModel(torch.nn.Module):
            def forward(self, input_ids):
                return torch.zeros(input_ids.size(0), input_ids.size(1), 8)

        input_ids = torch.tensor([[1, 2, 3, 4]])
        completion_mask = torch.tensor([[False, False, True, True]])

        logps = sequence_log_probs(UniformModel(), input_ids, completion_mask)

        self.assertAlmostEqual(logps.item(), -2 * math.log(8), places=5)

    def test_dpo_collator_marks_completion_after_prompt(self) -> None:
        class FakeProcessor:
            def encode(self, text, out_type=int):
                return [int(token) for token in text.split()]

            def pad_id(self):
                return 0

            def eos_id(self):
                return 0

            def unk_id(self):
                return 0

            def bos_id(self):
                return 0

        collator = DPODataCollator(FakeProcessor(), max_length=8)
        batch = collator(
            [
                {
                    "prompt": "1 2",
                    "chosen": "3 4",
                    "rejected": "5",
                }
            ]
        )

        self.assertEqual(batch["chosen_input_ids"].tolist(), [[1, 2, 3, 4]])
        self.assertEqual(batch["chosen_completion_mask"].tolist(), [[False, False, True, True]])
        self.assertEqual(batch["rejected_input_ids"].tolist(), [[1, 2, 5]])
        self.assertEqual(batch["rejected_completion_mask"].tolist(), [[False, False, True]])

    def test_dpo_loss_decreases_with_larger_policy_margin(self) -> None:
        ref_chosen = torch.tensor([-4.0])
        ref_rejected = torch.tensor([-4.0])
        worse_loss, _ = dpo_loss_from_logps(
            policy_chosen_logps=torch.tensor([-5.0]),
            policy_rejected_logps=torch.tensor([-4.0]),
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            beta=0.1,
        )
        better_loss, margin = dpo_loss_from_logps(
            policy_chosen_logps=torch.tensor([-3.0]),
            policy_rejected_logps=torch.tensor([-5.0]),
            ref_chosen_logps=ref_chosen,
            ref_rejected_logps=ref_rejected,
            beta=0.1,
        )

        self.assertGreater(margin.item(), 0)
        self.assertLess(better_loss.item(), worse_loss.item())

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

    def test_convert_to_transformers_supports_hybrid_tokenizer(self) -> None:
        if PinyinCodeConfig is None:
            self.skipTest("transformers is not installed")

        from hf.convert_to_transformers import convert
        from transformers import AutoTokenizer

        corpus_path = Path(self.temp_dir.name) / "hybrid-corpus.txt"
        corpus_path.write_text("Y0J7 Y0J7 H2\n", encoding="utf-8")
        tokenizer_dir = Path(self.temp_dir.name) / "hybrid-tokenizer"
        tokenizer_args = SimpleNamespace(
            input=[corpus_path],
            output_dir=tokenizer_dir,
            vocab_size=600,
            min_word_frequency=1,
            atomic_only=False,
            initial_alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
            digits="0123456789",
            permissive=False,
            max_invalid_examples=20,
        )
        vocab, metadata = build_vocab(tokenizer_args)
        write_tokenizer_files(tokenizer_dir, vocab, metadata)

        model_config = ModelConfig(
            vocab_size=len(vocab),
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
        )
        model = PinyinCodeLanguageModel(model_config)
        checkpoint_path = Path(self.temp_dir.name) / "hybrid-best.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": asdict(model_config),
                "epoch": 1,
                "global_step": 2,
                "validation_loss": 1.23,
            },
            checkpoint_path,
        )

        output_dir = Path(self.temp_dir.name) / "hf-hybrid"
        convert_args = SimpleNamespace(
            checkpoint=checkpoint_path,
            tokenizer=tokenizer_dir,
            output_dir=output_dir,
            transliteration="pinyin-code",
            jieba=True,
            safe_serialization=False,
        )

        with contextlib.redirect_stdout(io.StringIO()):
            convert(convert_args)

        tokenizer_config = json.loads(
            (output_dir / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        model_config_json = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        training_metadata = json.loads(
            (output_dir / "training_metadata.json").read_text(encoding="utf-8")
        )

        self.assertEqual(tokenizer_config["tokenizer_class"], "HybridPinyinCodeTokenizer")
        self.assertEqual(tokenizer_config["tokenizer_kind"], "hybrid")
        self.assertEqual(
            tokenizer_config["auto_map"]["AutoTokenizer"][0],
            "tokenization_hybrid_pinyin_code.HybridPinyinCodeTokenizer",
        )
        self.assertEqual(
            model_config_json["auto_map"]["AutoTokenizer"][0],
            "tokenization_hybrid_pinyin_code.HybridPinyinCodeTokenizer",
        )
        self.assertEqual(model_config_json["vocab_size"], len(vocab))
        self.assertEqual(training_metadata["tokenizer_kind"], "hybrid")
        self.assertTrue((output_dir / "vocab.json").exists())
        self.assertTrue((output_dir / "hybrid_tokenizer_metadata.json").exists())
        self.assertTrue((output_dir / "tokenization_hybrid_pinyin_code.py").exists())

        tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
        self.assertEqual(tokenizer.tokenize("Y0J7 H2 X4Q3"), ["Y0J7", "H2", "X4", "Q3"])

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

    def test_hf_base_model_forward_returns_hidden_states(self) -> None:
        if PinyinCodeConfig is None or PinyinCodeModel is None:
            self.skipTest("transformers is not installed")

        config = PinyinCodeConfig(vocab_size=16, block_size=8, n_layer=1, n_head=1, n_embd=8)
        model = PinyinCodeModel(config)
        out = model(torch.tensor([[1, 2, 3]]), output_hidden_states=True)

        self.assertEqual(tuple(out.last_hidden_state.shape), (1, 3, 8))
        self.assertIsNotNone(out.hidden_states)
        assert out.hidden_states is not None
        self.assertEqual(len(out.hidden_states), 3)
        self.assertEqual(tuple(out.hidden_states[-1].shape), (1, 3, 8))


if __name__ == "__main__":
    unittest.main()
