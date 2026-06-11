from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import TensorDataset

import train_model as train_model_module
from create_dataset import iter_chunks, write_dataset
from generate import load_model as load_generation_model
from generate import prepare_prompt
from preprocessing.preprocess import hanzi_to_encoded, process_text
from bert.scoring import score_sentence_pseudo_likelihood
from train_sentencepiece import train_tokenizer
from train_model import (
    BabyBertForMaskedLM,
    BertSpecialTokenIds,
    BidirectionalSelfAttention,
    BinaryTokenDataset,
    JsonlTokenDataset,
    MLMDataCollator,
    ModelConfig,
    PinyinCodeLanguageModel,
    build_model,
    checkpoint_payload,
    load_token_dataset,
    learning_rate_for_step,
    normalize_model_config_dict,
    split_dataset,
    train,
    validate_bert_special_tokens,
    validate_dataset_compatibility,
)

try:
    from hf.configuration_pinyin_code import PinyinCodeConfig
    from hf.modeling_pinyin_code import (
        PinyinCodeEncoderModel,
        PinyinCodeForCausalLM,
        PinyinCodeForMaskedLM,
        PinyinCodeModel,
    )
    from hf.tokenization_pinyin_code import EncodedMandarinTokenizer
except ModuleNotFoundError:
    EncodedMandarinTokenizer = None
    PinyinCodeConfig = None
    PinyinCodeEncoderModel = None
    PinyinCodeForCausalLM = None
    PinyinCodeForMaskedLM = None
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
        self.assertEqual(captured["pad_piece"], "<pad>")
        self.assertEqual(captured["unk_piece"], "<unk>")

    def test_sentencepiece_training_can_emit_bert_special_pieces(self) -> None:
        captured = {}

        class FakeTrainer:
            @staticmethod
            def train(**kwargs):
                captured.update(kwargs)

        fake_sentencepiece = SimpleNamespace(SentencePieceTrainer=FakeTrainer)
        args = SimpleNamespace(
            output_dir=Path(self.temp_dir.name),
            model_name="toy-bert",
            input=[Path("toy.txt")],
            vocab_size=32,
            model_type="bpe",
            character_coverage=1.0,
            input_sentence_size=0,
            max_sentence_length=1024,
            shuffle_input_sentence=True,
            train_extremely_large_corpus=False,
            hard_vocab_limit=False,
            special_token_style="bert",
        )

        with (
            patch("train_sentencepiece.require_sentencepiece", return_value=fake_sentencepiece),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train_tokenizer(args)

        self.assertEqual(captured["pad_piece"], "[PAD]")
        self.assertEqual(captured["unk_piece"], "[UNK]")
        self.assertEqual(captured["bos_piece"], "[CLS]")
        self.assertEqual(captured["eos_piece"], "[SEP]")
        self.assertIn("[MASK]", captured["user_defined_symbols"].split(","))

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

    def test_gpt_loss_keeps_next_token_shift_behavior(self) -> None:
        model = PinyinCodeLanguageModel(
            ModelConfig(vocab_size=32, block_size=8, n_layer=1, n_head=2, n_embd=16)
        )
        input_ids = torch.randint(0, 32, (2, 8))
        logits, loss = model(input_ids, labels=input_ids)
        manual_loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
        )

        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.allclose(loss, manual_loss))

    def test_bert_special_token_validation_reports_missing_pieces(self) -> None:
        class FakeProcessor:
            pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
            ids = {piece: index for index, piece in enumerate(pieces)}

            def piece_to_id(self, piece: str) -> int:
                return self.ids.get(piece, self.ids["[UNK]"])

            def id_to_piece(self, index: int) -> str:
                return self.pieces[index]

        with self.assertRaisesRegex(ValueError, r"\[MASK\]"):
            validate_bert_special_tokens(FakeProcessor())

    def test_mlm_collator_creates_masked_labels_and_ignores_special_tokens(self) -> None:
        special_ids = BertSpecialTokenIds(mask=4, pad=0, unk=1, cls=2, sep=3)
        collator = MLMDataCollator(
            special_token_ids=special_ids,
            vocab_size=12,
            mlm_probability=1.0,
        )

        batch = collator(
            [
                torch.tensor([2, 5, 6, 3]),
                torch.tensor([0, 7, 1, 0]),
            ]
        )

        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1, 1], [0, 1, 1, 0]])
        self.assertEqual(batch["labels"].tolist(), [[-100, 5, 6, -100], [-100, 7, -100, -100]])
        self.assertEqual(batch["input_ids"][0, 0].item(), 2)
        self.assertEqual(batch["input_ids"][0, 3].item(), 3)
        self.assertEqual(batch["input_ids"][1, 0].item(), 0)

    def test_tiny_bert_forward_has_mlm_logits_and_finite_loss(self) -> None:
        model = BabyBertForMaskedLM(
            ModelConfig(
                vocab_size=16,
                block_size=5,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                model_type="bert",
            )
        )
        input_ids = torch.tensor([[2, 5, 6, 7, 3], [2, 6, 7, 8, 3]])
        attention_mask = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        labels[0, 2] = 6
        labels[1, 1] = 6

        logits, loss = model(input_ids, attention_mask=attention_mask, labels=labels)

        self.assertEqual(tuple(logits.shape), (2, 5, 16))
        self.assertIsNotNone(loss)
        assert loss is not None
        self.assertTrue(torch.isfinite(loss))

    def test_bert_attention_does_not_use_causal_masking(self) -> None:
        model = BabyBertForMaskedLM(
            ModelConfig(
                vocab_size=16,
                block_size=4,
                n_layer=1,
                n_head=2,
                n_embd=8,
                dropout=0.0,
                model_type="bert",
            )
        )
        self.assertIsInstance(model.blocks[0].attn, BidirectionalSelfAttention)

        calls = []
        original_attention = train_model_module.F.scaled_dot_product_attention

        def wrapped_attention(*args, **kwargs):
            calls.append(kwargs.get("is_causal"))
            return original_attention(*args, **kwargs)

        with patch("train_model.F.scaled_dot_product_attention", side_effect=wrapped_attention):
            model(torch.tensor([[2, 5, 6, 3]]), attention_mask=torch.ones(1, 4, dtype=torch.long))

        self.assertEqual(calls, [False])

    def test_bert_checkpoint_can_save_and_reload(self) -> None:
        config = ModelConfig(
            vocab_size=16,
            block_size=4,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            model_type="bert",
        )
        model = BabyBertForMaskedLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        checkpoint = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            config=config,
            epoch=1,
            global_step=2,
            validation_loss=3.0,
            best_loss=3.0,
            save_optimizer=False,
        )
        checkpoint_path = Path(self.temp_dir.name) / "bert.pt"
        torch.save(checkpoint, checkpoint_path)

        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        loaded_config = ModelConfig(**normalize_model_config_dict(loaded["model_config"]))
        reloaded_model = build_model(loaded_config)
        reloaded_model.load_state_dict(loaded["model_state_dict"])
        logits, _ = reloaded_model(torch.tensor([[2, 5, 6, 3]]), attention_mask=torch.ones(1, 4))

        self.assertEqual(loaded["model_config"]["model_type"], "bert")
        self.assertEqual(tuple(logits.shape), (1, 4, 16))

    def test_generation_rejects_bert_checkpoints(self) -> None:
        config = ModelConfig(
            vocab_size=16,
            block_size=4,
            n_layer=1,
            n_head=2,
            n_embd=8,
            dropout=0.0,
            model_type="bert",
        )
        model = BabyBertForMaskedLM(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        checkpoint_path = Path(self.temp_dir.name) / "bert-generation.pt"
        torch.save(
            checkpoint_payload(model, optimizer, config, 1, 1, 1.0, 1.0, False),
            checkpoint_path,
        )

        with self.assertRaisesRegex(SystemExit, "not autoregressive"):
            load_generation_model(checkpoint_path, torch.device("cpu"))

    def test_bert_pseudo_likelihood_scores_non_special_tokens(self) -> None:
        class FakeTokenizer:
            pieces = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "A", "B"]
            ids = {piece: index for index, piece in enumerate(pieces)}

            def piece_to_id(self, piece: str) -> int:
                return self.ids.get(piece, self.ids["[UNK]"])

            def id_to_piece(self, index: int) -> str:
                return self.pieces[index]

        model = BabyBertForMaskedLM(
            ModelConfig(
                vocab_size=8,
                block_size=4,
                n_layer=1,
                n_head=1,
                n_embd=8,
                dropout=0.0,
                model_type="bert",
            )
        )

        score = score_sentence_pseudo_likelihood(
            model=model,
            input_ids=[2, 5, 6, 3],
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
        )
        mean_score = score_sentence_pseudo_likelihood(
            model=model,
            input_ids=[2, 5, 6, 3],
            tokenizer=FakeTokenizer(),
            device=torch.device("cpu"),
            normalize="mean",
        )

        self.assertTrue(math.isfinite(score))
        self.assertLess(score, 0.0)
        self.assertAlmostEqual(mean_score, score / 2)

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

    def test_tiny_bert_train_loop_writes_reloadable_checkpoint(self) -> None:
        dataset_path = self.write_jsonl_dataset(
            [[2, 5, 6, 3], [2, 6, 7, 3], [2, 7, 8, 3], [2, 8, 9, 3]]
        )
        validation_path = self.write_jsonl_dataset(
            [[2, 5, 7, 3], [2, 6, 8, 3]],
            name="bert-validation.jsonl",
        )
        output_dir = Path(self.temp_dir.name) / "bert-model"
        args = SimpleNamespace(
            model_type="bert",
            dataset=dataset_path,
            validation_dataset=validation_path,
            tokenizer=Path("unused.model"),
            output_dir=output_dir,
            vocab_size=16,
            block_size=4,
            n_layer=1,
            n_head=1,
            n_embd=8,
            dropout=0.0,
            mlm_probability=0.5,
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            gradient_accumulation_steps=1,
            lr_schedule="cosine",
            warmup_steps=0,
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

        special_ids = BertSpecialTokenIds(mask=4, pad=0, unk=1, cls=2, sep=3)
        with (
            patch("train_model.load_bert_special_token_ids", return_value=special_ids),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            train(args)

        checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["model_config"]["model_type"], "bert")
        reloaded_model = build_model(ModelConfig(**checkpoint["model_config"]))
        reloaded_model.load_state_dict(checkpoint["model_state_dict"])
        logits, _ = reloaded_model(
            torch.tensor([[2, 5, 6, 3]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
        self.assertEqual(tuple(logits.shape), (1, 4, 16))

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

    def test_hf_masked_lm_forward_returns_logits_and_loss(self) -> None:
        if PinyinCodeConfig is None or PinyinCodeForMaskedLM is None:
            self.skipTest("transformers is not installed")

        config = PinyinCodeConfig(
            vocab_size=16,
            block_size=8,
            n_layer=1,
            n_head=1,
            n_embd=8,
            training_model_type="bert",
            pad_token_id=0,
            unk_token_id=1,
            cls_token_id=2,
            sep_token_id=3,
            mask_token_id=4,
        )
        model = PinyinCodeForMaskedLM(config)
        input_ids = torch.tensor([[2, 4, 6, 3]])
        labels = torch.full_like(input_ids, -100)
        labels[0, 1] = 5
        out = model(input_ids, attention_mask=torch.ones_like(input_ids), labels=labels)

        self.assertEqual(tuple(out.logits.shape), (1, 4, 16))
        self.assertIsNotNone(out.loss)
        assert out.loss is not None
        self.assertTrue(torch.isfinite(out.loss))
        self.assertFalse(config.is_decoder)
        self.assertEqual(config.recommended_score_normalization, "mean")


if __name__ == "__main__":
    unittest.main()
