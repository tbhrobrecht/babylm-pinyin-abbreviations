"""Create a chunked language-modeling dataset from tokenized pinyin-code text."""

from __future__ import annotations

import argparse
import json
import random
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def require_sentencepiece():
    """Import SentencePiece or stop with an install hint."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `py -m pip install sentencepiece`."
        ) from exc

    return spm


def require_hybrid_tokenizer():
    """Import the repository's hybrid tokenizer or stop with an install hint."""
    try:
        from hf.tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency for the hybrid tokenizer: install `transformers` "
            "with `py -m pip install transformers`."
        ) from exc

    return HybridPinyinCodeTokenizer


TOKENIZATION_MODES = ("greedy", "softmax")


def hybrid_tokenization_kwargs(settings: dict[str, Any] | None) -> dict[str, Any]:
    """Return the subset of tokenization settings accepted by the hybrid tokenizer."""
    if not settings:
        return {}
    kwargs: dict[str, Any] = {}
    for source, target in (
        ("tokenization_mode", "tokenization_mode"),
        ("sampling_temperature", "sampling_temperature"),
        ("sampling_alpha", "sampling_alpha"),
        ("sampling_beta", "sampling_beta"),
        ("sampling_epsilon", "sampling_epsilon"),
        ("sampling_seed", "sampling_seed"),
    ):
        if settings.get(source) is not None:
            kwargs[target] = settings[source]
    return kwargs


class HybridProcessorAdapter:
    """Expose the small SentencePiece-like API used by this script."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encode(self, text: str, out_type=int) -> list[int] | list[str]:
        tokens = self.tokenizer.tokenize(text)
        if out_type is str:
            return tokens
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def eos_id(self) -> int:
        eos_id = self.tokenizer.eos_token_id
        return eos_id if eos_id is not None else -1


def apply_tokenization_mode(processor, mode: str | None) -> None:
    """Set the tokenization mode on a hybrid processor; no-op for SentencePiece."""
    if mode is None:
        return
    tokenizer = getattr(processor, "tokenizer", None)
    setter = getattr(tokenizer, "set_tokenization_mode", None)
    if callable(setter):
        setter(mode)


def load_tokenizer_processor(tokenizer_path: Path, tokenization_settings: dict[str, Any] | None = None):
    """Load either a legacy SentencePiece model or a hybrid tokenizer directory.

    ``tokenization_settings`` (tokenization_mode, sampling_*) only affects the
    hybrid tokenizer; SentencePiece has a single deterministic segmentation and
    ignores them.
    """
    if tokenizer_path.is_dir() or tokenizer_path.name == "vocab.json":
        HybridPinyinCodeTokenizer = require_hybrid_tokenizer()
        load_path = tokenizer_path.parent if tokenizer_path.name == "vocab.json" else tokenizer_path
        return HybridProcessorAdapter(
            HybridPinyinCodeTokenizer.from_pretrained(
                load_path,
                **hybrid_tokenization_kwargs(tokenization_settings),
            )
        )

    spm = require_sentencepiece()
    return spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def iter_token_ids(input_paths: Iterable[Path], processor) -> Iterable[int]:
    """Yield token ids from processed text files, adding EOS after each document."""
    for document_ids in iter_document_token_ids(input_paths, processor):
        yield from document_ids


def iter_document_token_ids(input_paths: Iterable[Path], processor) -> Iterable[list[int]]:
    """Yield one token-id list per processed document, with EOS appended."""
    eos_id = processor.eos_id()

    for text in iter_documents(input_paths):
        token_ids = encode_document(processor, text, eos_id)
        if token_ids:
            yield token_ids


def iter_documents(input_paths: Iterable[Path]) -> Iterable[str]:
    """Yield each non-empty processed document line."""
    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    yield text


def encode_document(processor, text: str, eos_id: int) -> list[int]:
    """Encode one document and append EOS when the tokenizer defines one."""
    token_ids = processor.encode(text, out_type=int)
    if eos_id >= 0:
        token_ids.append(eos_id)
    return token_ids


def iter_chunks(token_ids: Iterable[int], block_size: int, stride: int) -> Iterable[list[int]]:
    """Yield fixed-size token chunks using a sliding window."""
    if block_size <= 0:
        raise ValueError("--block-size must be greater than zero")
    if stride <= 0:
        raise ValueError("--stride must be greater than zero")
    if stride > block_size:
        raise ValueError("--stride must be less than or equal to --block-size")

    buffer: list[int] = []
    for token_id in token_ids:
        buffer.append(token_id)
        if len(buffer) == block_size:
            yield list(buffer)
            del buffer[:stride]


@dataclass
class ChunkWriter:
    """Write fixed-size chunks while keeping sliding-window state."""

    handle: object
    output_format: str
    block_size: int
    stride: int
    include_labels: bool
    buffer: list[int]
    written: int = 0
    consumed_tokens: int = 0
    max_token_id: int = -1

    def add_tokens(self, token_ids: Iterable[int]) -> None:
        for token_id in token_ids:
            self.consumed_tokens += 1
            self.max_token_id = max(self.max_token_id, token_id)
            self.buffer.append(token_id)
            if len(self.buffer) == self.block_size:
                self.write_chunk(self.buffer)
                del self.buffer[: self.stride]

    def write_chunk(self, chunk: list[int]) -> None:
        if self.output_format == "jsonl":
            record = {"input_ids": list(chunk)}
            if self.include_labels:
                record["labels"] = list(chunk)
            self.handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            self.handle.write("\n")
        elif self.output_format == "bin":
            if min(chunk) < 0 or max(chunk) > 2_147_483_647:
                raise ValueError("Binary datasets require token ids in int32 range")
            self.handle.write(struct.pack(f"<{len(chunk)}i", *chunk))
        else:
            raise ValueError(f"Unsupported dataset format: {self.output_format}")
        self.written += 1

    @property
    def dropped_tail_tokens(self) -> int:
        return len(self.buffer)


@dataclass
class DatasetWriteStats:
    train_examples: int
    train_tokens: int
    train_dropped_tail_tokens: int
    validation_examples: int = 0
    validation_tokens: int = 0
    validation_dropped_tail_tokens: int = 0


def make_chunk_writer(handle, args: argparse.Namespace) -> ChunkWriter:
    return ChunkWriter(
        handle=handle,
        output_format=args.format,
        block_size=args.block_size,
        stride=args.stride,
        include_labels=args.include_labels,
        buffer=[],
    )


def binary_metadata_path(path: Path) -> Path:
    """Return the sidecar metadata path for a binary chunk file."""
    return path.with_suffix(path.suffix + ".meta.json")


def tokenization_metadata(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    """Return tokenization settings recorded in a binary dataset sidecar."""
    return {
        "tokenizer": str(args.tokenizer),
        "tokenization_mode": mode,
        "train_tokenization_mode": getattr(args, "tokenization_mode", "greedy"),
        "eval_tokenization_mode": getattr(args, "eval_tokenization_mode", "greedy"),
        "sampling_temperature": getattr(args, "sampling_temperature", 1.0),
        "sampling_alpha": getattr(args, "sampling_alpha", 1.0),
        "sampling_beta": getattr(args, "sampling_beta", 1.0),
        "sampling_epsilon": getattr(args, "sampling_epsilon", 1e-8),
        "sampling_seed": getattr(args, "sampling_seed", None),
    }


def write_binary_metadata(
    path: Path,
    writer: ChunkWriter,
    args: argparse.Namespace,
    mode: str = "greedy",
) -> None:
    """Write metadata needed to memory-map a binary dataset."""
    payload = {
        "format": "pinyin-code-chunks-v1",
        "dtype": "int32_le",
        "num_examples": writer.written,
        "block_size": args.block_size,
        "stride": args.stride,
        "include_labels": False,
        "consumed_tokens": writer.consumed_tokens,
        "dropped_tail_tokens": writer.dropped_tail_tokens,
        "max_token_id": writer.max_token_id,
        "tokenization": tokenization_metadata(args, mode),
    }
    binary_metadata_path(path).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def open_output(path: Path, output_format: str):
    """Open an output dataset file in the mode required by the selected format."""
    if output_format == "jsonl":
        return path.open("w", encoding="utf-8", newline="\n")
    if output_format == "bin":
        return path.open("wb")
    raise ValueError(f"Unsupported dataset format: {output_format}")


def write_single_dataset(args: argparse.Namespace, processor) -> DatasetWriteStats:
    """Write all input documents into one chunked dataset."""
    train_mode = getattr(args, "tokenization_mode", "greedy")
    apply_tokenization_mode(processor, train_mode)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, args.format) as out:
        writer = make_chunk_writer(out, args)
        for document_ids in iter_document_token_ids(args.input, processor):
            writer.add_tokens(document_ids)

    if args.format == "bin":
        write_binary_metadata(args.output, writer, args, train_mode)

    return DatasetWriteStats(
        train_examples=writer.written,
        train_tokens=writer.consumed_tokens,
        train_dropped_tail_tokens=writer.dropped_tail_tokens,
    )


def write_train_validation_datasets(args: argparse.Namespace, processor) -> DatasetWriteStats:
    """Split processed documents, then write separate train/validation chunks."""
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be greater than 0 and less than 1")
    if args.validation_output is None:
        raise ValueError("--validation-output is required when using --validation-fraction")

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.validation_output.parent.mkdir(parents=True, exist_ok=True)

    train_mode = getattr(args, "tokenization_mode", "greedy")
    eval_mode = getattr(args, "eval_tokenization_mode", "greedy")
    same_mode = train_mode == eval_mode
    eos_id = processor.eos_id()

    with (
        open_output(args.output, args.format) as train_out,
        open_output(args.validation_output, args.format) as valid_out,
    ):
        train_writer = make_chunk_writer(train_out, args)
        valid_writer = make_chunk_writer(valid_out, args)

        if same_mode:
            # Encoding is mode-independent across splits, so keep the original
            # routing order exactly: encode, then draw once per non-empty
            # document. This preserves byte-for-byte greedy dataset outputs.
            apply_tokenization_mode(processor, train_mode)
            for document_ids in iter_document_token_ids(args.input, processor):
                writer = valid_writer if rng.random() < args.validation_fraction else train_writer
                writer.add_tokens(document_ids)
        else:
            # Train and eval use different segmentation policies, so decide the
            # split first, then encode that document with the matching mode.
            for text in iter_documents(args.input):
                to_valid = rng.random() < args.validation_fraction
                apply_tokenization_mode(processor, eval_mode if to_valid else train_mode)
                document_ids = encode_document(processor, text, eos_id)
                if not document_ids:
                    continue
                (valid_writer if to_valid else train_writer).add_tokens(document_ids)

    if args.format == "bin":
        write_binary_metadata(args.output, train_writer, args, train_mode)
        write_binary_metadata(args.validation_output, valid_writer, args, eval_mode)

    return DatasetWriteStats(
        train_examples=train_writer.written,
        train_tokens=train_writer.consumed_tokens,
        train_dropped_tail_tokens=train_writer.dropped_tail_tokens,
        validation_examples=valid_writer.written,
        validation_tokens=valid_writer.consumed_tokens,
        validation_dropped_tail_tokens=valid_writer.dropped_tail_tokens,
    )


def write_dataset(args: argparse.Namespace) -> DatasetWriteStats:
    """Tokenize processed text and write fixed-size chunk records."""
    if args.format == "bin" and args.include_labels:
        raise ValueError("--include-labels is only supported for --format jsonl")

    tokenization_settings = {
        "tokenization_mode": getattr(args, "tokenization_mode", "greedy"),
        "sampling_temperature": getattr(args, "sampling_temperature", 1.0),
        "sampling_alpha": getattr(args, "sampling_alpha", 1.0),
        "sampling_beta": getattr(args, "sampling_beta", 1.0),
        "sampling_epsilon": getattr(args, "sampling_epsilon", 1e-8),
        "sampling_seed": getattr(args, "sampling_seed", None),
    }
    processor = load_tokenizer_processor(args.tokenizer, tokenization_settings)

    if args.validation_output is not None or args.validation_fraction is not None:
        validation_fraction = args.validation_fraction
        if validation_fraction is None:
            validation_fraction = 0.05
        args.validation_fraction = validation_fraction
        stats = write_train_validation_datasets(args, processor)
    else:
        stats = write_single_dataset(args, processor)

    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line options for dataset creation."""
    parser = argparse.ArgumentParser(
        description="Build a chunked language-modeling dataset from processed pinyin-code text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("data/processed/10k_babylm_zho.txt")],
        help="One or more UTF-8 processed text files, one document per line.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizers/babylm_zho_pinyin_spm.model"),
        help=(
            "Tokenizer used to encode the dataset. Pass a SentencePiece .model "
            "file or a hybrid tokenizer directory containing vocab.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/datasets/10k_babylm_zho_spm.jsonl"),
        help="Output dataset path.",
    )
    parser.add_argument(
        "--format",
        choices=("jsonl", "bin"),
        default="jsonl",
        help=(
            "Dataset format. 'jsonl' is human-readable; 'bin' writes compact "
            "int32 chunks plus a .meta.json sidecar for faster training loads."
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Number of token ids per training example.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=128,
        help="Sliding-window stride. Use a value smaller than block size for overlap.",
    )
    parser.add_argument(
        "--include-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also write labels identical to input_ids. Disabled by default because "
            "train_model.py creates causal LM labels from input_ids directly."
        ),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        help=(
            "Optional output path for a document-level validation split. "
            "When provided, --output receives only training documents."
        ),
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of processed documents assigned to --validation-output. "
            "Defaults to 0.05 when --validation-output is set."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)

    tokenization = parser.add_argument_group("hybrid tokenization mode")
    tokenization.add_argument(
        "--tokenization-mode",
        choices=TOKENIZATION_MODES,
        default="greedy",
        help=(
            "Segmentation policy for the hybrid tokenizer's training split. "
            "'greedy' is deterministic longest match (default); 'softmax' samples "
            "locally at each atomic position. Ignored by SentencePiece tokenizers."
        ),
    )
    tokenization.add_argument(
        "--eval-tokenization-mode",
        choices=TOKENIZATION_MODES,
        default="greedy",
        help=(
            "Segmentation policy for the validation split. Defaults to greedy so "
            "evaluation is deterministic even when training uses softmax."
        ),
    )
    tokenization.add_argument("--sampling-temperature", type=float, default=1.0)
    tokenization.add_argument("--sampling-alpha", type=float, default=1.0)
    tokenization.add_argument("--sampling-beta", type=float, default=1.0)
    tokenization.add_argument("--sampling-epsilon", type=float, default=1e-8)
    tokenization.add_argument(
        "--sampling-seed",
        type=int,
        default=None,
        help="Seed for the hybrid softmax RNG. Makes softmax datasets reproducible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "tokenization_setup "
        f"tokenizer={args.tokenizer} "
        f"train_tokenization_mode={args.tokenization_mode} "
        f"eval_tokenization_mode={args.eval_tokenization_mode} "
        f"sampling_temperature={args.sampling_temperature:g} "
        f"sampling_alpha={args.sampling_alpha:g} "
        f"sampling_beta={args.sampling_beta:g} "
        f"sampling_epsilon={args.sampling_epsilon:g} "
        f"sampling_seed={args.sampling_seed}"
    )
    stats = write_dataset(args)
    print(f"Wrote {stats.train_examples:,} training examples to {args.output}")
    print(
        "Training tokens consumed: "
        f"{stats.train_tokens:,}; dropped tail tokens: {stats.train_dropped_tail_tokens:,}"
    )
    if args.validation_output is not None:
        print(f"Wrote {stats.validation_examples:,} validation examples to {args.validation_output}")
        print(
            "Validation tokens consumed: "
            f"{stats.validation_tokens:,}; dropped tail tokens: "
            f"{stats.validation_dropped_tail_tokens:,}"
        )


if __name__ == "__main__":
    main()
