"""Create a chunked language-modeling dataset from SentencePiece-tokenized text."""

from __future__ import annotations

import argparse
import json
import random
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def require_sentencepiece():
    """Import SentencePiece or stop with an install hint."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `py -m pip install sentencepiece`."
        ) from exc

    return spm


def iter_token_ids(input_paths: Iterable[Path], processor) -> Iterable[int]:
    """Yield token ids from processed text files, adding EOS after each document."""
    for document_ids in iter_document_token_ids(input_paths, processor):
        yield from document_ids


def iter_document_token_ids(input_paths: Iterable[Path], processor) -> Iterable[list[int]]:
    """Yield one token-id list per processed document, with EOS appended."""
    eos_id = processor.eos_id()

    for input_path in input_paths:
        with input_path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                token_ids = processor.encode(text, out_type=int)
                if eos_id >= 0:
                    token_ids.append(eos_id)
                if token_ids:
                    yield token_ids


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


def write_binary_metadata(path: Path, writer: ChunkWriter, args: argparse.Namespace) -> None:
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open_output(args.output, args.format) as out:
        writer = make_chunk_writer(out, args)
        for document_ids in iter_document_token_ids(args.input, processor):
            writer.add_tokens(document_ids)

    if args.format == "bin":
        write_binary_metadata(args.output, writer, args)

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

    with (
        open_output(args.output, args.format) as train_out,
        open_output(args.validation_output, args.format) as valid_out,
    ):
        train_writer = make_chunk_writer(train_out, args)
        valid_writer = make_chunk_writer(valid_out, args)

        for document_ids in iter_document_token_ids(args.input, processor):
            writer = valid_writer if rng.random() < args.validation_fraction else train_writer
            writer.add_tokens(document_ids)

    if args.format == "bin":
        write_binary_metadata(args.output, train_writer, args)
        write_binary_metadata(args.validation_output, valid_writer, args)

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

    spm = require_sentencepiece()
    processor = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

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
        help="SentencePiece .model file used to encode the dataset.",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
