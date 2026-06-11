"""Build DPO preference pairs from raw Hanzi samples and a pretrained LM."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo_utils import (  # noqa: E402
    decode_ids,
    load_checkpoint_model,
    load_sentencepiece,
    normalize_preference_text,
)
from generate import generate as generate_tokens  # noqa: E402
from preprocessing.preprocess import hanzi_to_encoded, require_dependencies  # noqa: E402


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
MALFORMED_MARKERS = {"\ufffd", "\u2047"}


def resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested generation device."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch environment cannot see a CUDA GPU.")
    return torch.device(requested_device)


def first_text_value(record: dict[str, Any]) -> str:
    """Extract the most likely source text field from a JSONL object."""
    for key in ("text", "source_hanzi", "hanzi", "content", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for value in record.values():
        if isinstance(value, str) and value.strip():
            return value
    return ""


def iter_input_samples(path: Path) -> Iterable[str]:
    """Yield Hanzi/plain-text samples from .txt or JSONL input."""
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            if path.suffix.lower() == ".jsonl":
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object on line {line_number}")
                text = first_text_value(record)
            else:
                text = line
            if text.strip():
                yield text.strip()


def encode_source_text(text: str, use_jieba: bool) -> str:
    """Convert Hanzi to compact pinyin-code, preserving already-encoded lines."""
    if CHINESE_RE.search(text):
        return hanzi_to_encoded(text, use_jieba=use_jieba)
    return text.strip()


def tokenize_encoded_document(processor, encoded_text: str, max_length: int) -> list[int]:
    """Tokenize one encoded document and append EOS when the tokenizer has one."""
    token_ids = [int(token_id) for token_id in processor.encode(encoded_text, out_type=int)]
    eos_id = processor.eos_id()
    if eos_id >= 0:
        token_ids.append(int(eos_id))
    return token_ids[:max_length]


def split_prompt_chosen(
    token_ids: list[int],
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Split a causal LM sequence into a 30-50 percent prompt and gold continuation."""
    if len(token_ids) < 4:
        raise ValueError("Need at least four tokens to create a prompt/continuation pair")
    min_split = max(1, int(len(token_ids) * 0.30))
    max_split = max(min_split, int(len(token_ids) * 0.50))
    split_index = rng.randint(min_split, max_split)
    split_index = min(max(1, split_index), len(token_ids) - 1)
    return token_ids[:split_index], token_ids[split_index:]


def candidate_is_valid(
    rejected_ids: list[int],
    rejected_text: str,
    chosen_ids: list[int],
    chosen_text: str,
    seen_rejections: set[str],
) -> bool:
    """Filter empty, malformed, duplicate, or identical generated continuations."""
    if not rejected_ids:
        return False
    if any(marker in rejected_text for marker in MALFORMED_MARKERS):
        return False
    normalized_rejected = normalize_preference_text(rejected_text)
    normalized_chosen = normalize_preference_text(chosen_text)
    if not normalized_rejected:
        return False
    if rejected_ids == chosen_ids or normalized_rejected == normalized_chosen:
        return False
    if normalized_rejected in seen_rejections:
        return False
    return True


@torch.no_grad()
def generate_rejections(
    model,
    processor,
    prompt_ids: list[int],
    chosen_ids: list[int],
    chosen_text: str,
    args: argparse.Namespace,
) -> list[tuple[list[int], str]]:
    """Generate incorrect continuations for one prompt."""
    rejections: list[tuple[list[int], str]] = []
    seen_rejections: set[str] = set()
    max_new_tokens = max(1, min(len(chosen_ids), args.max_length - len(prompt_ids)))
    top_k = args.top_k if args.top_k > 0 else None

    for _ in range(args.num_candidates):
        generated_ids = generate_tokens(
            model=model,
            input_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            eos_id=processor.eos_id(),
            device=args.resolved_device,
        )
        rejected_ids = generated_ids[len(prompt_ids) :]
        rejected_text = decode_ids(processor, rejected_ids)
        if not candidate_is_valid(
            rejected_ids,
            rejected_text,
            chosen_ids,
            chosen_text,
            seen_rejections,
        ):
            continue
        seen_rejections.add(normalize_preference_text(rejected_text))
        rejections.append((rejected_ids, rejected_text))

    return rejections


def build_preferences(args: argparse.Namespace) -> int:
    """Create preference JSONL from source samples."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be greater than zero")
    if args.max_length < 4:
        raise ValueError("--max-length must be at least 4")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than zero when provided")

    require_dependencies()
    device = resolve_device(args.device)
    args.resolved_device = device
    processor = load_sentencepiece(args.tokenizer)
    model, config, resolved_checkpoint, _ = load_checkpoint_model(
        args.model_checkpoint,
        device,
        eval_mode=True,
    )
    tokenizer_vocab_size = len(processor)
    if tokenizer_vocab_size != config.vocab_size:
        raise ValueError(
            "Tokenizer vocab size does not match checkpoint config: "
            f"tokenizer={tokenizer_vocab_size}, checkpoint={config.vocab_size}."
        )

    effective_max_length = min(args.max_length, config.block_size)
    if effective_max_length != args.max_length:
        print(
            "warning: clamping --max-length from "
            f"{args.max_length} to checkpoint block_size={config.block_size}"
        )
        args.max_length = effective_max_length

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    processed_sources = 0
    skipped_sources = 0
    written_pairs = 0

    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for source_hanzi in iter_input_samples(args.input_hanzi):
            if args.num_samples is not None and processed_sources >= args.num_samples:
                break
            processed_sources += 1

            encoded_text = encode_source_text(source_hanzi, use_jieba=args.jieba)
            token_ids = tokenize_encoded_document(processor, encoded_text, args.max_length)
            try:
                prompt_ids, chosen_ids = split_prompt_chosen(token_ids, rng)
            except ValueError:
                skipped_sources += 1
                continue

            chosen_text = decode_ids(processor, chosen_ids)
            if not normalize_preference_text(chosen_text):
                skipped_sources += 1
                continue

            rejections = generate_rejections(
                model,
                processor,
                prompt_ids,
                chosen_ids,
                chosen_text,
                args,
            )
            if not rejections:
                skipped_sources += 1
                continue

            prompt_text = decode_ids(processor, prompt_ids)
            for rejected_ids, rejected_text in rejections:
                record = {
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text,
                    "source_hanzi": source_hanzi,
                    "prompt_ids": prompt_ids,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                }
                out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
                out.write("\n")
                written_pairs += 1

    if written_pairs == 0:
        raise SystemExit("No DPO preference pairs were written; try more samples or candidates.")

    print(
        "dpo_dataset_built "
        f"output={args.output} "
        f"checkpoint={resolved_checkpoint} "
        f"sources={processed_sources} "
        f"skipped_sources={skipped_sources} "
        f"pairs={written_pairs}"
    )
    return written_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DPO preference pairs from Hanzi input and a pretrained pinyin-code LM."
    )
    parser.add_argument("--input-hanzi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--jieba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use jieba segmentation when converting raw Hanzi to pinyin-code.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_preferences(args)


if __name__ == "__main__":
    main()
