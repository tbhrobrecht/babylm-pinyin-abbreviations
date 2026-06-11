"""Evaluate base and DPO checkpoints on preference pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from dpo_utils import (
    DPODataCollator,
    DPOPreferenceDataset,
    batch_to_device,
    decode_ids,
    load_checkpoint_model,
    load_sentencepiece,
    sequence_log_probs,
)
from generate import generate as generate_tokens


def resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested evaluation device."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch environment cannot see a CUDA GPU.")
    return torch.device(requested_device)


def validate_args(args: argparse.Namespace) -> None:
    """Validate evaluation arguments."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.num_examples is not None and args.num_examples <= 0:
        raise ValueError("--num-examples must be greater than zero when provided")
    if args.examples < 0:
        raise ValueError("--examples must be greater than or equal to zero")
    if args.example_max_new_tokens <= 0:
        raise ValueError("--example-max-new-tokens must be greater than zero")


@torch.no_grad()
def score_model(model, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Compute preference accuracy and chosen-minus-rejected log-prob margin."""
    model.eval()
    margin_sum = 0.0
    correct = 0
    examples = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        chosen_logps = sequence_log_probs(
            model,
            batch["chosen_input_ids"],
            batch["chosen_completion_mask"],
        )
        rejected_logps = sequence_log_probs(
            model,
            batch["rejected_input_ids"],
            batch["rejected_completion_mask"],
        )
        margin = chosen_logps - rejected_logps
        margin_sum += margin.sum().item()
        correct += int((margin > 0).sum().item())
        examples += margin.numel()

    if examples == 0:
        raise ValueError("Evaluation dataset is empty")

    return {
        "preference_accuracy": correct / examples,
        "avg_margin": margin_sum / examples,
        "examples": float(examples),
    }


def clipped(value: str, limit: int = 180) -> str:
    """Keep printed examples readable."""
    value = " ".join(value.split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


@torch.no_grad()
def print_generation_examples(
    base_model,
    dpo_model,
    processor,
    dataset: DPOPreferenceDataset,
    collator: DPODataCollator,
    args: argparse.Namespace,
) -> None:
    """Print optional greedy continuations for a few validation prompts."""
    if args.examples == 0:
        return

    print("generation_examples")
    for index, record in enumerate(dataset.records[: args.examples], start=1):
        prompt_ids = collator._prompt_ids(record)
        chosen_ids = collator._record_ids(record, "chosen_ids", "chosen")
        if len(prompt_ids) >= args.max_length:
            prompt_ids = prompt_ids[-(args.max_length - 1) :]
        max_new_tokens = min(
            args.example_max_new_tokens,
            max(1, len(chosen_ids)),
            max(1, args.max_length - len(prompt_ids)),
        )

        print(f"example={index}")
        source_hanzi = record.get("source_hanzi")
        if isinstance(source_hanzi, str) and source_hanzi.strip():
            print(f"source_hanzi={clipped(source_hanzi)}")
        print(f"prompt={clipped(record['prompt'])}")
        print(f"chosen={clipped(record['chosen'])}")
        print(f"rejected={clipped(record['rejected'])}")

        for label, model in (("base", base_model), ("dpo", dpo_model)):
            output_ids = generate_tokens(
                model=model,
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                eos_id=processor.eos_id(),
                device=args.resolved_device,
            )
            continuation = decode_ids(processor, output_ids[len(prompt_ids) :])
            print(f"{label}_generation={clipped(continuation)}")


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate two checkpoints on the same preference dataset."""
    validate_args(args)
    device = resolve_device(args.device)
    args.resolved_device = device

    dataset = DPOPreferenceDataset(args.dpo_dataset)
    if args.num_examples is not None:
        dataset_for_loader = Subset(dataset, range(min(args.num_examples, len(dataset))))
    else:
        dataset_for_loader = dataset

    processor = load_sentencepiece(args.tokenizer)
    base_model, base_config, resolved_base, _ = load_checkpoint_model(
        args.base_checkpoint,
        device,
        eval_mode=True,
    )
    dpo_model, dpo_config, resolved_dpo, _ = load_checkpoint_model(
        args.dpo_checkpoint,
        device,
        eval_mode=True,
    )
    if base_config != dpo_config:
        raise ValueError("Base and DPO checkpoint configs differ")

    tokenizer_vocab_size = len(processor)
    if tokenizer_vocab_size != base_config.vocab_size:
        raise ValueError(
            "Tokenizer vocab size does not match checkpoint config: "
            f"tokenizer={tokenizer_vocab_size}, checkpoint={base_config.vocab_size}."
        )

    effective_max_length = min(args.max_length, base_config.block_size)
    if effective_max_length != args.max_length:
        print(
            "warning: clamping --max-length from "
            f"{args.max_length} to checkpoint block_size={base_config.block_size}"
        )
        args.max_length = effective_max_length

    collator = DPODataCollator(processor, max_length=args.max_length)
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    print(
        "dpo_evaluation_setup "
        f"device={device} "
        f"examples={len(dataset_for_loader):,} "
        f"base_checkpoint={resolved_base} "
        f"dpo_checkpoint={resolved_dpo}"
    )
    for label, model in (("base", base_model), ("dpo", dpo_model)):
        metrics = score_model(model, loader, device)
        print(
            f"model={label} "
            f"preference_accuracy={metrics['preference_accuracy']:.4f} "
            f"avg_margin={metrics['avg_margin']:.4f} "
            f"examples={int(metrics['examples'])}"
        )

    print_generation_examples(
        base_model,
        dpo_model,
        processor,
        dataset,
        collator,
        args,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and DPO pinyin-code checkpoints on preference pairs."
    )
    parser.add_argument("--dpo-dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--dpo-checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Print this many optional generation examples.",
    )
    parser.add_argument("--example-max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
