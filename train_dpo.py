"""Optional DPO fine-tuning for the compact pinyin-code causal LM."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from dpo_utils import (
    DPODataCollator,
    DPOPreferenceDataset,
    batch_to_device,
    checkpoint_payload,
    dpo_loss_from_logps,
    jsonable_namespace,
    load_checkpoint_model,
    load_sentencepiece,
    sequence_log_probs,
)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve the requested training device."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but this PyTorch environment cannot see a CUDA GPU.")
    return torch.device(requested_device)


def set_seed(seed: int) -> None:
    """Set random seeds used by splitting and optimization."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_args(args: argparse.Namespace) -> None:
    """Validate optimization arguments before loading large objects."""
    if args.beta <= 0:
        raise ValueError("--beta must be greater than zero")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be greater than zero")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than zero")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be greater than zero")
    if not 0.0 <= args.eval_split < 1.0:
        raise ValueError("--eval-split must be greater than or equal to 0 and less than 1")
    if args.max_length < 2:
        raise ValueError("--max-length must be at least 2")
    if args.log_every <= 0:
        raise ValueError("--log-every must be greater than zero")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip must be greater than zero")


def split_preference_dataset(
    dataset: Dataset,
    eval_split: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    """Create a deterministic train/eval split for preference records."""
    if eval_split == 0.0 or len(dataset) < 2:
        return dataset, dataset

    eval_size = max(1, int(len(dataset) * eval_split))
    eval_size = min(eval_size, len(dataset) - 1)
    train_size = len(dataset) - eval_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, eval_size], generator=generator)


def compute_dpo_batch(
    policy_model,
    reference_model,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute DPO loss and policy chosen-minus-rejected margin for a batch."""
    policy_chosen = sequence_log_probs(
        policy_model,
        batch["chosen_input_ids"],
        batch["chosen_completion_mask"],
    )
    policy_rejected = sequence_log_probs(
        policy_model,
        batch["rejected_input_ids"],
        batch["rejected_completion_mask"],
    )

    with torch.no_grad():
        ref_chosen = sequence_log_probs(
            reference_model,
            batch["chosen_input_ids"],
            batch["chosen_completion_mask"],
        )
        ref_rejected = sequence_log_probs(
            reference_model,
            batch["rejected_input_ids"],
            batch["rejected_completion_mask"],
        )

    # DPO compares how much more the policy prefers chosen over rejected than
    # the frozen reference model did before preference tuning.
    return dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta,
    )


@torch.no_grad()
def evaluate(
    policy_model,
    reference_model,
    loader: DataLoader,
    device: torch.device,
    beta: float,
) -> dict[str, float]:
    """Return DPO loss, preference accuracy, and policy margin on a split."""
    policy_model.eval()
    reference_model.eval()
    loss_sum = 0.0
    margin_sum = 0.0
    correct = 0
    examples = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        loss, margin = compute_dpo_batch(policy_model, reference_model, batch, beta)
        batch_size = margin.numel()
        loss_sum += loss.item() * batch_size
        margin_sum += margin.sum().item()
        correct += int((margin > 0).sum().item())
        examples += batch_size

    if examples == 0:
        raise ValueError("Evaluation split is empty")

    return {
        "loss": loss_sum / examples,
        "preference_accuracy": correct / examples,
        "avg_margin": margin_sum / examples,
    }


def write_training_config(args: argparse.Namespace, output_dir: Path, base_checkpoint: Path) -> dict[str, Any]:
    """Persist the DPO run configuration next to checkpoints."""
    config = jsonable_namespace(args)
    config["resolved_base_checkpoint"] = str(base_checkpoint)
    (output_dir / "dpo_training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return config


def train(args: argparse.Namespace) -> None:
    """Run offline DPO fine-tuning from a pretrained causal LM checkpoint."""
    set_seed(args.seed)
    validate_args(args)
    device = resolve_device(args.device)

    dataset = DPOPreferenceDataset(args.dpo_dataset)
    processor = load_sentencepiece(args.tokenizer)

    policy_model, config, resolved_base_checkpoint, _ = load_checkpoint_model(
        args.base_checkpoint,
        device,
        eval_mode=False,
    )
    reference_model, reference_config, _, _ = load_checkpoint_model(
        args.base_checkpoint,
        device,
        eval_mode=True,
    )
    if reference_config != config:
        raise ValueError("Policy and reference checkpoint configs differ")

    tokenizer_vocab_size = len(processor)
    if tokenizer_vocab_size != config.vocab_size:
        raise ValueError(
            "Tokenizer vocab size does not match checkpoint config: "
            f"tokenizer={tokenizer_vocab_size}, checkpoint={config.vocab_size}."
        )

    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    if any(parameter.requires_grad for parameter in reference_model.parameters()):
        raise AssertionError("Reference model parameters must be frozen for DPO")

    effective_max_length = min(args.max_length, config.block_size)
    if effective_max_length != args.max_length:
        print(
            "warning: clamping --max-length from "
            f"{args.max_length} to checkpoint block_size={config.block_size}"
        )
        args.max_length = effective_max_length

    train_dataset, eval_dataset = split_preference_dataset(dataset, args.eval_split, args.seed)
    collator = DPODataCollator(processor, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if len(train_loader) == 0:
        raise ValueError("DPO training split is empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dpo_config = write_training_config(args, args.output_dir, resolved_base_checkpoint)
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    global_step = 0
    last_checkpoint: dict[str, Any] | None = None

    print(
        "dpo_training_setup "
        f"device={device} "
        f"examples={len(dataset):,} "
        f"train_examples={len(train_dataset):,} "
        f"eval_examples={len(eval_dataset):,} "
        f"beta={args.beta:g} "
        f"learning_rate={args.learning_rate:g} "
        f"batch_size={args.batch_size} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"max_length={args.max_length} "
        f"base_checkpoint={resolved_base_checkpoint}"
    )

    for epoch in range(1, args.epochs + 1):
        policy_model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_index = 0
        pending_loss = 0.0
        pending_margin = 0.0
        pending_batches = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = batch_to_device(batch, device)
            loss, margin = compute_dpo_batch(
                policy_model,
                reference_model,
                batch,
                args.beta,
            )

            pending_loss += loss.item()
            pending_margin += margin.mean().item()
            pending_batches += 1
            accumulation_index += 1

            final_group_size = len(train_loader) % args.gradient_accumulation_steps
            if final_group_size and batch_index > len(train_loader) - final_group_size:
                accumulation_target = final_group_size
            else:
                accumulation_target = args.gradient_accumulation_steps

            (loss / accumulation_target).backward()
            should_step = (
                accumulation_index == accumulation_target
                or batch_index == len(train_loader)
            )
            if not should_step:
                continue

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_index = 0
            global_step += 1

            if global_step % args.log_every == 0:
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={pending_loss / max(1, pending_batches):.4f} "
                    f"avg_margin={pending_margin / max(1, pending_batches):.4f}"
                )
                pending_loss = 0.0
                pending_margin = 0.0
                pending_batches = 0

        metrics = evaluate(policy_model, reference_model, eval_loader, device, args.beta)
        print(
            f"epoch={epoch} eval_loss={metrics['loss']:.4f} "
            f"preference_accuracy={metrics['preference_accuracy']:.4f} "
            f"avg_margin={metrics['avg_margin']:.4f}"
        )

        checkpoint = checkpoint_payload(
            policy_model,
            config,
            epoch,
            global_step,
            metrics["loss"],
            min(best_loss, metrics["loss"]),
            dpo_config,
        )
        torch.save(checkpoint, args.output_dir / "last.pt")
        last_checkpoint = checkpoint
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            torch.save(checkpoint, args.output_dir / "best.pt")

    if last_checkpoint is not None:
        torch.save(last_checkpoint, args.output_dir / "final.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained pinyin-code causal LM with offline DPO."
    )
    parser.add_argument("--dpo-dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--eval-split", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
