"""Train a small causal Transformer on the pinyin-code BabyLM dataset."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


SUPPORTED_ARCHITECTURES = ("gpt2", "qwen2")


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the compact GPT-style language model."""

    architecture: str = "gpt2"
    vocab_size: int = 8000
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    rms_norm_eps: float = 1.0e-6
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    tie_word_embeddings: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    unk_token_id: int | None = None
    attn_implementation: str | None = None


def normalize_architecture(value: str | None) -> str:
    """Return a supported lowercase architecture name."""
    architecture = (value or "gpt2").lower()
    if architecture not in SUPPORTED_ARCHITECTURES:
        supported = ", ".join(SUPPORTED_ARCHITECTURES)
        raise ValueError(
            f"Unsupported model architecture: {architecture}. "
            f"Supported architectures are: {supported}."
        )
    return architecture


def canonical_model_config(config: ModelConfig) -> ModelConfig:
    """Normalize model config values without changing legacy defaults."""
    architecture = normalize_architecture(config.architecture)
    if architecture == "qwen2":
        num_key_value_heads = config.num_key_value_heads or config.n_head
        intermediate_size = config.intermediate_size or 4 * config.n_embd
        return ModelConfig(
            **{
                **asdict(config),
                "architecture": architecture,
                "num_key_value_heads": num_key_value_heads,
                "intermediate_size": intermediate_size,
            }
        )
    return ModelConfig(**{**asdict(config), "architecture": architecture})


def model_config_from_mapping(mapping: dict[str, Any]) -> ModelConfig:
    """Load model config while tolerating legacy checkpoints without architecture."""
    allowed = set(ModelConfig.__dataclass_fields__)
    filtered = {key: value for key, value in mapping.items() if key in allowed}
    return canonical_model_config(ModelConfig(**filtered))


def model_config_from_checkpoint(checkpoint: dict[str, Any]) -> ModelConfig:
    """Load model config while honoring and validating checkpoint architecture metadata."""
    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint does not contain `model_config`")
    model_config = dict(checkpoint["model_config"])
    top_level_architecture = checkpoint.get("architecture")
    nested_architecture = model_config.get("architecture")
    if top_level_architecture is not None and nested_architecture is not None:
        if normalize_architecture(str(top_level_architecture)) != normalize_architecture(
            str(nested_architecture)
        ):
            raise ValueError(
                "Checkpoint architecture metadata conflict: "
                f"architecture={top_level_architecture!r}, "
                f"model_config.architecture={nested_architecture!r}."
            )
    if top_level_architecture is not None:
        model_config["architecture"] = top_level_architecture
    return model_config_from_mapping(model_config)


class JsonlTokenDataset(Dataset):
    """Load fixed-length JSONL token chunks produced by create_dataset.py."""

    def __init__(self, path: Path) -> None:
        self.examples: list[torch.Tensor] = []
        self.sequence_lengths: set[int] = set()
        self.max_token_id = -1

        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                input_ids = record["input_ids"]
                if not input_ids:
                    raise ValueError(f"Empty input_ids at {path}:{line_number}")
                if not all(isinstance(token_id, int) for token_id in input_ids):
                    raise ValueError(f"Non-integer token id at {path}:{line_number}")
                if min(input_ids) < 0:
                    raise ValueError(f"Negative token id at {path}:{line_number}")

                self.sequence_lengths.add(len(input_ids))
                self.max_token_id = max(self.max_token_id, max(input_ids))
                self.examples.append(torch.tensor(input_ids, dtype=torch.long))

        if not self.examples:
            raise ValueError(f"No examples found in {path}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.examples[index]


def binary_metadata_path(path: Path) -> Path:
    """Return the sidecar metadata path for a binary chunk file."""
    return path.with_suffix(path.suffix + ".meta.json")


class BinaryTokenDataset(Dataset):
    """Memory-map fixed-size int32 token chunks produced by create_dataset.py."""

    def __init__(self, path: Path) -> None:
        metadata_path = binary_metadata_path(path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Binary dataset metadata not found: {metadata_path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("format") != "pinyin-code-chunks-v1":
            raise ValueError(f"Unsupported binary dataset format in {metadata_path}")
        if metadata.get("dtype") != "int32_le":
            raise ValueError(f"Unsupported binary dataset dtype in {metadata_path}")

        self.path = path
        self.block_size = int(metadata["block_size"])
        self.num_examples = int(metadata["num_examples"])
        self.sequence_lengths = {self.block_size}
        self.max_token_id = int(metadata.get("max_token_id", -1))

        if self.num_examples <= 0:
            raise ValueError(f"No examples found in {path}")

        expected_bytes = self.num_examples * self.block_size * 4
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Binary dataset size mismatch for {path}: "
                f"expected={expected_bytes} bytes, actual={actual_bytes} bytes."
            )

        storage = torch.from_file(
            str(path),
            shared=False,
            size=self.num_examples * self.block_size,
            dtype=torch.int32,
        )
        self.examples = storage.view(self.num_examples, self.block_size)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.examples[index].to(torch.long)


def load_token_dataset(path: Path) -> Dataset:
    """Load a JSONL or binary token dataset."""
    if binary_metadata_path(path).exists():
        return BinaryTokenDataset(path)
    return JsonlTokenDataset(path)


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embd = x.shape
        q, k, v = self.qkv(x).split(embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """Transformer MLP block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PinyinCodeLanguageModel(nn.Module):
    """A compact GPT-style causal language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(TransformerBlock(config) for _ in range(config.n_layer)))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def get_input_embeddings(self) -> nn.Embedding:
        """Expose the token embedding through the Transformers-style name."""
        return self.token_embedding

    def get_output_embeddings(self) -> nn.Linear:
        """Expose the tied LM head through the Transformers-style name."""
        return self.lm_head

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
            )

        return logits, loss


def tokenizer_length(tokenizer: Any | None, fallback_vocab_size: int) -> int:
    """Return the tokenizer length when one is supplied, otherwise the config size."""
    if tokenizer is None:
        return fallback_vocab_size
    return len(tokenizer)


def build_gpt2_model(config: ModelConfig, tokenizer: Any | None = None) -> PinyinCodeLanguageModel:
    """Build the repository's original GPT-style model."""
    config = canonical_model_config(config)
    vocab_size = tokenizer_length(tokenizer, config.vocab_size)
    return PinyinCodeLanguageModel(ModelConfig(**{**asdict(config), "vocab_size": vocab_size}))


def build_qwen2_model(config: ModelConfig, tokenizer: Any | None = None) -> nn.Module:
    """Build a Qwen2 causal LM from scratch."""
    try:
        from transformers import Qwen2Config, Qwen2ForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Qwen2 support requires a Transformers version that provides "
            "Qwen2Config and Qwen2ForCausalLM."
        ) from exc

    config = canonical_model_config(config)
    if config.n_embd % config.n_head != 0:
        raise ValueError("Qwen2 configuration error: hidden_size must be divisible by num_attention_heads")
    if config.n_head % int(config.num_key_value_heads or config.n_head) != 0:
        raise ValueError(
            "Qwen2 configuration error: num_attention_heads must be divisible by "
            "num_key_value_heads"
        )

    qwen_kwargs = {
        "vocab_size": tokenizer_length(tokenizer, config.vocab_size),
        "hidden_size": config.n_embd,
        "intermediate_size": int(config.intermediate_size or 4 * config.n_embd),
        "num_hidden_layers": config.n_layer,
        "num_attention_heads": config.n_head,
        "num_key_value_heads": int(config.num_key_value_heads or config.n_head),
        "max_position_embeddings": config.block_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "attention_dropout": config.attention_dropout,
        "tie_word_embeddings": config.tie_word_embeddings,
        "bos_token_id": config.bos_token_id,
        "eos_token_id": config.eos_token_id,
        "pad_token_id": config.pad_token_id,
    }
    if config.attn_implementation is not None:
        qwen_kwargs["attn_implementation"] = config.attn_implementation
    qwen_config = Qwen2Config(**qwen_kwargs)
    model = Qwen2ForCausalLM(qwen_config)
    if model.get_input_embeddings().num_embeddings != qwen_config.vocab_size:
        model.resize_token_embeddings(qwen_config.vocab_size)
    return model


def build_model(config: ModelConfig, tokenizer: Any | None = None) -> nn.Module:
    """Build the selected causal LM architecture."""
    config = canonical_model_config(config)
    if config.architecture == "gpt2":
        return build_gpt2_model(config, tokenizer)
    if config.architecture == "qwen2":
        return build_qwen2_model(config, tokenizer)
    raise AssertionError(f"Unhandled architecture: {config.architecture}")


def model_max_sequence_length(config: ModelConfig) -> int:
    """Return the configured maximum causal context length."""
    return config.block_size


def mask_padding_labels(
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Hide padded positions from causal LM loss."""
    if attention_mask is None:
        return labels
    masked = labels.clone()
    masked[attention_mask == 0] = -100
    return masked


def causal_lm_outputs(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
) -> Any:
    """Run either supported causal LM and return its native output object."""
    if labels is not None:
        labels = mask_padding_labels(labels, attention_mask)
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def output_logits(outputs: Any) -> torch.Tensor:
    """Extract logits from tuple-style or Transformers-style outputs."""
    if isinstance(outputs, torch.Tensor):
        return outputs
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs[0]


def output_loss(outputs: Any) -> torch.Tensor | None:
    """Extract loss from tuple-style or Transformers-style outputs."""
    if hasattr(outputs, "loss"):
        return outputs.loss
    if isinstance(outputs, tuple) and len(outputs) > 1:
        return outputs[1]
    return None


def split_dataset(dataset: Dataset, validation_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    """Create deterministic train/validation splits."""
    if len(dataset) < 2:
        raise ValueError("At least two examples are required for train/validation split")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be greater than 0 and less than 1")

    validation_size = max(1, int(len(dataset) * validation_fraction))
    validation_size = min(validation_size, len(dataset) - 1)
    train_size = len(dataset) - validation_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, validation_size], generator=generator)


def validate_dataset_compatibility(dataset: Dataset, config: ModelConfig) -> None:
    """Fail early when dataset chunks cannot be consumed by the model config."""
    if dataset.max_token_id >= config.vocab_size:
        raise ValueError(
            "Dataset contains token id "
            f"{dataset.max_token_id}, but --vocab-size is {config.vocab_size}. "
            "Use the tokenizer vocabulary size used to create the dataset."
        )

    max_sequence_length = max(dataset.sequence_lengths)
    if max_sequence_length > model_max_sequence_length(config):
        raise ValueError(
            "Dataset contains examples of length "
            f"{max_sequence_length}, but --block-size is {model_max_sequence_length(config)}."
        )

    if len(dataset.sequence_lengths) > 1:
        lengths = ", ".join(str(length) for length in sorted(dataset.sequence_lengths))
        raise ValueError(
            "Dataset examples have varying lengths "
            f"({lengths}); use fixed-size chunks or add a padding collator."
        )


def checkpoint_architecture(checkpoint: dict) -> str:
    """Return checkpoint architecture, treating legacy checkpoints as GPT-style."""
    metadata_architecture = checkpoint.get("architecture")
    if metadata_architecture is not None:
        return normalize_architecture(str(metadata_architecture))
    model_config = checkpoint.get("model_config") or {}
    return normalize_architecture(model_config.get("architecture", "gpt2"))


def config_comparison_keys(config: ModelConfig) -> tuple[str, ...]:
    """Return checkpoint config fields that define the model shape."""
    if config.architecture == "qwen2":
        return (
            "architecture",
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_embd",
            "num_key_value_heads",
            "intermediate_size",
            "rms_norm_eps",
            "rope_theta",
            "attention_dropout",
            "tie_word_embeddings",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "attn_implementation",
        )
    return ("vocab_size", "block_size", "n_layer", "n_head", "n_embd", "dropout")


def validate_checkpoint_config(checkpoint: dict, config: ModelConfig, path: Path) -> None:
    """Ensure a resumed checkpoint matches the requested model shape."""
    checkpoint_config = checkpoint.get("model_config")
    if checkpoint_config is None:
        raise KeyError(f"{path} does not contain `model_config`")

    config = canonical_model_config(config)
    checkpoint_config_obj = model_config_from_mapping(checkpoint_config)
    checkpoint_arch = checkpoint_architecture(checkpoint)
    if checkpoint_arch != config.architecture:
        raise ValueError(
            "Checkpoint architecture does not match requested architecture: "
            f"checkpoint={checkpoint_arch}, requested={config.architecture}."
        )

    requested_config = asdict(config)
    saved_config = asdict(checkpoint_config_obj)
    mismatches = {
        key: (saved_config.get(key), requested_config.get(key))
        for key in config_comparison_keys(config)
        if saved_config.get(key) != requested_config.get(key)
    }
    if mismatches:
        details = ", ".join(
            f"{key}: checkpoint={old!r}, requested={new!r}"
            for key, (old, new) in mismatches.items()
        )
        raise ValueError(f"Checkpoint config does not match requested args: {details}")


def resolve_device(requested_device: str | None) -> torch.device:
    """Choose a training device and fail clearly for unavailable CUDA requests."""
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested, but this PyTorch environment cannot see a CUDA GPU. "
            "Install a CUDA-enabled PyTorch build or omit --device cuda to train on CPU."
        )
    return torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))


def configure_runtime(args: argparse.Namespace, device: torch.device) -> None:
    """Enable safe runtime knobs that improve training throughput."""
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    if device.type == "cuda" and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return total and trainable model parameter counts."""
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total_parameters, trainable_parameters


def model_summary_fields(config: ModelConfig, model: nn.Module) -> dict[str, Any]:
    """Return architecture summary fields for startup logging and tests."""
    config = canonical_model_config(config)
    total_parameters, trainable_parameters = count_parameters(model)
    summary = {
        "architecture": config.architecture,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "vocab_size": config.vocab_size,
        "layer_count": config.n_layer,
        "hidden_size": config.n_embd,
        "attention_heads": config.n_head,
        "max_sequence_length": config.block_size,
    }
    if config.architecture == "qwen2":
        summary["key_value_heads"] = config.num_key_value_heads
    return summary


def amp_dtype_from_name(name: str) -> torch.dtype:
    """Map CLI precision names to torch dtypes."""
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported AMP dtype: {name}")


def make_grad_scaler(enabled: bool):
    """Create a GradScaler across PyTorch versions."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def maybe_synchronize(device: torch.device) -> None:
    """Synchronize CUDA before timing/logging if needed."""
    if device.type == "cuda":
        torch.cuda.synchronize()


def resolve_metrics_log_path(args: argparse.Namespace) -> Path:
    """Return the JSONL metrics log path for this training run."""
    metrics_log = getattr(args, "metrics_log", None)
    if metrics_log is None:
        return args.output_dir / "metrics.jsonl"
    return Path(metrics_log)


def write_metrics_event(metrics_log: TextIO, event: dict[str, Any]) -> None:
    """Append one structured training metric event."""
    metrics_log.write(json.dumps(event, sort_keys=True) + "\n")
    metrics_log.flush()


def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.optim.Optimizer:
    """Create AdamW, using the fused CUDA implementation when available."""
    optimizer_kwargs = {
        "lr": args.learning_rate,
        "betas": (0.9, 0.95),
        "weight_decay": args.weight_decay,
    }
    if device.type == "cuda" and args.fused_adamw:
        optimizer_kwargs["fused"] = True
    try:
        return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        return torch.optim.AdamW(model.parameters(), **optimizer_kwargs)


def validate_training_args(args: argparse.Namespace) -> None:
    """Validate optimization options that interact with the training loop."""
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be greater than zero")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be greater than or equal to zero")
    if args.min_learning_rate < 0:
        raise ValueError("--min-learning-rate must be greater than or equal to zero")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate must be less than or equal to --learning-rate")
    if args.log_every <= 0:
        raise ValueError("--log-every must be greater than zero")


def optimizer_steps_per_epoch(loader: DataLoader, gradient_accumulation_steps: int) -> int:
    """Return the number of optimizer updates in one epoch."""
    return math.ceil(len(loader) / gradient_accumulation_steps)


def learning_rate_for_step(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    schedule: str,
) -> float:
    """Return the scheduled learning rate for a one-indexed optimizer step."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    if schedule == "constant":
        return base_lr
    if schedule != "cosine":
        raise ValueError(f"Unsupported LR schedule: {schedule}")

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(max((step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


def set_optimizer_lr(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set every optimizer param group to the scheduled learning rate."""
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    epoch: int,
    global_step: int,
    validation_loss: float,
    best_loss: float,
    save_optimizer: bool,
) -> dict:
    """Build a checkpoint, optionally including optimizer state for resuming."""
    config = canonical_model_config(config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "architecture": config.architecture,
        "tokenizer_type": "jieba_atomic",
        "vocab_size": config.vocab_size,
        "epoch": epoch,
        "global_step": global_step,
        "validation_loss": validation_loss,
        "best_loss": best_loss,
    }
    if save_optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    return checkpoint


def load_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: ModelConfig,
    device: torch.device,
) -> tuple[int, int, float]:
    """Load model/optimizer state and return start_epoch, global_step, best_loss."""
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_config(checkpoint, config, path)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{path} does not contain `model_state_dict`")

    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print(
            "warning: resume checkpoint has no optimizer state; "
            "continuing with a freshly initialized optimizer."
        )

    completed_epoch = int(checkpoint.get("epoch", 0))
    start_epoch = completed_epoch + 1
    global_step = int(checkpoint.get("global_step", 0))
    validation_loss = checkpoint.get("validation_loss", float("inf"))
    best_loss = float(checkpoint.get("best_loss", validation_loss))
    return start_epoch, global_step, best_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    """Return mean validation loss."""
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = causal_lm_outputs(model, batch, labels=batch)
            loss = output_loss(outputs)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(args: argparse.Namespace) -> None:
    """Train the language model and write checkpoints."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    validate_training_args(args)
    device = resolve_device(args.device)
    configure_runtime(args, device)

    dataset = load_token_dataset(args.dataset)
    config = canonical_model_config(ModelConfig(
        architecture=getattr(args, "architecture", "gpt2"),
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        num_key_value_heads=getattr(args, "num_key_value_heads", None),
        intermediate_size=getattr(args, "intermediate_size", None),
        rms_norm_eps=getattr(args, "rms_norm_eps", 1.0e-6),
        rope_theta=getattr(args, "rope_theta", 10000.0),
        attention_dropout=getattr(args, "attention_dropout", 0.0),
        tie_word_embeddings=getattr(args, "tie_word_embeddings", True),
        attn_implementation=getattr(args, "attn_implementation", None),
    ))
    validate_dataset_compatibility(dataset, config)

    if args.validation_dataset is not None:
        valid_dataset = load_token_dataset(args.validation_dataset)
        validate_dataset_compatibility(valid_dataset, config)
        train_dataset = dataset
    else:
        train_dataset, valid_dataset = split_dataset(dataset, args.validation_fraction, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    steps_per_epoch = optimizer_steps_per_epoch(
        train_loader,
        args.gradient_accumulation_steps,
    )
    total_optimizer_steps = max(1, steps_per_epoch * args.epochs)

    amp_dtype = amp_dtype_from_name(args.amp_dtype)
    use_amp = args.amp and device.type == "cuda"
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("bfloat16 AMP was requested, but this CUDA device does not support it.")
    scaler = make_grad_scaler(enabled=use_amp and amp_dtype == torch.float16)

    raw_model = build_model(config).to(device)
    optimizer = build_optimizer(raw_model, args, device)
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    if args.resume is not None:
        start_epoch, global_step, best_loss = load_resume_checkpoint(
            args.resume,
            raw_model,
            optimizer,
            config,
            device,
        )

    model: nn.Module = raw_model
    if args.compile:
        if not hasattr(torch, "compile"):
            raise SystemExit("This PyTorch version does not support torch.compile.")
        model = torch.compile(raw_model)

    tokens_per_train_epoch = len(train_dataset) * max(dataset.sequence_lengths)
    summary = model_summary_fields(config, raw_model)
    qwen_summary = ""
    if config.architecture == "qwen2":
        qwen_summary = f" key_value_heads={summary['key_value_heads']}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve_metrics_log_path(args)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    print(
        "training_setup "
        f"architecture={summary['architecture']} "
        f"device={device} "
        f"cuda_name={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'} "
        f"total_parameters={summary['total_parameters']:,} "
        f"trainable_parameters={summary['trainable_parameters']:,} "
        f"vocab_size={summary['vocab_size']:,} "
        f"layer_count={summary['layer_count']} "
        f"hidden_size={summary['hidden_size']} "
        f"attention_heads={summary['attention_heads']}"
        f"{qwen_summary} "
        f"max_sequence_length={summary['max_sequence_length']} "
        f"examples={len(dataset):,} "
        f"train_tokens_per_epoch={tokens_per_train_epoch:,} "
        f"gradient_accumulation_steps={args.gradient_accumulation_steps} "
        f"optimizer_steps_per_epoch={steps_per_epoch:,} "
        f"total_optimizer_steps={total_optimizer_steps:,} "
        f"amp={use_amp} "
        f"amp_dtype={args.amp_dtype if use_amp else 'none'} "
        f"tf32={args.tf32 and device.type == 'cuda'} "
        f"lr_schedule={args.lr_schedule} "
        f"warmup_steps={args.warmup_steps} "
        f"min_learning_rate={args.min_learning_rate:g} "
        f"compile={args.compile} "
        f"resume={args.resume or 'none'} "
        f"metrics_log={metrics_path}"
    )
    if device.type == "cpu":
        print("warning: training on CPU; this will be much slower than CUDA-based runs.")

    tokens_since_log = 0
    log_start = time.perf_counter()
    metrics_mode = "a" if args.resume is not None and metrics_path.exists() else "w"
    metrics_log = metrics_path.open(metrics_mode, encoding="utf-8")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_index = 0
        pending_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = causal_lm_outputs(model, batch, labels=batch)
                loss = output_loss(outputs)
            if loss is None:
                raise RuntimeError("Training loss was not computed")

            pending_loss += loss.item()
            accumulation_index += 1
            final_group_size = len(train_loader) % args.gradient_accumulation_steps
            if final_group_size and batch_index > len(train_loader) - final_group_size:
                accumulation_target = final_group_size
            else:
                accumulation_target = args.gradient_accumulation_steps
            scaled_loss = loss / accumulation_target
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (
                accumulation_index == accumulation_target
                or batch_index == len(train_loader)
            )
            if not should_step:
                tokens_since_log += batch.numel()
                continue

            next_step = global_step + 1
            current_lr = learning_rate_for_step(
                step=next_step,
                total_steps=total_optimizer_steps,
                base_lr=args.learning_rate,
                min_lr=args.min_learning_rate,
                warmup_steps=args.warmup_steps,
                schedule=args.lr_schedule,
            )
            set_optimizer_lr(optimizer, current_lr)

            step_completed = True
            if scaler.is_enabled():
                scale_before = scaler.get_scale()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                step_completed = scaler.get_scale() >= scale_before
            else:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
                optimizer.step()
            tokens_since_log += batch.numel()

            if step_completed:
                global_step += 1

            if step_completed and global_step % args.log_every == 0:
                maybe_synchronize(device)
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                tokens_per_second = tokens_since_log / elapsed
                train_loss = pending_loss / accumulation_index
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={train_loss:.4f} "
                    f"lr={current_lr:.6g} "
                    f"tokens_per_second={tokens_per_second:,.0f}"
                )
                write_metrics_event(
                    metrics_log,
                    {
                        "event": "train",
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": train_loss,
                        "learning_rate": current_lr,
                        "tokens_per_second": tokens_per_second,
                    },
                )
                tokens_since_log = 0
                log_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            accumulation_index = 0
            pending_loss = 0.0

        valid_loss = evaluate(model, valid_loader, device, use_amp, amp_dtype)
        print(f"epoch={epoch} validation_loss={valid_loss:.4f}")
        checkpoint_best_loss = min(best_loss, valid_loss)
        write_metrics_event(
            metrics_log,
            {
                "event": "validation",
                "epoch": epoch,
                "step": global_step,
                "validation_loss": valid_loss,
                "best_loss": checkpoint_best_loss,
            },
        )

        checkpoint = checkpoint_payload(
            raw_model,
            optimizer,
            config,
            epoch,
            global_step,
            valid_loss,
            checkpoint_best_loss,
            args.save_optimizer,
        )
        torch.save(checkpoint, args.output_dir / "last.pt")
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(checkpoint, args.output_dir / "best.pt")
    metrics_log.close()


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(description="Train a compact causal LM on pinyin-code chunks.")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/10k_babylm_zho_spm.jsonl"))
    parser.add_argument(
        "--validation-dataset",
        type=Path,
        default=None,
        help=(
            "Optional separate validation dataset created from held-out documents. "
            "When omitted, the training dataset is split randomly by chunk."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/pinyin-code-gpt-small"))
    parser.add_argument(
        "--architecture",
        choices=SUPPORTED_ARCHITECTURES,
        default="gpt2",
        type=normalize_architecture,
        help="Causal LM architecture to train. Defaults to the existing GPT-style model.",
    )
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--num-key-value-heads",
        type=int,
        default=None,
        help="Qwen2 grouped-query attention KV heads. Defaults to --n-head.",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="Qwen2 MLP intermediate size. Defaults to 4 * --n-embd.",
    )
    parser.add_argument("--rms-norm-eps", type=float, default=1.0e-6)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--attention-dropout", type=float, default=0.0)
    parser.add_argument(
        "--tie-word-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tie Qwen2 input and output embeddings.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
        help="Optional Transformers attention implementation for Qwen2.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of mini-batches to accumulate before each optimizer step.",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "constant"),
        default="cosine",
        help="Learning-rate schedule applied per optimizer step.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Number of optimizer steps used for linear LR warmup.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=0.0,
        help="Final LR for cosine decay.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--metrics-log",
        type=Path,
        default=None,
        help="JSONL metrics output path. Defaults to <output-dir>/metrics.jsonl.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from a checkpoint written by train_model.py.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use CUDA automatic mixed precision when training on GPU.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("float16", "bfloat16"),
        default="float16",
        help="CUDA autocast dtype used when AMP is enabled.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matmul/convolution on CUDA devices that support it.",
    )
    parser.add_argument(
        "--fused-adamw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW on CUDA when supported.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Opt into torch.compile for the training model.",
    )
    parser.add_argument(
        "--save-optimizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include optimizer state in checkpoints. Disabled by default to reduce checkpoint I/O.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
