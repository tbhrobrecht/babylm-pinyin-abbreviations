"""Train compact GPT or BERT-style Transformers on pinyin-code BabyLM chunks."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split


@dataclass(frozen=True)
class ModelConfig:
    """Configuration shared by the compact GPT and BERT-style models."""

    vocab_size: int = 8000
    block_size: int = 128
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    model_type: str = "gpt"

    def __post_init__(self) -> None:
        if self.model_type not in {"gpt", "bert"}:
            raise ValueError("model_type must be either 'gpt' or 'bert'")


@dataclass(frozen=True)
class BertSpecialTokenIds:
    """Special token ids needed for BERT-style MLM corruption."""

    mask: int
    pad: int
    unk: int
    cls: int
    sep: int

    @property
    def all_special_ids(self) -> set[int]:
        return {self.mask, self.pad, self.unk, self.cls, self.sep}


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


BERT_SPECIAL_PIECES = {
    "mask": "[MASK]",
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
    "sep": "[SEP]",
}


def require_sentencepiece():
    """Import SentencePiece only when BERT tokenizer metadata is needed."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "BERT/MLM training requires sentencepiece to read special token ids. "
            "Install it with `py -m pip install sentencepiece`."
        ) from exc

    return spm


def sentencepiece_has_piece(processor, piece: str) -> bool:
    """Return true when a SentencePiece model contains exactly this piece."""
    try:
        token_id = int(processor.piece_to_id(piece))
        return token_id >= 0 and processor.id_to_piece(token_id) == piece
    except (IndexError, RuntimeError, TypeError, ValueError):
        return False


def validate_bert_special_tokens(processor) -> BertSpecialTokenIds:
    """Read and validate BERT special token ids from a SentencePiece processor."""
    missing = [
        piece
        for piece in BERT_SPECIAL_PIECES.values()
        if not sentencepiece_has_piece(processor, piece)
    ]
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(
            "BERT/MLM training requires tokenizer pieces "
            "[MASK], [PAD], [UNK], [CLS], and [SEP]. "
            f"Missing: {formatted}. Train a tokenizer with BERT special tokens "
            "or pass --tokenizer pointing to one that already has them."
        )

    return BertSpecialTokenIds(
        mask=int(processor.piece_to_id(BERT_SPECIAL_PIECES["mask"])),
        pad=int(processor.piece_to_id(BERT_SPECIAL_PIECES["pad"])),
        unk=int(processor.piece_to_id(BERT_SPECIAL_PIECES["unk"])),
        cls=int(processor.piece_to_id(BERT_SPECIAL_PIECES["cls"])),
        sep=int(processor.piece_to_id(BERT_SPECIAL_PIECES["sep"])),
    )


def load_bert_special_token_ids(tokenizer_path: Path) -> BertSpecialTokenIds:
    """Load BERT special token ids from a SentencePiece model file."""
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"BERT/MLM training requires --tokenizer, but file was not found: {tokenizer_path}"
        )
    spm = require_sentencepiece()
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return validate_bert_special_tokens(processor)


class MLMDataCollator:
    """Dynamically corrupt fixed token blocks with standard BERT 80/10/10 masking."""

    def __init__(
        self,
        special_token_ids: BertSpecialTokenIds,
        vocab_size: int,
        mlm_probability: float = 0.15,
    ) -> None:
        if not 0.0 < mlm_probability <= 1.0:
            raise ValueError("mlm_probability must be greater than 0 and less than or equal to 1")
        self.special_token_ids = special_token_ids
        self.vocab_size = vocab_size
        self.mlm_probability = mlm_probability

    def __call__(self, examples: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([example.to(torch.long) for example in examples])
        attention_mask = input_ids.ne(self.special_token_ids.pad).to(torch.long)
        labels = input_ids.clone()

        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in self.special_token_ids.all_special_ids:
            special_tokens_mask |= input_ids.eq(token_id)

        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, 0.0)
        masked_indices = torch.bernoulli(probability_matrix).to(torch.bool)

        if not torch.any(masked_indices):
            eligible_indices = (~special_tokens_mask).nonzero(as_tuple=False)
            if len(eligible_indices) > 0:
                chosen = eligible_indices[torch.randint(len(eligible_indices), (1,)).item()]
                masked_indices[chosen[0], chosen[1]] = True

        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).to(torch.bool) & masked_indices
        )
        input_ids[indices_replaced] = self.special_token_ids.mask

        indices_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).to(torch.bool)
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(self.vocab_size, input_ids.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


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


class BidirectionalSelfAttention(nn.Module):
    """Multi-head self-attention for encoder-only MLM training."""

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

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, embd = x.shape
        q, k, v = self.qkv(x).split(embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        attn_mask = None
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "attention_mask must have shape "
                    f"{(batch_size, seq_len)}, got {tuple(attention_mask.shape)}"
                )
            # BERT sees both left and right context; this mask only removes padding keys.
            attn_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
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
    """Pre-norm GPT Transformer block."""

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


class EncoderTransformerBlock(nn.Module):
    """Pre-norm BERT-style encoder block without causal masking."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
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

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
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


class BabyBertForMaskedLM(nn.Module):
    """A compact BERT-style encoder trained with masked language modeling."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(EncoderTransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.mlm_transform = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.GELU(),
            nn.LayerNorm(config.n_embd),
        )
        self.mlm_decoder = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.mlm_decoder.weight = self.token_embedding.weight
        self.mlm_bias = nn.Parameter(torch.zeros(config.vocab_size))

        self.apply(self._init_weights)

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
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"Sequence length {seq_len} exceeds block size {self.config.block_size}")

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=input_ids.device)

        positions = torch.arange(seq_len, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        hidden_states = self.ln_f(x)
        logits = self.mlm_decoder(self.mlm_transform(hidden_states)) + self.mlm_bias

        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError(
                    f"labels must have shape {tuple(input_ids.shape)}, got {tuple(labels.shape)}"
                )
            if torch.any(labels != -100):
                loss = F.cross_entropy(
                    logits.contiguous().view(-1, logits.size(-1)),
                    labels.contiguous().view(-1),
                    ignore_index=-100,
                )
            else:
                loss = logits.sum() * 0.0

        return logits, loss


def build_model(config: ModelConfig) -> nn.Module:
    """Instantiate the model architecture requested by the config."""
    if config.model_type == "gpt":
        return PinyinCodeLanguageModel(config)
    if config.model_type == "bert":
        return BabyBertForMaskedLM(config)
    raise ValueError(f"Unsupported model_type: {config.model_type}")


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
    if max_sequence_length > config.block_size:
        raise ValueError(
            "Dataset contains examples of length "
            f"{max_sequence_length}, but --block-size is {config.block_size}."
        )

    if len(dataset.sequence_lengths) > 1:
        lengths = ", ".join(str(length) for length in sorted(dataset.sequence_lengths))
        raise ValueError(
            "Dataset examples have varying lengths "
            f"({lengths}); use fixed-size chunks or add a padding collator."
        )


def normalize_model_config_dict(config_dict: dict) -> dict:
    """Treat older checkpoints without model_type as GPT checkpoints."""
    normalized = dict(config_dict)
    normalized.setdefault("model_type", "gpt")
    return normalized


def validate_checkpoint_config(checkpoint: dict, config: ModelConfig, path: Path) -> None:
    """Ensure a resumed checkpoint matches the requested model shape."""
    checkpoint_config = checkpoint.get("model_config")
    if checkpoint_config is None:
        raise KeyError(f"{path} does not contain `model_config`")

    checkpoint_config = normalize_model_config_dict(checkpoint_config)
    requested_config = asdict(config)
    mismatches = {
        key: (checkpoint_config.get(key), value)
        for key, value in requested_config.items()
        if checkpoint_config.get(key) != value
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


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable model parameters."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


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
    model_type = getattr(args, "model_type", "gpt")
    if model_type not in {"gpt", "bert"}:
        raise ValueError("--model-type must be either 'gpt' or 'bert'")
    mlm_probability = getattr(args, "mlm_probability", 0.15)
    if model_type == "bert" and not 0.0 < mlm_probability <= 1.0:
        raise ValueError("--mlm-probability must be greater than 0 and less than or equal to 1")
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


def objective_name(model_type: str) -> str:
    """Return the training objective name for logging."""
    if model_type == "bert":
        return "masked_language_modeling"
    return "causal_language_modeling"


def move_batch_to_device(batch, device: torch.device):
    """Move either a GPT tensor batch or a BERT MLM dict batch to the device."""
    non_blocking = device.type == "cuda"
    if isinstance(batch, dict):
        return {
            key: value.to(device, non_blocking=non_blocking)
            for key, value in batch.items()
        }
    return batch.to(device, non_blocking=non_blocking)


def batch_token_count(batch) -> int:
    """Return the number of input tokens represented by a training batch."""
    if isinstance(batch, dict):
        return int(batch["input_ids"].numel())
    return int(batch.numel())


def forward_for_objective(
    model: nn.Module,
    batch,
    model_type: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the correct objective-specific forward pass."""
    if model_type == "bert":
        return model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
    return model(batch, labels=batch)


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
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
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
    model_type: str,
) -> float:
    """Return mean validation loss."""
    model.eval()
    losses: list[float] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            _, loss = forward_for_objective(model, batch, model_type)
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def train(args: argparse.Namespace) -> None:
    """Train the language model and write checkpoints."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    validate_training_args(args)
    model_type = getattr(args, "model_type", "gpt")
    mlm_probability = getattr(args, "mlm_probability", 0.15)
    device = resolve_device(args.device)
    configure_runtime(args, device)

    dataset = load_token_dataset(args.dataset)
    config = ModelConfig(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        model_type=model_type,
    )
    validate_dataset_compatibility(dataset, config)
    bert_special_ids = None
    if model_type == "bert":
        tokenizer_path = getattr(args, "tokenizer", Path("tokenizers/babylm_zho_pinyin_spm.model"))
        bert_special_ids = load_bert_special_token_ids(tokenizer_path)
        max_special_id = max(bert_special_ids.all_special_ids)
        if max_special_id >= config.vocab_size:
            raise ValueError(
                "Tokenizer special token id "
                f"{max_special_id} is outside --vocab-size {config.vocab_size}."
            )

    if args.validation_dataset is not None:
        valid_dataset = load_token_dataset(args.validation_dataset)
        validate_dataset_compatibility(valid_dataset, config)
        train_dataset = dataset
    else:
        train_dataset, valid_dataset = split_dataset(dataset, args.validation_fraction, args.seed)

    collate_fn = None
    if model_type == "bert":
        assert bert_special_ids is not None
        collate_fn = MLMDataCollator(
            special_token_ids=bert_special_ids,
            vocab_size=config.vocab_size,
            mlm_probability=mlm_probability,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_fn,
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
    print(
        "training_setup "
        f"model_type={model_type} "
        f"objective={objective_name(model_type)} "
        f"device={device} "
        f"cuda_name={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'} "
        f"parameters={count_parameters(raw_model):,} "
        f"mask_probability={mlm_probability if model_type == 'bert' else 'none'} "
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
        f"resume={args.resume or 'none'}"
    )
    if device.type == "cpu":
        print("warning: training on CPU; this will be much slower than CUDA-based runs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokens_since_log = 0
    log_start = time.perf_counter()

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accumulation_index = 0
        pending_loss = 0.0
        for batch_index, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                _, loss = forward_for_objective(model, batch, model_type)
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
                tokens_since_log += batch_token_count(batch)
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
            tokens_since_log += batch_token_count(batch)

            if step_completed:
                global_step += 1

            if step_completed and global_step % args.log_every == 0:
                maybe_synchronize(device)
                elapsed = max(time.perf_counter() - log_start, 1e-9)
                tokens_per_second = tokens_since_log / elapsed
                print(
                    f"epoch={epoch} step={global_step} "
                    f"train_loss={pending_loss / accumulation_index:.4f} "
                    f"lr={current_lr:.6g} "
                    f"tokens_per_second={tokens_per_second:,.0f}"
                )
                tokens_since_log = 0
                log_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            accumulation_index = 0
            pending_loss = 0.0

        valid_loss = evaluate(model, valid_loader, device, use_amp, amp_dtype, model_type)
        if model_type == "bert":
            print(f"epoch={epoch} validation_loss={valid_loss:.4f} validation_mlm_loss={valid_loss:.4f}")
        else:
            print(
                f"epoch={epoch} validation_loss={valid_loss:.4f} "
                f"validation_causal_lm_loss={valid_loss:.4f}"
            )
        checkpoint_best_loss = min(best_loss, valid_loss)

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


def parse_args() -> argparse.Namespace:
    """Parse training options."""
    parser = argparse.ArgumentParser(
        description="Train a compact GPT causal LM or BERT MLM on pinyin-code chunks."
    )
    parser.add_argument(
        "--model-type",
        choices=("gpt", "bert"),
        default="gpt",
        help="Model architecture/objective to train. Defaults to the existing GPT causal LM.",
    )
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
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizers/babylm_zho_pinyin_spm.model"),
        help=(
            "SentencePiece .model used to validate BERT special token ids. "
            "Only required when --model-type bert."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/pinyin-code-gpt-small"))
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--n-embd", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--mlm-probability",
        type=float,
        default=0.15,
        help="Fraction of eligible tokens selected for BERT masked-language modeling.",
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
