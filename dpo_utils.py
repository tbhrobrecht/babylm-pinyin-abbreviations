"""Shared helpers for optional DPO preference optimization.

DPO is used here as an offline post-pretraining stage for the existing causal
LM. The helpers in this module keep prompt tokens out of the preference loss and
reuse the repository's original PyTorch checkpoint format.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from train_model import ModelConfig, build_model, canonical_model_config, model_config_from_checkpoint


WHITESPACE_RE = re.compile(r"\s+")


def normalize_preference_text(value: str) -> str:
    """Normalize generated text before chosen/rejected equality checks."""
    return WHITESPACE_RE.sub(" ", value).strip()


def resolve_checkpoint_path(path: Path) -> Path:
    """Resolve either a checkpoint file or a directory containing best/last.pt."""
    if path.is_dir():
        for candidate_name in ("best.pt", "last.pt"):
            candidate = path / candidate_name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No best.pt or last.pt found in checkpoint directory: {path}")
    return path


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
    eval_mode: bool = True,
) -> tuple[nn.Module, ModelConfig, Path, dict[str, Any]]:
    """Load a train_model.py-compatible checkpoint."""
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {resolved_path}")

    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)
    if "model_config" not in checkpoint:
        raise KeyError(f"{resolved_path} does not contain `model_config`")
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"{resolved_path} does not contain `model_state_dict`")

    config = model_config_from_checkpoint(checkpoint)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    if eval_mode:
        model.eval()
    return model, config, resolved_path, checkpoint


def require_sentencepiece():
    """Import SentencePiece or stop with the same install hint as the main scripts."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `py -m pip install sentencepiece`."
        ) from exc

    return spm


def load_sentencepiece(tokenizer_path: Path):
    """Load the SentencePiece model used by the pretraining dataset."""
    spm = require_sentencepiece()
    return spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def tokenizer_pad_id(processor) -> int:
    """Choose a valid padding token id for right-padded DPO batches."""
    for method_name in ("pad_id", "eos_id", "unk_id", "bos_id"):
        token_id = getattr(processor, method_name)()
        if token_id >= 0:
            return int(token_id)
    return 0


def tokenizer_start_id(processor) -> int | None:
    """Return a valid BOS/EOS fallback for empty prompts, if available."""
    for method_name in ("bos_id", "eos_id"):
        token_id = getattr(processor, method_name)()
        if token_id >= 0:
            return int(token_id)
    return None


def encode_text(processor, text: str) -> list[int]:
    """Encode text with the repository's SentencePiece tokenizer."""
    return [int(token_id) for token_id in processor.encode(text, out_type=int)]


def decode_ids(processor, token_ids: list[int]) -> str:
    """Decode ids while tolerating older SentencePiece Python APIs."""
    if not token_ids:
        return ""
    return processor.decode(token_ids)


def validate_token_ids(token_ids: Any, field_name: str) -> list[int] | None:
    """Return a sanitized id list when a preference record stores token ids."""
    if token_ids is None:
        return None
    if not isinstance(token_ids, list):
        raise ValueError(f"`{field_name}` must be a list of integers when provided")
    sanitized: list[int] = []
    for token_id in token_ids:
        if not isinstance(token_id, int):
            raise ValueError(f"`{field_name}` contains a non-integer token id")
        if token_id < 0:
            raise ValueError(f"`{field_name}` contains a negative token id")
        sanitized.append(token_id)
    return sanitized


class DPOPreferenceDataset(Dataset):
    """Load JSONL prompt/chosen/rejected preference records."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._validate_record(record, line_number)
                self.records.append(record)

        if not self.records:
            raise ValueError(f"No DPO preference records found in {path}")

    def _validate_record(self, record: Mapping[str, Any], line_number: int) -> None:
        for field_name in ("prompt", "chosen", "rejected"):
            if field_name not in record:
                raise ValueError(f"Missing `{field_name}` at {self.path}:{line_number}")
            if not isinstance(record[field_name], str):
                raise ValueError(f"`{field_name}` must be a string at {self.path}:{line_number}")

        chosen = normalize_preference_text(record["chosen"])
        rejected = normalize_preference_text(record["rejected"])
        if not chosen:
            raise ValueError(f"Empty `chosen` at {self.path}:{line_number}")
        if not rejected:
            raise ValueError(f"Empty `rejected` at {self.path}:{line_number}")
        if chosen == rejected:
            raise ValueError(f"`chosen` and `rejected` are identical at {self.path}:{line_number}")

        for field_name in ("prompt_ids", "chosen_ids", "rejected_ids"):
            validate_token_ids(record.get(field_name), field_name)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


class DPODataCollator:
    """Tokenize and right-pad prompt/chosen/rejected records for DPO.

    The completion mask marks only target continuation tokens. During causal
    shifting, that mask is shifted with the labels so prompt log-probabilities
    are not included in the DPO objective.
    """

    def __init__(self, processor, max_length: int) -> None:
        if max_length < 2:
            raise ValueError("--max-length must be at least 2")
        self.processor = processor
        self.max_length = max_length
        self.pad_id = tokenizer_pad_id(processor)
        self.start_id = tokenizer_start_id(processor)

    def _record_ids(self, record: Mapping[str, Any], id_field: str, text_field: str) -> list[int]:
        token_ids = validate_token_ids(record.get(id_field), id_field)
        if token_ids is not None:
            return token_ids
        return encode_text(self.processor, record[text_field])

    def _prompt_ids(self, record: Mapping[str, Any]) -> list[int]:
        prompt_ids = self._record_ids(record, "prompt_ids", "prompt")
        if prompt_ids:
            return prompt_ids
        if self.start_id is not None:
            return [self.start_id]
        raise ValueError("DPO records require a non-empty prompt or a tokenizer BOS/EOS id")

    def _build_sequence(
        self,
        prompt_ids: list[int],
        completion_ids: list[int],
    ) -> tuple[list[int], list[bool], list[int]]:
        if not completion_ids:
            raise ValueError("DPO completions must contain at least one token")

        prompt = list(prompt_ids)
        if len(prompt) >= self.max_length:
            prompt = prompt[-(self.max_length - 1) :]

        available_completion_tokens = self.max_length - len(prompt)
        completion = list(completion_ids[:available_completion_tokens])
        if not completion:
            raise ValueError("No completion tokens remain after DPO max-length truncation")

        input_ids = prompt + completion
        if len(input_ids) > self.max_length:
            raise AssertionError("DPO sequence exceeded max_length after truncation")

        completion_mask = [False] * len(prompt) + [True] * len(completion)
        return input_ids, completion_mask, prompt

    def _pad(self, sequences: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(sequence) for sequence in sequences)
        input_ids = []
        attention_mask = []
        for sequence in sequences:
            pad_len = max_len - len(sequence)
            input_ids.append(sequence + [self.pad_id] * pad_len)
            attention_mask.append([1] * len(sequence) + [0] * pad_len)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.bool)

    def _pad_bool(self, masks: list[list[bool]], max_len: int) -> torch.Tensor:
        padded = [mask + [False] * (max_len - len(mask)) for mask in masks]
        return torch.tensor(padded, dtype=torch.bool)

    def __call__(self, records: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        prompt_sequences: list[list[int]] = []
        chosen_sequences: list[list[int]] = []
        rejected_sequences: list[list[int]] = []
        chosen_masks: list[list[bool]] = []
        rejected_masks: list[list[bool]] = []

        for record in records:
            prompt_ids = self._prompt_ids(record)
            chosen_ids = self._record_ids(record, "chosen_ids", "chosen")
            rejected_ids = self._record_ids(record, "rejected_ids", "rejected")

            chosen_input_ids, chosen_completion_mask, truncated_prompt = self._build_sequence(
                prompt_ids,
                chosen_ids,
            )
            rejected_input_ids, rejected_completion_mask, _ = self._build_sequence(
                truncated_prompt,
                rejected_ids,
            )

            prompt_sequences.append(truncated_prompt)
            chosen_sequences.append(chosen_input_ids)
            rejected_sequences.append(rejected_input_ids)
            chosen_masks.append(chosen_completion_mask)
            rejected_masks.append(rejected_completion_mask)

        prompt_input_ids, prompt_attention_mask = self._pad(prompt_sequences)
        chosen_input_ids, chosen_attention_mask = self._pad(chosen_sequences)
        rejected_input_ids, rejected_attention_mask = self._pad(rejected_sequences)

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_completion_mask": self._pad_bool(chosen_masks, chosen_input_ids.size(1)),
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_completion_mask": self._pad_bool(rejected_masks, rejected_input_ids.size(1)),
        }


def batch_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    """Move all tensor values in a DPO batch to the requested device."""
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Return summed log-probability over completion tokens only."""
    outputs = model(input_ids)
    if isinstance(outputs, torch.Tensor):
        logits = outputs
    elif hasattr(outputs, "logits"):
        logits = outputs.logits
    else:
        logits = outputs[0]

    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = input_ids[:, 1:].contiguous()
    shifted_completion_mask = completion_mask[:, 1:].contiguous()

    token_log_probs = F.log_softmax(shifted_logits, dim=-1)
    gathered = token_log_probs.gather(
        dim=-1,
        index=shifted_labels.unsqueeze(-1),
    ).squeeze(-1)

    mask = shifted_completion_mask.to(dtype=gathered.dtype)
    if torch.any(mask.sum(dim=1) <= 0):
        raise ValueError("Each DPO sequence must have at least one scored completion token")
    return (gathered * mask).sum(dim=1)


def dpo_loss_from_logps(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the direct preference optimization loss and policy margin."""
    pi_logratio = policy_chosen_logps - policy_rejected_logps
    ref_logratio = ref_chosen_logps - ref_rejected_logps
    loss = -F.logsigmoid(beta * (pi_logratio - ref_logratio)).mean()
    margin = pi_logratio.detach()
    return loss, margin


def checkpoint_payload(
    model: nn.Module,
    config: ModelConfig,
    epoch: int,
    global_step: int,
    validation_loss: float,
    best_loss: float,
    dpo_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a DPO checkpoint that remains compatible with generate.py."""
    config = canonical_model_config(config)
    return {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "architecture": config.architecture,
        "tokenizer_type": "jieba_atomic",
        "vocab_size": config.vocab_size,
        "epoch": epoch,
        "global_step": global_step,
        "validation_loss": validation_loss,
        "best_loss": best_loss,
        "dpo_config": dict(dpo_config),
    }


def jsonable_namespace(namespace: Any) -> dict[str, Any]:
    """Convert argparse namespaces containing Path values into JSON data."""
    payload: dict[str, Any] = {}
    for key, value in vars(namespace).items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload
