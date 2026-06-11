"""Scoring utilities for pinyin-code language models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


BERT_SPECIAL_PIECES = {
    "mask": "[MASK]",
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
    "sep": "[SEP]",
}


def _token_to_id(tokenizer, token: str) -> int | None:
    """Resolve a token id from either SentencePiece or Hugging Face tokenizers."""
    if hasattr(tokenizer, "piece_to_id"):
        token_id = int(tokenizer.piece_to_id(token))
        if token_id >= 0 and tokenizer.id_to_piece(token_id) == token:
            return token_id

    if hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return None
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        unk_token = getattr(tokenizer, "unk_token", None)
        if token_id == unk_token_id and token != unk_token:
            vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
            if token not in vocab:
                return None
        return int(token_id)

    return None


def _bert_special_ids(tokenizer) -> dict[str, int]:
    ids = {
        name: _token_to_id(tokenizer, piece)
        for name, piece in BERT_SPECIAL_PIECES.items()
    }
    missing = [BERT_SPECIAL_PIECES[name] for name, token_id in ids.items() if token_id is None]
    if missing:
        raise ValueError(
            "BERT pseudo-likelihood scoring requires tokenizer pieces "
            "[MASK], [PAD], [UNK], [CLS], and [SEP]. "
            f"Missing: {', '.join(missing)}."
        )
    return {name: int(token_id) for name, token_id in ids.items() if token_id is not None}


@torch.no_grad()
def score_sentence_pseudo_likelihood(
    model,
    input_ids: Sequence[int] | torch.Tensor,
    tokenizer,
    device: torch.device | str,
    normalize: str = "sum",
) -> float:
    """Return BERT pseudo-log-likelihood by masking one non-special token at a time.

    This is slower than GPT left-to-right scoring because it requires one forward
    pass per scored token position.
    """
    if normalize not in {"sum", "mean"}:
        raise ValueError("normalize must be either 'sum' or 'mean'")

    device = torch.device(device)
    special_ids = _bert_special_ids(tokenizer)
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("input_ids must be a single sequence with shape [seq_len] or [1, seq_len]")

    model.to(device)
    was_training = model.training
    model.eval()

    total_log_probability = torch.tensor(0.0, device=device)
    excluded_ids = set(special_ids.values())
    scored_tokens = 0
    for position in range(ids.shape[1]):
        original_token_id = int(ids[0, position].item())
        if original_token_id in excluded_ids:
            continue

        masked_ids = ids.clone()
        masked_ids[0, position] = special_ids["mask"]
        attention_mask = masked_ids.ne(special_ids["pad"]).to(torch.long)
        outputs = model(masked_ids, attention_mask=attention_mask)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        log_probs = F.log_softmax(logits[0, position], dim=-1)
        total_log_probability = total_log_probability + log_probs[original_token_id]
        scored_tokens += 1

    if was_training:
        model.train()

    if normalize == "mean" and scored_tokens:
        total_log_probability = total_log_probability / scored_tokens

    return float(total_log_probability.item())
