"""Smoke-test standard Transformers loading for the exported model folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


DEFAULT_MODEL_PATH = Path("hf_models") / "hf_full_chinese_gpu3"
DEFAULT_TEXT = "\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5\u3002"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify that a local path or HF repo ID loads via standard Auto classes."
    )
    parser.add_argument(
        "model_path",
        nargs="?",
        default=str(DEFAULT_MODEL_PATH),
        help="Local model folder or Hugging Face repo ID.",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text prompt used for the tokenizer/model forward smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    tokenizer(args.text)
    tokenizer(args.text, add_special_tokens=False)
    offset_batch = tokenizer(args.text, return_offsets_mapping=True)
    if "offset_mapping" not in offset_batch:
        raise RuntimeError("Tokenizer does not expose offset_mapping")
    if len(offset_batch["offset_mapping"]) != len(offset_batch["input_ids"]):
        raise RuntimeError("offset_mapping length does not match input_ids length")

    batch = tokenizer(
        [args.text, args.text],
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    base_model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True)
    base_model.eval()
    with torch.no_grad():
        base_out = base_model(**batch, output_hidden_states=True)
    if not hasattr(base_out, "last_hidden_state"):
        raise RuntimeError("AutoModel output does not expose last_hidden_state")
    if not getattr(base_out, "hidden_states", None):
        raise RuntimeError("AutoModel output does not expose hidden_states")

    model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
    model.eval()
    with torch.no_grad():
        out = model(**batch, output_hidden_states=True)

    if not hasattr(out, "logits"):
        raise RuntimeError("AutoModelForCausalLM output does not expose logits")
    if not getattr(out, "hidden_states", None):
        raise RuntimeError("AutoModelForCausalLM output does not expose hidden_states")
    expected_prefix = tuple(batch["input_ids"].shape)
    if tuple(out.logits.shape[:2]) != expected_prefix:
        raise RuntimeError(
            f"Unexpected logits prefix shape: got {tuple(out.logits.shape)}, "
            f"expected batch/sequence prefix {expected_prefix}"
        )
    if out.logits.shape[-1] != config.vocab_size:
        raise RuntimeError(
            f"Unexpected vocab dimension: got {out.logits.shape[-1]}, "
            f"expected {config.vocab_size}"
        )

    classifier = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        num_labels=3,
    )
    classifier.eval()
    labels = torch.tensor([0, 2], dtype=torch.long)
    with torch.no_grad():
        classifier_out = classifier(**batch, labels=labels)
    if tuple(classifier_out.logits.shape) != (2, 3):
        raise RuntimeError(
            f"Unexpected classifier logits shape: {tuple(classifier_out.logits.shape)}"
        )
    if classifier_out.loss is None or not torch.isfinite(classifier_out.loss):
        raise RuntimeError("Sequence-classification loss is not finite")

    print(config.model_type)
    print(tokenizer.__class__)
    print(base_out.last_hidden_state.shape)
    print(out.logits.shape)
    print(classifier_out.logits.shape)
    print("ok")


if __name__ == "__main__":
    main()
