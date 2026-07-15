"""Generate text from a trained pinyin-code language model checkpoint."""

from __future__ import annotations

import argparse
import random
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

import torch
from torch.nn import functional as F

from train_model import build_model, model_config_from_checkpoint, output_logits


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
PINYIN_CODE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]\d(?:[A-Za-z]\d)*(?![A-Za-z0-9])"
)
SPECIAL_MARKER_RE = re.compile(r"<[A-Z_]+>")


def require_sentencepiece():
    """Import SentencePiece or stop with an install hint."""
    try:
        import sentencepiece as spm
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install it with `py -m pip install sentencepiece`."
        ) from exc

    return spm


def load_generation_tokenizer(tokenizer_path: Path):
    """Load either a SentencePiece model or the repository's hybrid tokenizer."""
    if tokenizer_path.is_dir() or tokenizer_path.name == "vocab.json":
        from hf.tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer

        load_path = tokenizer_path.parent if tokenizer_path.name == "vocab.json" else tokenizer_path
        return HybridPinyinCodeTokenizer.from_pretrained(load_path)

    spm = require_sentencepiece()
    return spm.SentencePieceProcessor(model_file=str(tokenizer_path))


def tokenizer_encode(processor, text: str) -> list[int]:
    """Encode text with either supported tokenizer API."""
    if hasattr(processor, "tokenize") and hasattr(processor, "convert_tokens_to_ids"):
        tokens = processor.tokenize(text)
        return [int(token_id) for token_id in processor.convert_tokens_to_ids(tokens)]
    return [int(token_id) for token_id in processor.encode(text, out_type=int)]


def tokenizer_decode(processor, token_ids: list[int]) -> str:
    """Decode token ids with either supported tokenizer API."""
    if not token_ids:
        return ""
    return processor.decode(token_ids)


def tokenizer_id(processor, name: str) -> int:
    """Return a special token id or -1 when unavailable."""
    method = getattr(processor, f"{name}_id", None)
    if callable(method):
        return int(method())
    value = getattr(processor, f"{name}_token_id", None)
    return int(value) if value is not None else -1


def preprocess_prompt(text: str, transliteration: str, use_jieba: bool) -> str:
    """Convert raw Mandarin text to the pinyin-code representation."""
    try:
        from preprocessing.preprocess import process_text, require_dependencies
    except ImportError as exc:
        raise SystemExit(
            "Raw prompt preprocessing requires jieba and pypinyin. "
            "Install dependencies with `py -m pip install -r requirements.txt`."
        ) from exc

    require_dependencies()
    return process_text(text, transliteration, use_jieba)


def prompt_contains_chinese(text: str) -> bool:
    """Return true when a prompt contains Mandarin/Hanzi characters."""
    return bool(CHINESE_RE.search(text))


def prompt_looks_preprocessed(text: str, transliteration: str) -> bool:
    """Return true for prompts that already look like model-side text."""
    if SPECIAL_MARKER_RE.search(text):
        return True
    if transliteration == "pinyin-code" and PINYIN_CODE_TOKEN_RE.search(text):
        return True
    return False


def prepare_prompt(
    text: str,
    raw_prompt: bool,
    code_prompt: bool,
    transliteration: str,
    use_jieba: bool,
) -> str:
    """Preprocess Mandarin prompts while preserving explicit pinyin-code input."""
    if code_prompt:
        return text
    if raw_prompt or prompt_contains_chinese(text):
        return preprocess_prompt(text, transliteration, use_jieba)
    if prompt_looks_preprocessed(text, transliteration):
        return text
    if text.strip():
        return preprocess_prompt(text, transliteration, use_jieba)
    return text


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    """Load a checkpoint saved by train_model.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = model_config_from_checkpoint(checkpoint)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
) -> torch.Tensor:
    """Sample one token id from the final-step logits."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))

    probabilities = F.softmax(logits, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    input_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    eos_id: int,
    device: torch.device,
) -> list[int]:
    """Autoregressively extend token ids."""
    if not input_ids:
        raise ValueError("At least one input token is required for generation")

    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    if hasattr(model, "generate"):
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "eos_token_id": eos_id if eos_id >= 0 else None,
            "pad_token_id": getattr(model.config, "pad_token_id", None),
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        if top_k is not None and top_k > 0:
            generation_kwargs["top_k"] = top_k
        attention_mask = torch.ones_like(ids)
        generated = model.generate(
            input_ids=ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
        return generated[0].tolist()

    block_size = getattr(model.config, "block_size", None)
    if block_size is None:
        block_size = getattr(model.config, "max_position_embeddings")
    for _ in range(max_new_tokens):
        context = ids[:, -block_size:]
        logits = output_logits(model(context))
        next_id = sample_next_token(logits[:, -1, :], temperature, top_k)
        ids = torch.cat((ids, next_id), dim=1)

        if eos_id >= 0 and next_id.item() == eos_id:
            break

    return ids[0].tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample pinyin-code text from a trained compact causal LM."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/pinyin-code-gpt-small/best.pt"),
        help="Path to a checkpoint produced by train_model.py.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("tokenizers/babylm_zho_pinyin_spm.model"),
        help="SentencePiece .model file used during dataset creation.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help=(
            "Prompt text. Mandarin/Hanzi input is automatically converted to "
            "BabyLM pinyin-code; existing pinyin-code prompts still work."
        ),
    )
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Treat --prompt as raw text and preprocess it first.",
    )
    parser.add_argument(
        "--code-prompt",
        action="store_true",
        help="Treat --prompt as already-preprocessed pinyin-code text.",
    )
    parser.add_argument(
        "--transliteration",
        choices=("pinyin-code", "pinyin-initial", "hanzi"),
        default="pinyin-code",
        help="Preprocessing mode to use for raw Mandarin prompts.",
    )
    parser.add_argument(
        "--jieba",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use jieba word segmentation for raw Mandarin prompts. Disable with "
            "--no-jieba for character-level preprocessing."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--completion-only",
        action="store_true",
        help="Print only newly generated text, excluding the prompt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    processor = load_generation_tokenizer(args.tokenizer)

    prompt = prepare_prompt(
        args.prompt,
        args.raw_prompt,
        args.code_prompt,
        args.transliteration,
        args.jieba,
    )
    input_ids = tokenizer_encode(processor, prompt) if prompt.strip() else []
    if not input_ids:
        start_id = tokenizer_id(processor, "bos")
        if start_id < 0:
            start_id = tokenizer_id(processor, "eos")
        if start_id < 0:
            raise SystemExit("Tokenizer has no BOS/EOS id; provide a non-empty --prompt.")
        input_ids = [start_id]

    model = load_model(args.checkpoint, device)
    output_ids = generate(
        model=model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tokenizer_id(processor, "eos"),
        device=device,
    )

    ids_to_decode = output_ids[len(input_ids) :] if args.completion_only else output_ids
    print(tokenizer_decode(processor, ids_to_decode))


if __name__ == "__main__":
    main()
