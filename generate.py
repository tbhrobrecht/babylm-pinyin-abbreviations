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

from train_model import ModelConfig, PinyinCodeLanguageModel, normalize_model_config_dict


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


def load_model(checkpoint_path: Path, device: torch.device) -> PinyinCodeLanguageModel:
    """Load a checkpoint saved by train_model.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config_dict = normalize_model_config_dict(checkpoint["model_config"])
    if config_dict["model_type"] == "bert":
        raise SystemExit(
            "BERT/MLM checkpoints are not autoregressive and cannot be used for free-form generation."
        )
    config = ModelConfig(**config_dict)
    model = PinyinCodeLanguageModel(config)
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
    model: PinyinCodeLanguageModel,
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
    for _ in range(max_new_tokens):
        context = ids[:, -model.config.block_size :]
        logits, _ = model(context)
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
    spm = require_sentencepiece()
    processor = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    prompt = prepare_prompt(
        args.prompt,
        args.raw_prompt,
        args.code_prompt,
        args.transliteration,
        args.jieba,
    )
    input_ids = processor.encode(prompt, out_type=int) if prompt.strip() else []
    if not input_ids:
        start_id = processor.bos_id()
        if start_id < 0:
            start_id = processor.eos_id()
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
        eos_id=processor.eos_id(),
        device=device,
    )

    ids_to_decode = output_ids[len(input_ids) :] if args.completion_only else output_ids
    print(processor.decode(ids_to_decode))


if __name__ == "__main__":
    main()
