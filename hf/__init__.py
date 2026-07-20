"""Hugging Face Transformers integration for the pinyin-code language model."""

from .configuration_pinyin_code import PinyinCodeConfig
from .modeling_pinyin_code import (
    PinyinCodeForCausalLM,
    PinyinCodeForSequenceClassification,
    PinyinCodeModel,
)
from .tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer

try:
    from .tokenization_pinyin_code import (
        EncodedMandarinTokenizer,
        EncodedMandarinTokenizerFast,
        PinyinCodeTokenizer,
    )
except ModuleNotFoundError as exc:
    if exc.name != "sentencepiece":
        raise
    EncodedMandarinTokenizer = None
    EncodedMandarinTokenizerFast = None
    PinyinCodeTokenizer = None

__all__ = [
    "EncodedMandarinTokenizer",
    "EncodedMandarinTokenizerFast",
    "HybridPinyinCodeTokenizer",
    "PinyinCodeConfig",
    "PinyinCodeForCausalLM",
    "PinyinCodeForSequenceClassification",
    "PinyinCodeModel",
    "PinyinCodeTokenizer",
]
