"""Hugging Face Transformers integration for the pinyin-code language model."""

from .configuration_pinyin_code import PinyinCodeConfig
from .modeling_pinyin_code import (
    PinyinCodeForCausalLM,
    PinyinCodeForSequenceClassification,
    PinyinCodeModel,
)
from .tokenization_hybrid_pinyin_code import HybridPinyinCodeTokenizer

try:
    from .tokenization_pinyin_code import EncodedMandarinTokenizer, PinyinCodeTokenizer
except ModuleNotFoundError as exc:
    if exc.name != "sentencepiece":
        raise
    EncodedMandarinTokenizer = None
    PinyinCodeTokenizer = None

__all__ = [
    "EncodedMandarinTokenizer",
    "HybridPinyinCodeTokenizer",
    "PinyinCodeConfig",
    "PinyinCodeForCausalLM",
    "PinyinCodeForSequenceClassification",
    "PinyinCodeModel",
    "PinyinCodeTokenizer",
]
