"""Configuration for Transformers-compatible pinyin-code models."""

from __future__ import annotations

import functools
import os
import pathlib

from transformers import PretrainedConfig


_UTF8_PATH_OPEN_PATCH_MARKER = "_pinyin_code_utf8_path_open_patch"


def install_utf8_path_open_patch() -> None:
    """Default text-mode ``Path.open`` calls to UTF-8 when encoding is omitted.

    Some external Windows evaluation pipelines call ``Path.open("r")`` on
    UTF-8 JSONL data before specifying an encoding. The model is loaded before
    those datasets, so this narrow compatibility shim lets such pipelines read
    Mandarin evaluation files without repository-side changes. Explicit
    encodings and binary modes are left untouched.
    """
    current_open = pathlib.Path.open
    if getattr(current_open, _UTF8_PATH_OPEN_PATCH_MARKER, False):
        return

    @functools.wraps(current_open)
    def utf8_default_open(
        self,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        if encoding is None and "b" not in mode:
            encoding = "utf-8"
        return current_open(
            self,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    setattr(utf8_default_open, _UTF8_PATH_OPEN_PATCH_MARKER, True)
    pathlib.Path.open = utf8_default_open


class PinyinCodeConfig(PretrainedConfig):
    """Configuration for compact GPT-style and BERT-style pinyin-code models."""

    model_type = "pinyin_code"

    def __init__(
        self,
        vocab_size: int = 8000,
        block_size: int = 128,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        dropout: float = 0.1,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        unk_token_id: int | None = None,
        cls_token_id: int | None = None,
        sep_token_id: int | None = None,
        mask_token_id: int | None = None,
        training_model_type: str = "gpt",
        recommended_score_normalization: str | None = None,
        patch_pathlib_utf8_open: bool = False,
        **kwargs,
    ) -> None:
        if training_model_type not in {"gpt", "bert"}:
            raise ValueError("training_model_type must be either 'gpt' or 'bert'")
        if recommended_score_normalization is None:
            recommended_score_normalization = "mean" if training_model_type == "bert" else "sum"
        if recommended_score_normalization not in {"sum", "mean"}:
            raise ValueError("recommended_score_normalization must be either 'sum' or 'mean'")
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
            cls_token_id=cls_token_id,
            sep_token_id=sep_token_id,
            mask_token_id=mask_token_id,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.hidden_size = n_embd
        self.max_position_embeddings = block_size
        self.training_model_type = training_model_type
        self.recommended_score_normalization = recommended_score_normalization
        self.is_decoder = training_model_type == "gpt"
        self.is_encoder_decoder = False
        self.use_cache = False
        self.patch_pathlib_utf8_open = patch_pathlib_utf8_open
        if (
            patch_pathlib_utf8_open
            and os.environ.get("PINYIN_CODE_DISABLE_UTF8_OPEN_PATCH") != "1"
        ):
            install_utf8_path_open_patch()
