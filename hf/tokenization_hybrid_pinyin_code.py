"""Hybrid Jieba-word tokenizer for preprocessed pinyin-code text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "vocab.json"}
ENCODED_WORD_RE = re.compile(r"^(?:[A-Za-z][0-9])+$")
ATOM_RE = re.compile(r"^[A-Za-z][0-9]$")


class HybridPinyinCodeTokenizer(PreTrainedTokenizer):
    """Tokenize encoded Jieba words with whole-word lookup and atomic fallback.

    The tokenizer expects text that has already passed through the repository's
    pinyin-code preprocessing. Whitespace is used as word-boundary metadata and
    never becomes a token.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        strict_validation: bool = True,
        readable_decode: bool = False,
        **kwargs: Any,
    ) -> None:
        self.vocab_file = vocab_file
        self.vocab = self._load_vocab(vocab_file)
        self.ids_to_tokens = {token_id: token for token, token_id in self.vocab.items()}
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.strict_validation = strict_validation
        self.readable_decode = readable_decode

        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("unk_token", "<unk>")
        kwargs.setdefault("bos_token", "<s>")
        kwargs.setdefault("eos_token", "</s>")
        kwargs.setdefault("mask_token", "<mask>")
        kwargs.setdefault("add_bos_token", add_bos_token)
        kwargs.setdefault("add_eos_token", add_eos_token)
        kwargs.setdefault("strict_validation", strict_validation)
        kwargs.setdefault("readable_decode", readable_decode)
        super().__init__(**kwargs)

    @staticmethod
    def _load_vocab(vocab_file: str) -> dict[str, int]:
        with Path(vocab_file).open("r", encoding="utf-8") as handle:
            vocab = json.load(handle)
        if not isinstance(vocab, dict):
            raise ValueError(f"{vocab_file} must contain a JSON object")

        normalized: dict[str, int] = {}
        seen_ids: set[int] = set()
        for token, token_id in vocab.items():
            if not isinstance(token, str) or not isinstance(token_id, int):
                raise ValueError("vocab.json must map string tokens to integer ids")
            if token_id in seen_ids:
                raise ValueError(f"Duplicate token id in vocab.json: {token_id}")
            normalized[token] = token_id
            seen_ids.add(token_id)

        expected_ids = set(range(len(normalized)))
        if seen_ids != expected_ids:
            raise ValueError("vocab.json ids must be contiguous from 0")
        return normalized

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        vocab = dict(self.vocab)
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _handle_unsupported_token(self, token: str) -> list[str]:
        if self.strict_validation:
            raise ValueError(
                f"Unsupported or malformed pinyin-code token {token!r}. "
                "Expected a known special/preserved token or text matching "
                "^(?:[A-Za-z][0-9])+$."
            )
        return [self.unk_token]

    def _tokenize(self, text: str) -> list[str]:
        output: list[str] = []
        for item in text.split():
            if item in self.vocab:
                output.append(item)
            elif ENCODED_WORD_RE.fullmatch(item):
                output.extend(item[index : index + 2] for index in range(0, len(item), 2))
            else:
                output.extend(self._handle_unsupported_token(item))
        return output

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.unk_token_id)

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    def _token_is_atom(self, token: str) -> bool:
        return bool(ATOM_RE.fullmatch(token))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        if self.readable_decode:
            return " ".join(tokens)

        pieces: list[str] = []
        atom_buffer: list[str] = []
        for token in tokens:
            if self._token_is_atom(token):
                atom_buffer.append(token)
                continue
            if atom_buffer:
                pieces.append("".join(atom_buffer))
                atom_buffer.clear()
            pieces.append(token)
        if atom_buffer:
            pieces.append("".join(atom_buffer))
        return " ".join(pieces)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool | None = None,
        readable: bool | None = None,
        **kwargs: Any,
    ) -> str:
        if readable is None:
            return super().decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )

        previous = self.readable_decode
        self.readable_decode = readable
        try:
            return super().decode(
                token_ids,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
                **kwargs,
            )
        finally:
            self.readable_decode = previous

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        output = list(token_ids_0)
        if self.add_bos_token and self.bos_token_id is not None:
            output = [self.bos_token_id] + output
        if self.add_eos_token and self.eos_token_id is not None:
            output = output + [self.eos_token_id]
        if token_ids_1 is not None:
            output += list(token_ids_1)
            if self.add_eos_token and self.eos_token_id is not None:
                output.append(self.eos_token_id)
        return output

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            special_ids = set(self.all_special_ids)
            return [1 if token_id in special_ids else 0 for token_id in token_ids_0]

        mask = [0] * len(token_ids_0)
        if self.add_bos_token and self.bos_token_id is not None:
            mask = [1] + mask
        if self.add_eos_token and self.eos_token_id is not None:
            mask = mask + [1]
        if token_ids_1 is not None:
            mask += [0] * len(token_ids_1)
            if self.add_eos_token and self.eos_token_id is not None:
                mask.append(1)
        return mask

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
    ) -> list[int]:
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: str | None = None,
    ) -> tuple[str]:
        output_name = "vocab.json"
        if filename_prefix:
            output_name = f"{filename_prefix}-{output_name}"
        output_path = Path(save_directory) / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ordered = {
            token: token_id
            for token, token_id in sorted(self.vocab.items(), key=lambda item: item[1])
        }
        output_path.write_text(
            json.dumps(ordered, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return (str(output_path),)
