"""SentencePiece tokenizer wrapper for pinyin-code Transformers models."""

from __future__ import annotations

import shutil
from pathlib import Path

import sentencepiece as spm
from transformers import PreTrainedTokenizer


class PinyinCodeTokenizer(PreTrainedTokenizer):
    """Slow tokenizer that preserves the existing SentencePiece model."""

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        **kwargs,
    ) -> None:
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token

        kwargs.setdefault("unk_token", self._piece_or_none(self.sp_model.unk_id()))
        kwargs.setdefault("bos_token", self._piece_or_none(self.sp_model.bos_id()))
        kwargs.setdefault("eos_token", self._piece_or_none(self.sp_model.eos_id()))
        kwargs.setdefault("pad_token", self._piece_or_none(self.sp_model.pad_id()))
        super().__init__(**kwargs)

    def _piece_or_none(self, token_id: int) -> str | None:
        if token_id is None or token_id < 0:
            return None
        return self.sp_model.id_to_piece(token_id)

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> dict[str, int]:
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token: str) -> int:
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        return self.sp_model.id_to_piece(index)

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self.sp_model.decode(tokens)

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

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        output_name = "tokenizer.model"
        if filename_prefix:
            output_name = f"{filename_prefix}-{output_name}"
        output_path = Path(save_directory) / output_name
        if Path(self.vocab_file).resolve() != output_path.resolve():
            shutil.copyfile(self.vocab_file, output_path)
        return (str(output_path),)
