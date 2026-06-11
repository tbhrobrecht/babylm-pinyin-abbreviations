"""SentencePiece tokenizer wrapper for pinyin-code Transformers models."""

from __future__ import annotations

import logging
import math
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import sentencepiece as spm
from transformers import PreTrainedTokenizer


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CHINESE_SPAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
PINYIN_CODE_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z]\d(?:[A-Za-z]\d)*(?![A-Za-z0-9])"
)
SPECIAL_MARKER_RE = re.compile(r"<[A-Z_]+>")
PUNCTUATION = set(
    "\u3002\uff0c\u3001\uff1f\uff01\uff1a\uff1b.,?!:;()[]{}<>\u300a\u300b"
    "\u3010\u3011\u201c\u201d\"'\u2018\u2019\u300c\u300d\u300e\u300f"
    "\u2014-~\u2026/\\"
)
LATIN_LETTER = (
    r"A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff"
    r"\u0100-\u017f\u0180-\u024f\u0250-\u02af"
)
LATIN_ALNUM_PATTERN = (
    rf"(?:[{LATIN_LETTER}][{LATIN_LETTER}0-9]*"
    rf"(?:[-_][{LATIN_LETTER}0-9]+)*|"
    rf"[0-9]+[{LATIN_LETTER}][{LATIN_LETTER}0-9]*"
    rf"(?:[-_][{LATIN_LETTER}0-9]+)*)"
)
LATIN_ALNUM_RE = re.compile(LATIN_ALNUM_PATTERN)
URL_RE = re.compile(r"\b(?:https?://\S*|www\.\S+)", flags=re.I)
DISCARDED_UNICODE_CATEGORIES = {"Cc", "Cf", "Co", "Cs", "Cn"}
TOKEN_RE = re.compile(
    r"<[A-Z_]+>|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|"
    rf"{LATIN_ALNUM_PATTERN}|"
    r"\S"
)
LABELS = {
    "\u9898\u5e72": "<QUESTION>",
    "\u9009\u9879": "<OPTIONS>",
    "\u7b54\u6848": "<ANSWER>",
    "\u89e3\u6790": "<EXPLANATION>",
}
PINYIN_FORMAT_ALIASES = {
    "code": "pinyin-code",
    "codes": "pinyin-code",
    "pinyin-code": "pinyin-code",
    "initial": "pinyin-initial",
    "initials": "pinyin-initial",
    "pinyin-initial": "pinyin-initial",
    "hanzi": "hanzi",
}


def latin_token_to_model_token(token: str) -> str:
    upper = token.upper()
    return upper if upper in {"A", "B", "C", "D"} else token.lower()


def should_preserve_fallback_token(token: str) -> bool:
    if token == "\ufffd":
        return False
    for char in token:
        category = unicodedata.category(char)
        if category in DISCARDED_UNICODE_CATEGORIES:
            return False
        if category[0] not in {"L", "P", "S"}:
            return False
    return True


class PinyinCodeTokenizer(PreTrainedTokenizer):
    """Slow tokenizer that preserves the existing SentencePiece model."""

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        add_bos_token: bool = False,
        add_eos_token: bool = False,
        transliteration: str = "pinyin-code",
        pinyin_format: str | None = None,
        use_jieba: bool = True,
        jieba: bool | None = None,
        **kwargs,
    ) -> None:
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.transliteration = self._normalize_transliteration(
            pinyin_format or transliteration
        )
        self.use_jieba = use_jieba if jieba is None else jieba

        kwargs.setdefault("unk_token", self._piece_or_none(self.sp_model.unk_id()))
        kwargs.setdefault("bos_token", self._piece_or_none(self.sp_model.bos_id()))
        kwargs.setdefault("eos_token", self._piece_or_none(self.sp_model.eos_id()))
        kwargs.setdefault("pad_token", self._piece_or_none(self.sp_model.pad_id()))
        cls_token_id = self._piece_to_id_or_none("[CLS]")
        sep_token_id = self._piece_to_id_or_none("[SEP]")
        mask_token_id = self._piece_to_id_or_none("[MASK]")
        if cls_token_id is not None:
            kwargs.setdefault("cls_token", self.sp_model.id_to_piece(cls_token_id))
        if sep_token_id is not None:
            kwargs.setdefault("sep_token", self.sp_model.id_to_piece(sep_token_id))
        if mask_token_id is not None:
            kwargs.setdefault("mask_token", self.sp_model.id_to_piece(mask_token_id))
        kwargs.setdefault("transliteration", self.transliteration)
        kwargs.setdefault("pinyin_format", self.transliteration)
        kwargs.setdefault("use_jieba", self.use_jieba)
        kwargs.setdefault("jieba", self.use_jieba)
        super().__init__(**kwargs)

    def _normalize_transliteration(self, value: str) -> str:
        normalized = PINYIN_FORMAT_ALIASES.get(value.lower())
        if normalized is None:
            allowed = ", ".join(sorted(set(PINYIN_FORMAT_ALIASES.values())))
            raise ValueError(f"Unsupported transliteration {value!r}; choose from {allowed}")
        return normalized

    def _piece_or_none(self, token_id: int) -> str | None:
        if token_id is None or token_id < 0:
            return None
        return self.sp_model.id_to_piece(token_id)

    def _piece_to_id_or_none(self, piece: str) -> int | None:
        token_id = int(self.sp_model.piece_to_id(piece))
        if token_id < 0 or self.sp_model.id_to_piece(token_id) != piece:
            return None
        return token_id

    def _looks_preprocessed(self, text: str) -> bool:
        if SPECIAL_MARKER_RE.search(text):
            return True
        if self.transliteration == "pinyin-code" and PINYIN_CODE_TOKEN_RE.search(text):
            return True
        return False

    def _preprocess_raw_text(self, text: str) -> str:
        if not CHINESE_RE.search(text) and self._looks_preprocessed(text):
            return text
        try:
            from preprocessing.preprocess import (
                hanzi_to_encoded,
                process_text,
                require_dependencies,
            )
        except ImportError:
            return self._fallback_process_text(text)

        require_dependencies()
        if self.transliteration == "pinyin-code":
            return hanzi_to_encoded(text, self.use_jieba)
        return process_text(text, self.transliteration, self.use_jieba)

    def _fallback_process_text(self, text: str) -> str:
        if self.use_jieba:
            try:
                import jieba
            except ImportError as exc:
                raise ImportError(
                    "Tokenizing raw Mandarin benchmark text with jieba segmentation "
                    "requires jieba. Install the model dependencies before running "
                    "lm_eval."
                ) from exc

            jieba.setLogLevel(logging.WARNING)
        else:
            jieba = None

        if self.transliteration != "hanzi":
            try:
                from pypinyin import Style, pinyin
            except ImportError as exc:
                raise ImportError(
                    "Tokenizing raw Mandarin benchmark text as pinyin requires pypinyin. "
                    "Install the model dependencies before running lm_eval."
                ) from exc


        def normalize_text(value: str) -> str:
            value = unicodedata.normalize("NFKC", value)
            value = URL_RE.sub(" <URL> ", value)
            value = re.sub(r"\$\$.*?\$\$", " <MATH> ", value, flags=re.DOTALL)
            value = re.sub(r"[\uff08(]\s*[\uff09)]", " <BLANK> ", value)
            for label, marker in LABELS.items():
                value = re.sub(rf"{label}\s*[:\uff1a]", f" {marker} ", value)
            value = re.sub(
                rf"(?<![{LATIN_LETTER}])yes(?![{LATIN_LETTER}])",
                " <YES> ",
                value,
                flags=re.I,
            )
            value = re.sub(
                rf"(?<![{LATIN_LETTER}])no(?![{LATIN_LETTER}])",
                " <NO> ",
                value,
                flags=re.I,
            )
            value = re.sub(
                rf"(?<![{LATIN_LETTER}])[ABCD](?=\s*[:\uff1a.\uff0e\u3001\)])",
                r" \g<0> ",
                value,
            )
            value = re.sub(
                rf"(?<![{LATIN_LETTER}0-9])[-+]?\d+(?:[.,]\d+)*(?:%|\uff05)?"
                rf"(?![{LATIN_LETTER}0-9])",
                " <NUM> ",
                value,
            )
            value = value.replace("\uff08", "(").replace("\uff09", ")")
            return re.sub(r"\s+", " ", value).strip()

        def split_tone3_syllable(syllable: str) -> tuple[str, int]:
            match = re.fullmatch(r"([a-z\u00fcv]+)([1-5]?)", syllable.lower())
            if not match:
                return syllable, 5
            plain, tone = match.groups()
            return plain, int(tone or "5")

        def length_digit_offset(syllable: str) -> int:
            return min(max(len(syllable), 1), 5) - 1

        def syllable_to_initial_code(syllable: str) -> str:
            plain, tone = split_tone3_syllable(syllable)
            if not plain:
                return ""
            tone_offset = 5 if tone in {3, 4, 5} else 0
            digit = tone_offset + length_digit_offset(plain)
            initial = plain[0].upper() if tone in {1, 3, 5} else plain[0].lower()
            return f"{initial}{digit}"

        def syllable_to_initial_letter(syllable: str) -> str:
            plain, _ = split_tone3_syllable(syllable)
            return plain[:1].lower()

        def convert_word(word: str) -> str:
            if self.transliteration == "hanzi":
                return word
            syllables = pinyin(word, style=Style.TONE3, heteronym=False, errors="ignore")
            if self.transliteration == "pinyin-code":
                codes = [
                    syllable_to_initial_code(item[0])
                    for item in syllables
                    if item and item[0]
                ]
                return "".join(code for code in codes if code)
            initials = [
                syllable_to_initial_letter(item[0])
                for item in syllables
                if item and item[0]
            ]
            return "".join(initial for initial in initials if initial)

        def tokenize_chinese_span(value: str) -> list[str]:
            tokens = []
            words = jieba.cut(value, cut_all=False) if self.use_jieba else value
            for word in words:
                word = word.strip()
                if word and CHINESE_SPAN_RE.search(word):
                    token = convert_word(word)
                    if token:
                        tokens.append(token)
            return tokens

        tokens = []
        for part in TOKEN_RE.findall(normalize_text(text)):
            if part.startswith("<") and part.endswith(">"):
                tokens.append(part)
            elif CHINESE_SPAN_RE.fullmatch(part):
                tokens.extend(tokenize_chinese_span(part))
            elif part in PUNCTUATION:
                tokens.append(part)
            elif LATIN_ALNUM_RE.fullmatch(part):
                tokens.append(latin_token_to_model_token(part))
            elif part.isdigit():
                tokens.append("<NUM>")
            elif should_preserve_fallback_token(part):
                tokens.append(part.lower())

        return " ".join(tokens)

    def _preprocess_tokenizer_input(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            return self._preprocess_raw_text(value)
        if isinstance(value, tuple):
            return tuple(self._preprocess_tokenizer_input(item) for item in value)
        if isinstance(value, list):
            return [self._preprocess_tokenizer_input(item) for item in value]
        return value

    def _non_content_token_ids(self) -> set[int]:
        return {
            token_id
            for token_id in (
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.cls_token_id,
                self.sep_token_id,
                self.mask_token_id,
            )
            if token_id is not None
        }

    def _offset_source_text(self, value: Any, is_split_into_words: bool = False) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, tuple):
            return " ".join(self._offset_source_text(item) for item in value)
        if isinstance(value, list):
            separator = " " if is_split_into_words else ""
            return separator.join(self._offset_source_text(item) for item in value)
        return str(value)

    def _synthetic_offset_mapping(self, text: Any, input_ids: Any, is_split_into_words: bool = False) -> list[tuple[int, int]]:
        """Return slow-tokenizer-compatible offsets for evaluators that require them.

        SentencePiece offsets are not available for this Python tokenizer because
        raw Mandarin text is preprocessed into pinyin-code before encoding. These
        spans conservatively distribute non-special tokens across the original
        text so suffix/completion masking code can run without requiring a fast
        tokenizer.
        """
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        source = self._offset_source_text(text, is_split_into_words=is_split_into_words)
        source_length = len(source)
        if not ids:
            return []
        if source_length == 0:
            return [(0, 0) for _ in ids]

        non_content_ids = self._non_content_token_ids()
        content_positions = [
            index for index, token_id in enumerate(ids) if int(token_id) not in non_content_ids
        ]
        if not content_positions:
            return [(0, 0) for _ in ids]

        offsets = [(0, 0) for _ in ids]
        count = len(content_positions)
        for ordinal, position in enumerate(content_positions):
            start = math.floor(ordinal * source_length / count)
            end = math.ceil((ordinal + 1) * source_length / count)
            if end <= start:
                end = min(source_length, start + 1)
            offsets[position] = (start, end)
        return offsets

    def _with_optional_offsets(
        self,
        encoding,
        original_text: Any,
        return_offsets_mapping: bool,
        is_split_into_words: bool = False,
        return_tensors: str | None = None,
    ):
        if not return_offsets_mapping:
            return encoding

        input_ids = encoding["input_ids"]
        tensor_input = hasattr(input_ids, "ndim")
        input_ids_list = input_ids.tolist() if tensor_input else input_ids

        is_batched = False
        if tensor_input:
            is_batched = input_ids.ndim > 1
        elif input_ids_list and isinstance(input_ids_list[0], list):
            is_batched = True

        if is_batched:
            if isinstance(original_text, list) and not is_split_into_words:
                texts = original_text
            else:
                texts = [original_text] * len(input_ids_list)
            offsets = [
                self._synthetic_offset_mapping(text, ids, is_split_into_words=is_split_into_words)
                for text, ids in zip(texts, input_ids_list)
            ]
        else:
            offsets = self._synthetic_offset_mapping(
                original_text,
                input_ids_list,
                is_split_into_words=is_split_into_words,
            )

        if return_tensors == "pt" or tensor_input:
            try:
                import torch

                offsets = torch.tensor(offsets, dtype=torch.long)
            except ImportError:
                pass
        encoding["offset_mapping"] = offsets
        return encoding

    def __call__(self, text=None, text_pair=None, *args, **kwargs):
        original_text = text
        return_offsets_mapping = bool(kwargs.pop("return_offsets_mapping", False))
        is_split_into_words = bool(kwargs.get("is_split_into_words", False))
        return_tensors = kwargs.get("return_tensors")

        if "text_target" in kwargs:
            kwargs["text_target"] = self._preprocess_tokenizer_input(kwargs["text_target"])
        if "text_pair_target" in kwargs:
            kwargs["text_pair_target"] = self._preprocess_tokenizer_input(
                kwargs["text_pair_target"]
            )

        text = self._preprocess_tokenizer_input(text)
        text_pair = self._preprocess_tokenizer_input(text_pair)
        if text_pair is None:
            encoding = super().__call__(text, *args, **kwargs)
        else:
            encoding = super().__call__(text, text_pair, *args, **kwargs)
        return self._with_optional_offsets(
            encoding,
            original_text,
            return_offsets_mapping,
            is_split_into_words=is_split_into_words,
            return_tensors=return_tensors,
        )

    def encode(self, text, text_pair=None, add_special_tokens=True, *args, **kwargs):
        kwargs["add_special_tokens"] = add_special_tokens
        text = self._preprocess_tokenizer_input(text)
        text_pair = self._preprocess_tokenizer_input(text_pair)
        if text_pair is None:
            return super().encode(text, *args, **kwargs)
        return super().encode(text, text_pair, *args, **kwargs)

    def encode_plus(self, text, text_pair=None, *args, **kwargs):
        original_text = text
        return_offsets_mapping = bool(kwargs.pop("return_offsets_mapping", False))
        is_split_into_words = bool(kwargs.get("is_split_into_words", False))
        return_tensors = kwargs.get("return_tensors")

        text = self._preprocess_tokenizer_input(text)
        text_pair = self._preprocess_tokenizer_input(text_pair)
        if text_pair is None:
            encoding = super().encode_plus(text, *args, **kwargs)
        else:
            encoding = super().encode_plus(text, text_pair, *args, **kwargs)
        return self._with_optional_offsets(
            encoding,
            original_text,
            return_offsets_mapping,
            is_split_into_words=is_split_into_words,
            return_tensors=return_tensors,
        )

    def batch_encode_plus(self, batch_text_or_text_pairs, *args, **kwargs):
        original_batch = batch_text_or_text_pairs
        return_offsets_mapping = bool(kwargs.pop("return_offsets_mapping", False))
        is_split_into_words = bool(kwargs.get("is_split_into_words", False))
        return_tensors = kwargs.get("return_tensors")

        batch_text_or_text_pairs = self._preprocess_tokenizer_input(
            batch_text_or_text_pairs
        )
        encoding = super().batch_encode_plus(batch_text_or_text_pairs, *args, **kwargs)
        return self._with_optional_offsets(
            encoding,
            original_batch,
            return_offsets_mapping,
            is_split_into_words=is_split_into_words,
            return_tensors=return_tensors,
        )

    @property
    def vocab_size(self) -> int:
        return self.sp_model.get_piece_size()

    def get_vocab(self) -> dict[str, int]:
        vocab = {self.sp_model.id_to_piece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str) -> list[str]:
        text = self._preprocess_raw_text(text)
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


class EncodedMandarinTokenizer(PinyinCodeTokenizer):
    """Tokenizer wrapper that hides Hanzi-to-encoded-Mandarin preprocessing."""
