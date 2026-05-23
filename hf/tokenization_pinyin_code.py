"""SentencePiece tokenizer wrapper for pinyin-code Transformers models."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path

import sentencepiece as spm
from transformers import PreTrainedTokenizer


CHINESE_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
CHINESE_SPAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
PUNCTUATION = set("\u3002\uff0c\u3001\uff1f\uff01\uff1a\uff1b.,?!:;()[]\u201c\u201d\"'")
TOKEN_RE = re.compile(
    r"<[A-Z_]+>|"
    r"[\u3400-\u4dbf\u4e00-\u9fff]+|"
    r"[A-Za-z]+|"
    r"[\u3002\uff0c\u3001\uff1f\uff01\uff1a\uff1b.,?!:;()\[\]\u201c\u201d\"']|"
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
        **kwargs,
    ) -> None:
        self.vocab_file = vocab_file
        self.sp_model = spm.SentencePieceProcessor(model_file=vocab_file)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.transliteration = self._normalize_transliteration(
            pinyin_format or transliteration
        )

        kwargs.setdefault("unk_token", self._piece_or_none(self.sp_model.unk_id()))
        kwargs.setdefault("bos_token", self._piece_or_none(self.sp_model.bos_id()))
        kwargs.setdefault("eos_token", self._piece_or_none(self.sp_model.eos_id()))
        kwargs.setdefault("pad_token", self._piece_or_none(self.sp_model.pad_id()))
        kwargs.setdefault("transliteration", self.transliteration)
        kwargs.setdefault("pinyin_format", self.transliteration)
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

    def _preprocess_raw_text(self, text: str) -> str:
        if not CHINESE_RE.search(text):
            return text
        try:
            from preprocessing.preprocess import process_text, require_dependencies
        except ImportError:
            return self._fallback_process_text(text)

        require_dependencies()
        return process_text(text, self.transliteration)

    def _fallback_process_text(self, text: str) -> str:
        try:
            import jieba
            from pypinyin import Style, pinyin
        except ImportError as exc:
            raise ImportError(
                "Tokenizing raw Mandarin benchmark text requires jieba and pypinyin. "
                "Install the model dependencies before running lm_eval."
            ) from exc

        jieba.setLogLevel(logging.WARNING)

        def normalize_text(value: str) -> str:
            value = re.sub(r"\$\$.*?\$\$", " <MATH> ", value, flags=re.DOTALL)
            value = re.sub(r"[\uff08(]\s*[\uff09)]", " <BLANK> ", value)
            for label, marker in LABELS.items():
                value = re.sub(rf"{label}\s*[:\uff1a]", f" {marker} ", value)
            value = re.sub(
                r"(?<![A-Za-z])yes(?![A-Za-z])", " <YES> ", value, flags=re.I
            )
            value = re.sub(
                r"(?<![A-Za-z])no(?![A-Za-z])", " <NO> ", value, flags=re.I
            )
            value = re.sub(
                r"(?<![A-Za-z])[ABCD](?=\s*[:\uff1a.\uff0e\u3001\)])",
                r" \g<0> ",
                value,
            )
            value = re.sub(
                r"(?<![A-Za-z0-9])[-+]?\d+(?:[.,]\d+)*(?:%|\uff05)?",
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
            for word in jieba.cut(value, cut_all=False):
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
            elif re.fullmatch(r"[A-Za-z]+", part):
                upper = part.upper()
                tokens.append(upper if upper in {"A", "B", "C", "D"} else part.lower())

        return " ".join(tokens)

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
