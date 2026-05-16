"""Configuration for the Transformers-compatible pinyin-code causal LM."""

from __future__ import annotations

from transformers import PretrainedConfig


class PinyinCodeConfig(PretrainedConfig):
    """Configuration for the compact GPT-style pinyin-code decoder."""

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
        **kwargs,
    ) -> None:
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            unk_token_id=unk_token_id,
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
        self.is_decoder = True
        self.is_encoder_decoder = False
        self.use_cache = False
