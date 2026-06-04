"""Transformers-compatible implementation of the pinyin-code causal LM."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutput

from .configuration_pinyin_code import PinyinCodeConfig


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention matching the original training module."""

    def __init__(self, config: PinyinCodeConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout_p = config.dropout
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, embd = x.shape
        q, k, v = self.qkv(x).split(embd, dim=2)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        dropout_p = self.dropout_p if self.training else 0.0
        if attention_mask is not None:
            causal_mask = torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ).tril()
            key_mask = attention_mask[:, None, None, :seq_len].to(dtype=torch.bool)
            attn_mask = causal_mask.view(1, 1, seq_len, seq_len) & key_mask
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=False,
            )
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=dropout_p,
                is_causal=True,
            )

        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embd)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """Transformer MLP block."""

    def __init__(self, config: PinyinCodeConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block."""

    def __init__(self, config: PinyinCodeConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class PinyinCodePreTrainedModel(PreTrainedModel):
    """Base class for pinyin-code Transformers models."""

    config_class = PinyinCodeConfig
    base_model_prefix = "pinyin_code"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class PinyinCodeForCausalLM(PinyinCodePreTrainedModel, GenerationMixin):
    """Compact GPT-style causal language model using the original architecture."""

    _tied_weights_keys = {"lm_head.weight": "token_embedding.weight"}
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight"]

    def __init__(self, config: PinyinCodeConfig) -> None:
        super().__init__(config)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.token_embedding = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def tie_weights(self, *args, **kwargs) -> None:
        self.lm_head.weight = self.token_embedding.weight

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict:
        if input_ids.shape[1] > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.block_size :]
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutput | tuple:
        return_dict = True if return_dict is None else return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must provide either input_ids or inputs_embeds")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot provide both input_ids and inputs_embeds")

        if inputs_embeds is None:
            _, seq_len = input_ids.shape
            if seq_len > self.config.block_size:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds block size {self.config.block_size}"
                )
            inputs_embeds = self.token_embedding(input_ids)
        else:
            seq_len = inputs_embeds.shape[1]
            if seq_len > self.config.block_size:
                raise ValueError(
                    f"Sequence length {seq_len} exceeds block size {self.config.block_size}"
                )

        positions = torch.arange(seq_len, device=inputs_embeds.device)
        x = inputs_embeds + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        logits = self.lm_head(self.ln_f(x))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(loss=loss, logits=logits)
