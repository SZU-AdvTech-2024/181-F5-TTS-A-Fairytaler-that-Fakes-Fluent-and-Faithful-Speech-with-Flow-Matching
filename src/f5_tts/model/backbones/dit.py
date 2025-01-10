"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)
from torch import nn
from x_transformers.x_transformers import RotaryEmbedding


# Text embedding


class TextEmbedding(nn.Module):
    """
    A PyTorch module for generating embeddings from input text data.

    This class extends `nn.Module` to provide text embedding functionality, optionally followed by 
    one or more ConvNeXtV2 blocks for additional context modeling. It is designed to be integrated into 
    machine learning models that require text-to-vector transformations, particularly useful for tasks 
    like text classification, sentiment analysis, or any application needing text representations compatible 
    with numerical processing.

    Attributes:
    - `text_embed`: An `nn.Embedding` layer that maps discrete text tokens to continuous vector space.
    - `extra_modeling`: A boolean indicating whether additional convolutional modeling is applied.
    - `precompute_max_pos`: The maximum sequence length for precomputed positional embeddings.
    - `freqs_cis`: Precomputed positional frequency encodings used when `extra_modeling` is enabled.
    - `text_blocks`: A sequence of ConvNeXtV2 blocks for enhanced text feature extraction.

    Methods:
    - `__init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2)`: 
      Initializes the TextEmbedding module with specified embedding dimensions, number of embeddings, 
      and optional convolutional layers for further text processing.

    - `forward(self, text: torch.Tensor, seq_len: int, drop_text: bool=False) -> torch.Tensor`: 
      Processes input text data to produce embeddings. Allows for dropping text information based on 
      `drop_text` flag. If `extra_modeling` is enabled, applies positional embeddings and passes the 
      text through a series of ConvNeXtV2 blocks before returning the final embeddings.

    Parameters:
    - `text_num_embeds`: The size of the vocabulary, determining the number of embedding vectors.
    - `text_dim`: The dimensionality of the embedding space.
    - `conv_layers`: The number of ConvNeXtV2 blocks to apply for extra text modeling. Defaults to 0.
    - `conv_mult`: Multiplicative factor for the hidden dimension in ConvNeXtV2 blocks. Defaults to 2.

    Raises:
    - No explicit exceptions are documented; however, typical PyTorch execution may raise 
      runtime errors related to tensor dimensions or invalid operations if inputs are incompatible.

    Usage Example:
    ```python
    # Instantiate the model
    embedding_model = TextEmbedding(text_num_embeds=10000, text_dim=256, conv_layers=2)

    # Assume 'texts' is a tensor of token indices shaped (batch_size, seq_length)
    embedded_texts = embedding_model(texts, seq_len=128)
    ```
    """
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, conv_mult=2):
        super().__init__()
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)  # use 0 as filler token

        if conv_layers > 0:
            self.extra_modeling = True
            self.precompute_max_pos = 4096  # ~44s of 24khz audio
            self.register_buffer("freqs_cis", precompute_freqs_cis(text_dim, self.precompute_max_pos), persistent=False)
            self.text_blocks = nn.Sequential(
                *[ConvNeXtV2Block(text_dim, text_dim * conv_mult) for _ in range(conv_layers)]
            )
        else:
            self.extra_modeling = False

    def forward(self, text: int["b nt"], seq_len, drop_text=False):  # noqa: F722
        text = text + 1  # use 0 as filler token. preprocess of batch pad -1, see list_str_to_idx()
        text = text[:, :seq_len]  # curtail if character tokens are more than the mel spec tokens
        batch, text_len = text.shape[0], text.shape[1]
        text = F.pad(text, (0, seq_len - text_len), value=0)

        if drop_text:  # cfg for text
            text = torch.zeros_like(text)

        text = self.text_embed(text)  # b n -> b n d

        # possible extra modeling
        if self.extra_modeling:
            # sinus pos emb
            batch_start = torch.zeros((batch,), dtype=torch.long)
            pos_idx = get_pos_embed_indices(batch_start, seq_len, max_pos=self.precompute_max_pos)
            text_pos_embed = self.freqs_cis[pos_idx]
            text = text + text_pos_embed

            # convnextv2 blocks
            text = self.text_blocks(text)

        return text


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim + text_dim, out_dim)  # Adjusted input dimension
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], text_embed: float["b n d"], drop_text=False):  # Removed 'cond' input
        if drop_text:  # cfg for text
            text_embed = torch.zeros_like(text_embed)

        x = torch.cat((x, text_embed), dim=-1)  # Concatenate only mel and text embeddings
        x = self.proj(x)
        x = self.conv_pos_embed(x) + x
        return x


class InputEmbedding(nn.Module):
    """
    This module represents an input embedding layer for audio-text conditional models. It combines mel spectrogram features,
    conditional audio features, and text embeddings through linear projection followed by positional embedding integration.

    Attributes:
        mel_dim (int): The dimensionality of the mel spectrogram features.
        text_dim (int): The dimensionality of the text embeddings.
        out_dim (int): The output dimensionality after projection.

    Methods:
        __init__(self, mel_dim, text_dim, out_dim): Initializes the InputEmbedding module with specified dimensions.

        forward(self, x, cond, text_embed, drop_audio_cond=False): Processes the input data by concatenating, projecting,
        and applying positional embedding. Allows for conditional audio to be dropped during training.
    """
    def __init__(self, mel_dim, text_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(mel_dim * 2 + text_dim, out_dim)
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)

    def forward(self, x: float["b n d"], cond: float["b n d"], text_embed: float["b n d"], drop_audio_cond=False):  # noqa: F722
        if drop_audio_cond:  # cfg for cond audio
            cond = torch.zeros_like(cond)

        x = self.proj(torch.cat((x, cond, text_embed), dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


# Transformer backbone using DiT blocks


class DiT(nn.Module):
    """
    DiT - Diffusion-based Text-conditioned Audio Synthesis Module

    This class implements a neural network model for generating audio samples conditioned on text inputs, 
    utilizing diffusion processes. It's designed to synthesize audio by combining text embeddings, 
    noisy input audio, and masked conditional audio, leveraging the Transformer architecture for sequence modeling.

    Attributes:
        time_embed (TimestepEmbedding): Embeds time steps for temporal conditioning.
        text_embed (TextEmbedding): Embeds input text into a continuous representation.
        input_embed (InputEmbedding): Fuses mel-spectrogram and text embeddings for input conditioning.
        rotary_embed (RotaryEmbedding): Implements Rotary Positional Embedding for attention mechanism.
        dim (int): The dimensionality of the model.
        depth (int): The number of Transformer blocks in the model.
        transformer_blocks (nn.ModuleList): A list of DiTBlock modules forming the core of the Transformer stack.
        long_skip_connection (nn.Linear|None): An optional linear layer for skip connection across blocks.
        norm_out (AdaLayerNormZero_Final): Normalization layer applied before the output projection.
        proj_out (nn.Linear): Projects the final hidden representations to the output mel-spectrogram space.
        checkpoint_activations (bool): Flag to enable gradient checkpointing for memory-efficient training.

    Methods:
        __init__: Initializes the DiT model with specified architectural configurations and options.
        ckpt_wrapper: Wraps a module with gradient checkpointing functionality.
        forward: Processes input data through the model, generating audio outputs conditioned on text and time.

    Forward Method Args:
        x (torch.Tensor): Noised input audio of shape (batch_size, sequence_length, feature_dim).
        cond (torch.Tensor): Masked conditional audio of the same shape as `x`.
        text (torch.Tensor): Tokenized text input of shape (batch_size, text_sequence_length).
        time (torch.Tensor): Time steps of shape (batch_size,) or a scalar to be broadcasted.
        drop_audio_cond (float): Dropout rate for conditional audio input.
        drop_text (float): Dropout rate for text input.
        mask (torch.Tensor|None): Optional boolean mask for sequences, shape (batch_size, sequence_length).

    Forward Method Returns:
        torch.Tensor: Synthesized mel-spectrogram output of shape (batch_size, sequence_length, feature_dim).
    """
    def __init__(
        self,
        *,
        dim,
        depth=8,
        heads=8,
        dim_head=64,
        dropout=0.1,
        ff_mult=4,
        mel_dim=100,
        text_num_embeds=256,
        text_dim=None,
        conv_layers=0,
        long_skip_connection=False,
        checkpoint_activations=False,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(dim)
        if text_dim is None:
            text_dim = mel_dim
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers)
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        self.norm_out = AdaLayerNormZero_Final(dim)  # final modulation
        self.proj_out = nn.Linear(dim, mel_dim)

        self.checkpoint_activations = checkpoint_activations

    def ckpt_wrapper(self, module):
        # https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def forward(
        self,
        x: float["b n d"],  # nosied input audio  # noqa: F722
        cond: float["b n d"],  # masked cond audio  # noqa: F722
        text: int["b nt"],  # text  # noqa: F722
        time: float["b"] | float[""],  # time step  # noqa: F821 F722
        drop_audio_cond,  # cfg for cond audio
        drop_text,  # cfg for text
        mask: bool["b n"] | None = None,  # noqa: F722
    ):
        batch, seq_len = x.shape[0], x.shape[1]
        if time.ndim == 0:
            time = time.repeat(batch)

        # t: conditioning time, c: context (text + masked cond audio), x: noised input audio
        t = self.time_embed(time)
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)
        x = self.input_embed(x, cond, text_embed, drop_audio_cond=drop_audio_cond)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope)
            else:
                x = block(x, t, mask=mask, rope=rope)

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        output = self.proj_out(x)

        return output
