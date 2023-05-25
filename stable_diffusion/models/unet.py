#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   unet_2d_conditional.py
@Time    :   2023/05/14 15:37:56
@Author  :   Wenbo Li
@Desc    :   implementation of Unet2d model and sub modules
"""

import numpy as np
import torch
import torch.nn as nn

from stable_diffusion.dataclass import BaseDataclass
from .utils import (
    build_conv_in,
    build_input_blocks,
    build_bottleneck,
    build_output_blocks,
    build_final_output,
)

from ..modules.timestep_embedding import sinusoidal_time_proj

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class UnetConfig(BaseDataclass):
    num_res_blocks: int = field(
        default=2, metadata={"help": "Number of residual blocks at each level."}
    )
    n_heads: int = field(
        default=1, metadata={"help": "Number of attention heads in transformers."}
    )
    attention_resolutions: List[int] = field(
        default_factory=lambda: [1],
        metadata={
            "help": "At which level attention should be performed. e.g., [1, 2] means attention is performed at level 1 and 2."
        },
    )
    channels_list: List[int] = field(
        default_factory=lambda: [64, 128], metadata={"help": "Channels at each level."}
    )
    time_emb_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": "Time embedding dimension. If not specified, use 4 * channels_list[0] instead."
        },
    )
    dropout: float = field(default=0.0, metadata={"help": "Dropout rate."})
    n_layers: int = field(default=1, metadata={"help": "Number of transformer layers."})
    context_dim: int = field(
        default=768, metadata={"help": "Embedding dim of context condition."}
    )


class UNetModel(nn.Module):
    """
    The full `U-Net` model=`ε_θ(x_t, t, condition)` that takes noised latent x, time step and context condition, predicts the noise

    `U-Net` as several levels(total `len(channel_mult)`), each level is a `TimestepEmbSequential`,
    at each level there are sevreal(total `num_res_blocks`)
    `ResBlocks`/`SpatialTransformer`/`AttentionBlock`/`UpSample`/`DownSample` blocks

    Archetecture:
      - input_blocks: [
                    TimestepEmbSeq[`Conv2d]`,
                    (num_levels-1) * (
                        (num_res_blocks)* (TimestepEmbSeq[`Resblock`, Optioanl[`SpatialTransformer`]]),
                        TimestepEmbSeq[`DownSample`]
                    ),
                    (num_res_blocks)* (TimestepEmbSeq[`Resblock`, Optioanl[`SpatialTransformer`]])
            ]
      - bottleneck: TimestepEmbSeq[`ResBlock`, `SpatialTransformer`, `ResBlock`]

      - output_blocks: [
                    (num_levels-1) * (
                        (num_res_blocks)* (TimestepEmbSeq[`Resblock`, Optioanl[`SpatialTransformer`]]),
                        TimestepEmbSeq[`UpSample`]
                    ),
                    (num_res_blocks)* (TimestepEmbSeq[`Resblock`, Optioanl[`SpatialTransformer`]])
               ]
      - out: [GroupNorm, SiLU, Conv2d]

    Args:
        - in_channels (int):
                number of channels in the input feature map
        - out_channels (int):
                number of channels in the output feature map
        - channels (int):
                base channel count for the model
        - num_res_blocks (int):
                number of residual blocks at each level
        - n_heads (int):
                num of attention heads in transformers
        - attention_resolutions (List[int]):
                at which level should attention be performed.
                e.g. [1, 2] means attention is performed at level 1 and 2.
        - channel_mult (List[int]):
                channel multiplier for each level of the UNet.
                e.g. [1, 2, 4] means the first level has 1*channel, second level has 2*channel, third level has 4*channel.
        - dropout (float, optional):
                dropout rate. Default: `0`.
        - n_layers (int, optional):
                num of transformer layers. Default: `1`.
        - context_dim (int, optional):
                embedding dim of context condition. Default: `768`.
    """

    @staticmethod
    def add_unet_args(parser):
        unet_group = parser.add_argument_group("unet")
        unet_group.add_argument(
            "--num_res_blocks",
            type=int,
            default=2,
            help="number of residual blocks at each level",
        )
        unet_group.add_argument(
            "--n_heads",
            type=int,
            default=1,
            help="num of attention heads in transformers",
        )
        unet_group.add_argument(
            "--attention_resolutions",
            type=int,
            nargs="+",
            default=[1],
            help="at which level should attention be performed. e.g. [1, 2] means attention is performed at level 1 and 2.",
        )
        unet_group.add_argument(
            "--channels_list",
            type=int,
            nargs="+",
            default=[64, 128],
            help="channels at each level",
        )
        unet_group.add_argument(
            "--time_emb_dim",
            type=int,
            default=None,
            help="time embedding dimension, if not specified, use 4 * channels_list[0] instead",
        )
        unet_group.add_argument(
            "--dropout",
            type=float,
            default=0.0,
            help="dropout rate",
        )
        unet_group.add_argument(
            "--n_layers",
            type=int,
            default=1,
            help="num of transformer layers",
        )
        unet_group.add_argument(
            "--context_dim",
            type=int,
            default=768,
            help="embedding dim of context condition",
        )

    def __init__(
        self,
        latent_channels,
        cfg,  # unet config
    ):
        super().__init__()
        num_res_blocks = cfg.num_res_blocks
        n_heads = cfg.n_heads
        attention_resolutions = cfg.attention_resolutions
        channels_list = cfg.channels_list
        dropout = cfg.dropout
        n_layers = cfg.n_layers
        context_dim = cfg.context_dim
        self.context_dim = cfg.context_dim

        # * 1. time emb
        time_emb_dim = cfg.time_emb_dim or channels_list[0] * 4
        timestep_input_dim = channels_list[0]
        self.time_embedding = nn.Sequential(
            nn.Linear(timestep_input_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # * 2. conv in
        self.conv_in = build_conv_in(latent_channels, channels_list[0])
        # num of levels
        levels = len(channels_list)
        # * 3. input blocks
        (
            self.input_blocks,
            input_block_channels,
            mid_ch,
            d_head,
            attn_mult,
        ) = build_input_blocks(
            channels_list[0],
            num_res_blocks,
            attention_resolutions,
            n_heads,
            n_layers,
            dropout,
            context_dim,
            levels,
            time_emb_dim,
            channels_list,
        )
        # @ note: openai recalculated d_heads for attention in the bottoleneck, but that seems redundant(so as out_ch=ch and then ch=out_ch)
        # * 4. bottleneck
        # bottleneck has one resblock, then one attention block/spatial transformer, and then one resblock
        self.middle_block = build_bottleneck(
            mid_ch, time_emb_dim, n_heads, d_head, n_layers, dropout, context_dim
        )
        # * 5. output blocks
        self.output_blocks, out_ch = build_output_blocks(
            num_res_blocks,
            attention_resolutions,
            n_heads,
            n_layers,
            dropout,
            context_dim,
            input_block_channels,
            levels,
            time_emb_dim,
            channels_list,
            mid_ch,
            attn_mult,
        )
        # * 6. final output
        self.out = build_final_output(out_ch, latent_channels)

    def time_proj(self, time_steps: torch.Tensor, max_len: int = 10000) -> torch.Tensor:
        """
        time_step_embedding use sinusoidal time step embedding as default, feel free to try out other embeddings

        Args:
            - time_steps (torch.Tensor):
                  time step of shape `[batch_size,]`
            - max_len (int, optional):
                  max len of embedding. Default: `10000`.

        Returns:
            - torch.Tensor:
                  time embedding of shape `[batch, emb_dim]`
        """
        return sinusoidal_time_proj(time_steps, channels, max_len)

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor, context_emb: torch.Tensor = None
    ):
        """
        forward pass, predict noise given noised image x, time step and context

        Args:
            - x (torch.Tensor):
                  input latent of shape `[batch_size, channels, width, height]`
            - timesteps (torch.Tensor):
                  time steps of shape `[batch_size]`
            - context_emb (torch.Tensor, optional):
                  cross attention context embedding of shape `[batch_size, seq_len, context_dim]`. Default: None.

        Returns:
            - torch.Tensor:
                  output feature map of shape `[batch_size, channels, width, height]`
        """
        # check parameters
        if context_emb is not None:
            assert (
                context_emb.shape[-1] == self.context_dim
            ), f"context dim from passed in context({context_emb.shape}) should be equal to self.context_dim({self.context_dim})"

        # store input blocks for skip connection
        x_input_blocks = []
        # * Get time embedding
        time_emb = self.time_proj(timesteps)
        time_emb = self.time_emb(time_emb)
        # * conv in layer
        x = self.conv_in(x)
        # * input blocks
        for module in self.input_blocks:
            x = module(x, time_emb, context_emb)
            x_input_blocks.append(x)
        # * bottleneck
        x = self.middle_block(x, time_emb, context_emb)
        # * output blocks
        for module in self.output_blocks:
            # skip connection from input blocks
            x = torch.cat([x, x_input_blocks.pop()], dim=1)
            x = module(x, time_emb, context_emb)
        return self.out(x)


if __name__ == "__main__":
    batch = 10
    image_size = 64
    in_channels = 3
    channels = 128
    out_channels = 3
    tim_emb_dim = 512
    seq_len = 77
    context_dim = 768
    x = np.random.randn(batch, in_channels, image_size, image_size)
    x = torch.from_numpy(x).float()
    timestep = torch.ones(size=(batch,))
    # test U-Net
    context = torch.ones((batch, seq_len, context_dim))
    unet = UNetModel(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        num_res_blocks=2,
        n_heads=4,
        attention_resolutions=(1, 2),
        channel_mult=[1, 2],
        dropout=0.1,
        n_layers=2,
        context_dim=context_dim,
    )
    out = unet(x, timestep, context)
    print(out.shape)
