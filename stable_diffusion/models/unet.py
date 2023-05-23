#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   unet_2d_conditional.py
@Time    :   2023/05/14 15:37:56
@Author  :   Wenbo Li
@Desc    :   implementation of Unet2d model and sub modules
'''

from abc import abstractmethod
from typing import List
import numpy as np
import torch 
import torch.nn as nn
from utils import build_bottleneck, build_conv_in, build_final_output, build_input_blocks, build_output_blocks
from ..modules.resnet2d import ResBlock
from ..modules.timestep_embedding import sinusoidal_time_step_embedding
from ..modules.transformer import SpatialTransformer

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    @abstractmethod
    def forward(self, x, time_emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, context: torch.Tensor=None):
        """
        forward pass of TimestepEmbeddingSequential

        passed in layers are basically: 
            `ResBlock`, `Conv2d`, `UpSample`,`DownSample`, `SpatialTransformer`
        apply different forward pass depending on instance type
        
        Args:
            - x (Tensor):   
                  input shape=[batch, in_channels, height, width]
            - time_emb (Tensor):   
                  input shape=[batch, time_emb_dim]
            - context (Tensor, optional):
            #TODO: figure out shape  
                  input shape=[]. Default: None.

        Returns:
            - Tensor:   
                - output shape:
                    - `ResBlock`:           [batch, out_channels, height, width]
                    - `SpatialTransformer`: [batch, out_channels, height, width]
                    - `Conv2d`:             [batch, out_channels, height, width]
                    - `UpSample`:           [batch, out_channels, height*scale_factor, width*scale_factor]
                    - `DownSample`:         [batch, out_channels, height/scale_factor, width/scale_factor]
        """
        for module in self:
            # pass ResBlock
            if isinstance(module, ResBlock):
                x = module(x, time_emb)
            # pass spatial transformer
            elif isinstance(module, SpatialTransformer):
                x = module(x, context)
            # pass Conv2d, UpSample, DownSample
            else:
                x = module(x)
        return x



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
                #? base channel count for the model
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
    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                channels: int,
                num_res_blocks: int,
                n_heads: int,
                attention_resolutions: List[int],
                channel_mult: List[int],
                dropout: float=0.,
                n_layers: int=1,
                context_dim: int=768,
            ):
        super().__init__()
        self.context_dim = context_dim
        # check parameters
        # attention can't be performed on levels that don't exists
        assert max(attention_resolutions) <= len(channel_mult), f"attention_resolutions({attention_resolutions}) should be less than len(channel_mult)({len(channel_mult)})"
        
        #* 1. time emb
        time_emb_dim = channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        #* 2. conv in
        self.conv_in = build_conv_in(in_channels, channels)
        # num of levels
        levels = len(channel_mult)
        # Number of channels at each level
        channels_list = [channels * m for m in channel_mult]
        #* 3. input blocks
        self.input_blocks, input_block_channels, mid_ch, d_head, attn_mult = build_input_blocks(
                in_channels, channels, num_res_blocks, attention_resolutions, n_heads, n_layers, dropout, context_dim, 
                levels, time_emb_dim, channels_list
            )
        #@ note: openai recalculated d_heads for attention in the bottoleneck, but that seems redundant(so as out_ch=ch and then ch=out_ch)
        #* 4. bottleneck
        # bottleneck has one resblock, then one attention block/spatial transformer, and then one resblock
        self.middle_block = build_bottleneck(mid_ch, time_emb_dim, n_heads, d_head, n_layers, dropout, context_dim)
        #* 5. output blocks
        self.output_blocks, out_ch = build_output_blocks(
            num_res_blocks,attention_resolutions,n_heads, n_layers, dropout, context_dim,
            input_block_channels, levels, time_emb_dim, channels_list, mid_ch, attn_mult,
        )
        #* 6. final output
        self.out = build_final_output(out_ch, out_channels)

    def time_step_embedding(self, time_steps: torch.Tensor, max_len: int=10000) -> torch.Tensor:
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
        return sinusoidal_time_step_embedding(time_steps, channels, max_len)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context_emb: torch.Tensor=None):
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
            assert context_emb.shape[-1] == self.context_dim, f"context dim from passed in context({context_emb.shape}) should be equal to self.context_dim({self.context_dim})"

        # store input blocks for skip connection
        x_input_blocks = []
        #* Get time embedding
        time_emb = self.time_step_embedding(timesteps) 
        time_emb = self.time_emb(time_emb)
        #* conv in layer
        x = self.conv_in(x)
        #* input blocks
        for module in self.input_blocks:
            x = module(x, time_emb, context_emb)
            x_input_blocks.append(x)
        #* bottleneck
        x = self.middle_block(x, time_emb, context_emb)
        #* output blocks
        for module in self.output_blocks:
            # skip connection from input blocks
            x = torch.cat([x,x_input_blocks.pop()], dim=1)
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
    unet = UNetModel(in_channels=in_channels, out_channels=out_channels, channels=channels, num_res_blocks=2, n_heads=4,
                     attention_resolutions=(1, 2), channel_mult=[1,2], dropout=0.1,  n_layers=2, context_dim=context_dim)
    out = unet(x, timestep, context)
    print(out.shape)
