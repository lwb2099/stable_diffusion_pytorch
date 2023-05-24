#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/05/15 12:55:11
@Author  :   Wenbo Li
@Desc    :   Util class for build model blocks
'''

from typing import Optional
import numpy as np
import torch
import torch.nn as nn

from torch.nn import ModuleList
from .unet import TimestepEmbedSequential
from modules.resnet2d import DownSample, ResBlock, UpSample
import os
from huggingface_hub import snapshot_download
from .latent_diffusion import LatentDiffusion
from ..modules.transformer import SpatialTransformer, CrossAttention

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def build_conv_in(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


def build_input_blocks(in_channels: int,
                        channels: int, 
                        num_res_blocks: int, 
                        attention_resolutions: Optional[list]=None, 
                        n_heads: Optional[int]=1, 
                        n_layers: Optional[int]=1, 
                        dropout: float=0., 
                        context_dim: Optional[int]=None,
                        levels: int=1,
                        time_emb_dim: Optional[int]=None, 
                        channels_list: list=None,
                        ):
    input_blocks = nn.ModuleList([
                TimestepEmbedSequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
                )
            ])
    # num of channels at each block
    # will use reversly in output blocks
    input_block_channels = [channels]
    #* add levels input blocks(down sample layers)
    in_ch = channels
    #@ note: in labmlai, they use level in attn_resolutions but in origin stable diffusion paper, they use ds = [1,2,4,...],total num levels
    attn_mult = 1
    for level in range(levels):
        for _ in range(num_res_blocks):
            # Residual block maps from previous number of channels to the number of
            # channels in the current level
            out_ch = channels_list[level]
            layers = [ResBlock(in_channels=in_ch, out_channels=out_ch, time_emb_dim=time_emb_dim)]
            in_ch = out_ch  # this level's output channels is next level's input channels
            #* add attention layers
            if attn_mult in attention_resolutions:
                d_head = in_ch // n_heads
                layers.append(SpatialTransformer(in_ch, n_heads=n_heads, 
                                                    d_head=d_head, n_layers=n_layers, 
                                                    dropout=dropout, context_dim=context_dim))
            input_blocks.append(TimestepEmbedSequential(*layers))
            input_block_channels.append(in_ch)
        #* add DownSample block except last level
        if level != levels-1:
            input_blocks.append(TimestepEmbedSequential(DownSample(in_ch)))
            # add additional in_channels for downsample
            input_block_channels.append(in_ch)
            attn_mult *= 2  # double it for next itr, this won't be multiplied in the last layer
    return input_blocks, input_block_channels, in_ch, d_head, attn_mult
 

def build_bottleneck(in_ch: int, 
                     time_emb_dim: Optional[int]=None, 
                     n_heads: int=1, 
                     d_head: int=None, 
                     n_layers: Optional[int]=1, 
                     dropout: float=0., 
                     context_dim: Optional[int]=None,
                     use_attn_only: bool=False
                     ):
    return TimestepEmbedSequential(
        ResBlock(in_channels=in_ch, time_emb_dim=time_emb_dim, dropout=dropout),
        (
            CrossAttention(query_dim=in_ch,n_heads=n_heads, d_head=d_head,dropout=dropout) if use_attn_only else SpatialTransformer(in_ch, n_heads=n_heads, d_head=d_head, n_layers=n_layers, dropout=dropout, context_dim=context_dim)
        ),
        ResBlock(in_channels=in_ch, time_emb_dim=time_emb_dim, dropout=dropout)
    )


def build_output_blocks(num_res_blocks: int, 
                        attention_resolutions: Optional[list]=None, 
                        n_heads: Optional[int]=None, 
                        n_layers: Optional[int]=None, 
                        dropout: float=0., 
                        context_dim: Optional[int]=None, 
                        input_block_channels: Optional[ModuleList]=None,
                        levels: int=1,
                        time_emb_dim: Optional[int]=None,
                        channels_list: list=None, 
                        in_ch: int=1,
                        attn_mult: Optional[int]=None,
                        ):
    output_blocks = nn.ModuleList([])
    for level in reversed(range(levels)):
        #* add resblocks
        for _ in range(num_res_blocks + 1):
            # Residual block maps from previous number of channels to the number of
            # channels in the current level
            #* note: here should add input_block_channels.pop() because input blocks are used as skip connection
            out_ch = channels_list[level]
            layers = [ResBlock(in_channels=in_ch + input_block_channels.pop() if input_block_channels else in_ch, out_channels=out_ch, time_emb_dim=time_emb_dim, dropout=dropout)]
            in_ch = out_ch
            #* add attention layers
            if attn_mult in attention_resolutions:
                d_head = in_ch // n_heads
                layers.append(SpatialTransformer(in_ch, n_heads=n_heads, 
                                                    d_head=d_head, n_layers=n_layers, 
                                                    dropout=dropout, context_dim=context_dim))
            output_blocks.append(TimestepEmbedSequential(*layers))
        # *add UpSample except the last one, note that in reversed order, level==0 is the last
        #@ note: both openai and labmlai use num_res_blocks here and placed it in the inner loop, this can be simply copied from upsample...
        if level != 0:
            output_blocks.append(TimestepEmbedSequential(UpSample(in_ch)))
            attn_mult //= 2
        return output_blocks, in_ch


def build_final_output(out_ch: int, out_channels: int):
    return nn.Sequential(
        nn.GroupNorm(32, out_ch),
        nn.SiLU(),
        nn.Conv2d(in_channels=out_ch, out_channels=out_channels, kernel_size=3, padding=1)
    )


def load_model(autoencoder_path: str="runwayml/stable-diffusion-v1-5/vae",
               text_encoder_path: str="runwayml/stable-diffusion-v1-5/text_encoder",
               unet_path: str="runwayml/stable-diffusion-v1-5/unet",
               scale_factor: float=1.0,
               noise_steps: int=1000,
               beta_start: float=1e-4,
               beta_end: float=1e-2,
               local_path: str="",
               ) -> LatentDiffusion:
    
    # load pretrained model
    autoencoder = snapshot_download.load_snapshot(
            autoencoder_path, local_path=os.path.join(local_path, "autoencoder")
        )
    text_encoder = snapshot_download.load_snapshot(
            text_encoder_path,local_path=os.path.join(local_path, "text_encoder")
        )
    unet = snapshot_download.load_snapshot(
            unet_path, local_path=os.path.join(local_path, "unet")
        )
    # load model
    return LatentDiffusion(
            model=unet, 
            text_encoder=text_encoder, 
            autoencoder=autoencoder, 
            scale_factor=scale_factor, 
            noise_steps=noise_steps, 
            beta_start=beta_start, 
            beta_end=beta_end
        )


if __name__ == "__main__":
    batch = 10
    image_size = 64
    in_channels = 3
    channels = 128
    out_channels = 3
    tim_emb_dim = 512
    x = np.random.randn(batch, in_channels, image_size, image_size)
    x = torch.from_numpy(x).float()
    timestep = torch.ones(size=(batch,))
    emb_dim = 32
    max_len = 100
    print(sinusoidal_time_step_embedding(timestep, emb_dim, max_len).shape)
    # test upsample
    upsample = UpSample(in_channels=channels, )
    up = upsample(x)
    print(up.shape)
    # test downsample
    downsample = DownSample(in_channels=channels, )
    down = downsample(x)
    print(down.shape)
    # test resblock
    resblock = ResBlock(in_channels=channels, out_channels=out_channels, time_emb_dim=tim_emb_dim)
    res = resblock(x, time_emb=torch.ones(size=(batch, tim_emb_dim)))
    print(res.shape)