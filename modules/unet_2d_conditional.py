#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   unet_2d_conditional.py
@Time    :   2023/05/14 15:37:56
@Author  :   Wenbo Li
@Desc    :   implementation of Unet2d model and sub modules
            reference:
            origin stable diffusion github:
            https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/diffusionmodules/openaimodel.py
            labmlai annotated deep learning paper implementation(simplify origin implementation):
            https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/model/unet.py

'''
from abc import abstractmethod
from typing import List, Optional
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

from stable_diffusion_pytorch.modules.unet_transformer import SpatialTransformer

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
"""
    @abstractmethod
    def forward(self, x, time_emb):
        pass


class TimestepEmbeddingSequential(nn.Sequential, TimestepBlock):
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
            #TODO: figure out shapee   
                  input shape=[]. Default: None.

        Returns:
            - Tensor:   
                - output shape:
                    - `ResBlock`:           [batch, out_channels, height, width]
                    - `SpatialTransformer`: [batch, out_channels, height, width]
                    - `Conv2d`:             [batch, out_channels, height, width]
                    - `UpSample`:           [batch, out_channels, height, width]
                    - `DownSample`:         [batch, out_channels, height, width]
        """
        for module in self:
            # pass ResBlock
            if isinstance(module, TimestepBlock):
                x = module(x, time_emb)
            # pass spatial transformer
            elif isinstance(module, SpatialTransformer):
                x = module(x, context)
            # pass Conv2d, UpSample, DownSample
            else:
                x = module(x)
        return x


class UpSample(nn.Module):
    """
    constructor for UpSample layer

    Args:
        - in_channels (int):   
                _description_
        - out_channels (int, optional):   
                _description_
        - scale_factor (int, optional):   
                Up Sample by a factor of `scale factor`. Default: 2.
        - padding (int, optional):   
                padding for Conv2d. Default: 1.
    """
    def __init__(self, 
                 in_channels: int, 
                 out_channels: Optional[int]=None, 
                 scale_factor: float=2, 
                 padding: int=1
                 ):
        
        super().__init__()
        self.in_channels = in_channels
        # if has out_channels passed in, use it or use default = in_channels
        self.out_channels = out_channels or in_channels
        self.scale_factor = scale_factor
        #* default we use Conv2d, use_conv and conv_nd are not implemented for simpilicity
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=padding)

    def forward(self, x) -> torch.Tensor:
        """
        forward Up Sample layer which 

        Args:
            - x (Tensor):   
                  input shape=[batch, in_channels, height, width]

        Returns:
            - Tensor:   
                  output shape=[batch, out_channels, height*scale_factor, width*scale_factor]
        """
        assert x.shape[1] == self.in_channels, f"input channel does not match: x.shape[1]({x.shape[1]}) != self.in_channels({self.in_channels}))"
        #* e.g. [1,3,256,256] -> [1,3,512,512]
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    """
    constructor for DownSample layer

    Args:
        - in_channels (int):   
                _description_
        - out_channels (int):   
                _description_
        - scale_factor (int, optional):   
                Down Sample by a factor of `scale factor`. Default: 1/2.
        - padding (int, optional):   
                padding for Conv2d. Default: 1.
    """
    def __init__(self, 
                 in_channels: int,
                 out_channels: Optional[int]=None,
                 scale_factor: float=1/2,
                 padding: int=1
                ):
        super().__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        # if has out_channels passed in, use it or use default = in_channels
        self.out_channels = out_channels or in_channels
        # It turn out that origin openai unet model did not use UpSample and DownSample,
        # instead, they use ResNetBlock with parameters up, down
        # Here we use UpSample and DownSample to make it clean
        # Change: => add stride=2, to downsample by a factor of 2
        # or use scale factor = 1/2
        #TODO: compare difference between stride=2 and scale_factor=1/2
        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=padding)

    def forward(self, x) -> torch.Tensor:
        """
        forward Down Sample layer which reduce height and width of x by a factor of 2

        Args:
            - x (Tensor):   
                  input shape=[batch, in_channels, height, width]

        Returns:
            - Tensor:   
                  output shape=[batch, out_channels, height/scale_factor, width/scale_factor]
        """
        assert x.shape[1] == self.in_channels, f"input channel does not match: x.shape[1]({x.shape[1]}) != self.in_channels({self.in_channels}))"
        x = self.conv(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        return x



class ResBlock(nn.Module):
    def __init__(self,
                in_channels, 
                out_channels: Optional[int]=None,
                time_emb_dim: int=512,
                dropout: int=0,
                padding: int=1
                  ) -> None:
        super().__init__()
        # check parameters
        assert in_channels % 32 == 0, f"in_channels{in_channels} must be divisible by num_groups(32)"
        self.in_channels = in_channels
        # if has out_channels passed in, use it or use default = in_channels
        self.out_channels = out_channels or in_channels
        self.time_emb_dim = time_emb_dim
        self.dropout = dropout
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=padding),
        )
        # Time embedding
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=self.out_channels),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            #? openai model used zero_model(conv_nd)
            #TODO: figure out why zero_model is used
            # [batch, in_channel, height, width] => [batch, out_channel, height, height]
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=padding),
        )
        # Map input to output channel
        if self.in_channels != self.out_channels:
            self.skip_connection = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        forward pass of ResBlock

        Args:
            - x (torch.Tensor):   
                  input shape = [batch, in_channels, height, width]
            - time_emb (torch.Tensor):   
                  input shape = [batch, time_emb_dim]

        Returns:
            - torch.Tensor:   
                  output shape = [batch, out_channels, height, width]
        """
        assert x.shape[0] == time_emb.shape[0], f"batch size does not match: x.shape[0]({x.shape[0]}) != time_emb.shape[0]({time_emb.shape[0]})"
        assert time_emb.shape[1] ==self. time_emb_dim, f"time_emb_dim does not match: time_emb.shape[1]({time_emb.shape[1]}) != self.time_emb_dim({self.time_emb_dim})"
        # h: [batch, out_channels, height, width]
        h = self.in_layers(x)
        # [batch, time_emb_dim] => [batch, out_channels] => [batch, out_channels, 1, 1]
        time_emb = self.time_emb(time_emb)
        h += time_emb[:,:,None,None]
        h = self.out_layers(h)
        return h + self.skip_connection(x)


class UNet2DCondition(nn.Module):

    def __init__(self, 
                in_channels: int, 
                out_channels: int,
                channels: int,
                num_res_blocks: int,
                n_head: int,
                attention_resolutions: List[int],
                channel_mult: List[int],
                dropout: float=0.,
                n_layers: int=1,
                context_dim: int=768,
            ):
        """
        The full `U-Net` model with `attention` and `timestep embedding`.

        `U-Net` as several levels(total `len(channel_mult)`), each level is a `TimestepEmbSequential`,
        at each level there are sevreal(total `num_res_blocks`) 
        `ResBlocks`/`SpatialTransformer`/`AttentionBlock`/`UpSample`/`DownSample` blocks
        
        Args:
            - in_channels (int):   
                  number of channels in the input feature map
            - out_channels (int):   
                  number of channels in the output feature map
            - channels (int):   
                  #? base channel count for the model
            - num_res_blocks (int):   
                  number of residual blocks at each #? downsample
            - n_head (int):   
                  num of attention heads in transformers 
            - attention_resolutions (List[int]):   
                  at which level should attention be performed.
                  e.g. [1, 2] means attention is performed at level 1 and 2.
            - channel_mult (List[int]):   
                  channel multiplier for each level of the UNet.
            - dropout (float, optional):   
                  dropout rate. Default: 0.
            - n_layers (int, optional):   
                  num of transformer layers. Default: 1.
            - context_dim (int, optional):   
                  embedding dim of context condition. Default: 768.
        """
        super.__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.n_head = n_head
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.dropout = dropout
        self.n_layers = n_layers
        self.context_dim = context_dim
        
        # check parameters
        # attention can't be performed on levels that don't exists
        assert max(attention_resolutions) <= len(channel_mult), f"attention_resolutions({attention_resolutions}) should be less than len(channel_mult)({len(channel_mult)})"
        
        # num of levels
        num_levels = len(self.channel_mult)
        
        #* time emb
        time_emb_dim = channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        #* input blocks
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbeddingSequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
                )
            ]
        )
        # num of channels at each block
        self.input_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_mult]


if __name__ == "__main__":
    x = np.random.randn(1, 128, 64, 64)
    x = torch.from_numpy(x).float()
    # test upsample
    upsample = UpSample(128, )
    up = upsample(x)
    print(up.shape)
    # test downsample
    downsample = DownSample(128, )
    down = downsample(x)
    print(down.shape)
    # test resblock
    resblock = ResBlock(128, 96, 512)
    res = resblock(x, torch.ones(1, 512))
    print(res.shape)
