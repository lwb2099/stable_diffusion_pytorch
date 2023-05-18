#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   unet_2d_conditional.py
@Time    :   2023/05/14 15:37:56
@Author  :   Wenbo Li
@Desc    :   implementation of Unet2d model and sub modules
'''

from abc import abstractmethod
from typing import List, Optional
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from utils import zero_module
from utils import sinusoidal_time_step_embedding

from unet_transformer import SpatialTransformer

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
            #TODO: figure out shapee   
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


class UpSample(nn.Module):
    """
    constructor for UpSample layer

    Architecture:
        - Interpolate
        - Conv2d

    Args:
        - in_channels (int):   
                input  num of channels
        - out_channels (int, optional):   
                output  num of channels
        - scale_factor (int, optional):   
                Up Sample by a factor of `scale factor`. Default: `2`.
        - padding (int, optional):   
                padding for `Conv2d`. Default: `1`.
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
                  input shape=`[batch, in_channels, height, width]`

        Returns:
            - Tensor:   
                  output shape=`[batch, out_channels, height*scale_factor, width*scale_factor]`
        """
        assert x.shape[1] == self.in_channels, f"input channel does not match: x.shape[1]({x.shape[1]}) != self.in_channels({self.in_channels}))"
        #* e.g. [1,3,256,256] -> [1,3,512,512]
        x = F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    """
    constructor for DownSample layer

    Architecture:
        - Conv2d
        - Interpolate

    Args:
        - in_channels (int):   
                input  num of channels
        - out_channels (int, optional):   
                output  num of channels
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
    """
        ResBlock used in U-Net
        
        Archetecture:: 
            - in_layers = [`GroupNorm`, `SiLU`, `Conv2d`]
            - time_emb = [`SiLU`, `Linear`]
            - out_layers = [`GroupNorm`, `SiLU`, `Dropout`]

        Args:
            - in_channels (int):   
                  input num of channels
            - out_channels (Optional[int], optional):   
                  output num of channels. Default: equal to `in_channels`.
            - time_emb_dim (int, optional):   
                  time embedding dim. Default: `512`.
            - dropout (int, optional):   
                  dropout rate. Default: `0.`
            - padding (int, optional):   
                  padding idx. Default: `1`.
        """
    def __init__(self,
                in_channels: int, 
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
            #? openai model used zero_module(conv_nd)
            #TODO: figure out why zero_module is used
            # [batch, in_channel, height, width] => [batch, out_channel, height, height]
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=padding)),
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
                  input shape = `[batch, in_channels, height, width]`
            - time_emb (torch.Tensor):   
                  input shape = `[batch, time_emb_dim]`

        Returns:
            - torch.Tensor:   
                  output shape = `[batch, out_channels, height, width]`
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


class UNetModel(nn.Module):
    """
    The full `U-Net` model with `attention` and `timestep embedding`.

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.n_heads = n_heads
        self.attention_resolutions = attention_resolutions
        self.channel_mult = channel_mult
        self.n_layers = n_layers
        self.context_dim = context_dim
        
        # check parameters
        # attention can't be performed on levels that don't exists
        assert max(attention_resolutions) <= len(channel_mult), f"attention_resolutions({attention_resolutions}) should be less than len(channel_mult)({len(channel_mult)})"
        
        # num of levels
        levels = len(self.channel_mult)
        
        #* time emb
        time_emb_dim = channels * 4
        self.time_emb = nn.Sequential(
            nn.Linear(channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # ==============================================================
        # ======================input blocks============================
        # ==============================================================
        #* init input blocks and add first layer
        # use TimestepEmbedSequential wrapped Conv2d because different model has different forward pass
        #? did not quite understand above, comment refered to https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/stable_diffusion/model/unet.py#L29
        #? line 69-74
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
                )
            ]
        )
        # num of channels at each block
        # will use reversly in output blocks
        self.input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_mult]
        #* add levels input blocks(down sample layers)
        in_ch = self.channels
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
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_block_channels.append(in_ch)
            #* add DownSample block except last level
            if level != levels-1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(in_ch)))
                # add additional in_channels for downsample
                self.input_block_channels.append(in_ch)
                attn_mult *= 2  # double it for next itr, this won't be multiplied in the last layer

            #@ note: openai recalculated d_heads for attention in the bottoleneck, but that seems redundant(so as out_ch=ch and then ch=out_ch)
        # ==============================================================
        # =======================bottleneck=============================
        # ==============================================================
        #* bottleneck has one resblock, then one attention block/spatial transformer, and then one resblock
        self.middle_block = TimestepEmbedSequential(
            ResBlock(in_channels=in_ch, time_emb_dim=time_emb_dim, dropout=dropout),
            SpatialTransformer(in_ch, n_heads=n_heads, d_head=d_head, n_layers=n_layers, dropout=dropout, context_dim=context_dim),
            ResBlock(in_channels=in_ch, time_emb_dim=time_emb_dim, dropout=dropout)
        )
        # ==============================================================
        # ======================output blocks===========================
        # ==============================================================
        self.output_blocks = nn.ModuleList([])
        for level in reversed(range(level)):
            #* add resblocks
            for _ in range(num_res_blocks + 1):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                #* note: here should add input_block_channels.pop() because input blocks are used as skip connection
                out_ch = channels_list[level]
                layers = [ResBlock(in_channels=in_ch + self.input_block_channels.pop(), out_channels=out_ch, time_emb_dim=time_emb_dim, dropout=dropout)]
                in_ch = out_ch
                #* add attention layers
                if attn_mult in attention_resolutions:
                    d_head = in_ch // n_heads
                    layers.append(SpatialTransformer(in_ch, n_heads=n_heads, 
                                                     d_head=d_head, n_layers=n_layers, 
                                                     dropout=dropout, context_dim=context_dim))
                self.output_blocks.append(TimestepEmbedSequential(*layers))
            # *add UpSample except the last one, note that in reversed order, level==0 is the last
            #@ note: both openai and labmlai use num_res_blocks here and placed it in the inner loop, this can be simply copied from upsample...
            if level != 0:
                self.output_blocks.append(TimestepEmbedSequential(UpSample(in_ch)))
                attn_mult //= 2
        # ==============================================================
        # ======================final output============================
        # ==============================================================
        self.out = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_channels=in_ch, out_channels=self.out_channels, kernel_size=3, padding=1)
        )

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
        return sinusoidal_time_step_embedding(time_steps, self.channels, max_len)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, context: torch.Tensor=None):
        """
        forward pass

        Args:
            - x (torch.Tensor):   
                  input feature map of shape `[batch_size, channels, width, height]`
            - timesteps (torch.Tensor):   
                  time steps of shape `[batch_size]`
            - context (torch.Tensor, optional):   
                  cross attention context of shape `[batch_size, seq_len, context_dim]`. Default: None.
        
        Returns:
            - torch.Tensor:   
                  output feature map of shape `[batch_size, channels, width, height]`
        """
        # check parameters
        if context is None:
            assert context.shape[-1] == self.context_dim, f"context dim from passed in context({context.shape}) should be equal to self.context_dim({self.context_dim})"

        # store input blocks for skip connection
        x_input_blocks = []
        #* Get time embedding
        time_emb = self.time_step_embedding(timesteps) 
        time_emb = self.time_emb(time_emb)
        #* input blocks
        for module in self.input_blocks:
            x = module(x, time_emb, context)
            x_input_blocks.append(x)
        #* bottleneck
        x = self.middle_block(x, time_emb, context)
        #* output blocks
        for module in self.output_blocks:
            # skip connection from input blocks
            x = torch.cat([x,x_input_blocks.pop()], dim=1)
            x = module(x, time_emb,context)
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
    # test upsample
    # upsample = UpSample(in_channels=channels, )
    # up = upsample(x)
    # print(up.shape)
    # # test downsample
    # downsample = DownSample(in_channels=channels, )
    # down = downsample(x)
    # print(down.shape)
    # # test resblock
    # resblock = ResBlock(in_channels=channels, out_channels=out_channels, time_emb_dim=tim_emb_dim)
    # res = resblock(x, time_emb=torch.ones(size=(batch, tim_emb_dim)))
    # print(res.shape)
    # test U-Net
    context = torch.ones((batch, seq_len, context_dim))
    unet = UNetModel(in_channels=in_channels, out_channels=out_channels, channels=channels, num_res_blocks=2, n_heads=4,
                     attention_resolutions=(1, 2), channel_mult=[1,2], dropout=0.1,  n_layers=2, context_dim=context_dim)
    out = unet(x, timestep, context)
    print(out.shape)
