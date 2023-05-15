#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   attention.py
@Time    :   2023/05/14 16:03:43
@Author  :   Wenbo Li
@Desc    :   Transformer module for Stable Diffusion U-Net
'''

from typing import Optional
import torch
import numpy as np
import torch.nn as nn

from utils import zero_module

class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int], n_heads: int, d_head: int, dropout: float=0.):
        super().__init__()
        if not context_dim:
            context_dim = query_dim
        d_attn = n_heads * d_head
        self.n_heads = n_heads
        self.scale = 1 / (d_head ** 0.5)  # 1 / sqrt(d_k) from paper
        self.to_q = nn.Linear(query_dim, d_attn, bias=False)
        self.to_k = nn.Linear(context_dim, d_attn, bias=False)
        self.to_v = nn.Linear(context_dim, d_attn, bias=False)
        self.out = nn.Sequential(
            nn.Linear(d_attn, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor=None, mask: torch.Tensor=None):
        """
        Cross attention forward pass

        Args:
            - query (torch.Tensor):   
                  feature map of shape `[batch, height*width, d_model]`
            - context (torch.Tensor, optional):   
                  conditional embeddings of shape `[batch, seq_len, context_dim]`. Default: `None`.
            - mask (torch.Tensor, optional):   
                  _description_. Default: `None`.
        """
        # no context, equal to self-attention
        if not context:
            context = query
        Q, K, V = self.to_q(query), self.to_k(context), self.to_v(context)
        # q,k,v: [batch, seq_len, d_attn] -> [batch, d_model, height, width]
        Q, K, V = map(lambda t: torch.rearange(t.reshape("b n (n_heads d_head) -> (b n_heads) n d_head", n_heads=self.n_heads)), [Q, K, V])


class BasicTransformerBlock(nn.Module):
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 d_head: int,
                 dropout: float=0.,
                 context_dim: int=768
                 ):
        super().__init__()
        self.attn1 = CrossAttention(d_model, d_model, n_heads, d_head,)



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and #? reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image

    Args:
    - in_channels (int):   
        input num of channels in the feature map 
    - n_heads (int):   
        num of attention heads
    - d_head (int):   
        dim of each head
    - n_layer (int, optional):   
        num of transformer block. Default: `1`.
    - dropout (float, optional):   
        dropout rate. Default: `0.`.
    - context_dim (int, optional):   
        dim of context condition. Default: `None`.
    """
    def __init__(self, 
            in_channels: int, 
            n_heads: int, 
            d_head: int,
            n_layers: int=1, 
            dropout: float=0., 
            context_dim: int=None
            ):
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.proj_in = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        # Transformer layers
        #@ note: origin openai code use inner_dim = n_heads * d_head, but if legacy, d_head = in_channels // n_heads
        # => here we use in_channels for simiplicity 
        self.transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(in_channels, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for _ in range(n_layers)])
        self.proj_out = zero_module(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0))

    def forward(self, x: torch.Tensor, context: torch.Tensor=None) -> torch.Tensor:
        """
        forward pass

        Args:
            - x (torch.Tensor):   
                  feature map of shape `[batch_size, channels, height, width]`
            - context (torch.Tensor, optional):   
                  conditional embeddings of shape `[batch_size,  seq_len, context_dim]`. Default: `None`.

        Returns:
            - torch.Tensor:   
                  shape=`[batch_size, channels, height, width]`
        """
        # use for skip connection
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = torch.rearange(x, "b c h w -> b (h w) c")
        for module in self.transformer_blocks:
            x = module(x, context=context)
        x = torch.rearange(x, "b (h w) c -> b c h w", h=x_in.shape[2])
        x = self.proj_out(x)
        return x + x_in
