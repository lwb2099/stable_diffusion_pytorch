#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   attention.py
@Time    :   2023/05/14 16:03:43
@Author  :   Wenbo Li
@Desc    :   Transformer module for Stable Diffusion U-Net
'''

import torch
import numpy as np
import torch.nn as nn

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    
    Args:
        - in_channels (int):   
                input num of channels in the feature map 
        - n_heads (int):   
                num of attention heads
        - d_head (int):   
                _description_
        - depth (int, optional):   
                _description_. Default: 1.
        - dropout (float, optional):   
                _description_. Default: 0..
        - context_dim (int, optional):   
                _description_. Default: None.
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
        

