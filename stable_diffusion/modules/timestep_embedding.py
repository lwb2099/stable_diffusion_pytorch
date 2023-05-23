#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   timestep_embedding.py
@Time    :   2023/05/22 22:54:29
@Author  :   Wenbo Li
@Desc    :   Time step embedding implement
'''

import torch
import math



def sinusoidal_time_step_embedding(time_steps: torch.Tensor, emb_dim: int, max_len: int=10000) -> torch.Tensor:
    """
    sinusoidal time step embedding implementation

    Args:
        - time_steps (torch.Tensor):   
                time step of shape [batch_size,]
        - emb_dim (int):   
                embed dimension of shape [batch,]
        - max_len (int, optional):   
                max len of embedding. Default: 10000.

    Returns:
        - torch.Tensor:   
                time embedding of shape [batch, emb_dim]
    """
    # half is sin, the other half is cos
    half = emb_dim // 2
    freq = torch.exp(
        math.log(max_len) / (half) *  torch.arange(0, end=half, dtype=torch.float32) # get the position of each time step
    ).to(time_steps.device)
    # shape=[batch, 1]*[1, half] = [batch, half]
    # freq[None]: shape=[half] -> [1, half]
    # time_steps[:, None]: shape=[batch] -> [batch, 1]
    args = time_steps[:, None].float() * freq[None]
    # shape=[batch, emb_dim]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
