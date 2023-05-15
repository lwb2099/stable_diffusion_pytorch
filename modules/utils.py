#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Time    :   2023/05/15 12:55:11
@Author  :   Wenbo Li
@Desc    :   Util class for models
'''

import torch
import torch.nn as nn
import math

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

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



if __name__ == "__main__":
    time_steps = torch.arange(0, 10)
    emb_dim = 32
    max_len = 100
    print(sinusoidal_time_step_embedding(time_steps, emb_dim, max_len).shape)