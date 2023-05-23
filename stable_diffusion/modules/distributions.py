import torch
from torch import nn


class GaussianDistribution(nn.Module):
    def __init__(self, param: torch.Tensor):
        super(GaussianDistribution, self).__init__
        self.mean, self.log_var = torch.chunk(param, 2, dim=1)
    
    def sample(self):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn_like(std)
        return self.mean + eps * std