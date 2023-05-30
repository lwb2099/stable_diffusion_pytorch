import torch
from torch import nn


class GaussianDistribution(nn.Module):
    def __init__(self, moments: torch.Tensor):
        super(GaussianDistribution, self).__init__
        self.mean, self.log_var = torch.chunk(moments, 2, dim=1)

    def sample(self):
        std = torch.exp(0.5 * self.log_var)
        eps = torch.randn_like(std)
        return self.mean + eps * std

    def kl(self):
        self.var = torch.exp(self.log_var)
        return 0.5 * torch.sum(
            torch.pow(self.mean, 2) + self.var - 1.0 - self.log_var, dim=[1, 2, 3]
        )
