import numpy as np
from torch import nn
import torch
from typing import Optional, List
from stable_diffusion.models.latent_diffusion import LatentDiffusion
from abc import ABC, abstractmethod

class DiffusionSampler(ABC):
    def __init__(self, model: LatentDiffusion):
        super().__init__()
        self.model = model
        self.noise_steps = model.noise_steps

    def prepare_noise_schedule(self, noise_steps: torch.Tensor, device: torch.device):
        pass

    def get_eps(self, 
                x: torch.Tensor, 
                time_step: torch.Tensor, 
                condition: torch.Tensor, 
                scale_factor: float=1.,
                uncondition: Optional[torch.Tensor]=None,
                ):
        if uncondition is None or scale_factor == 1.:
            e_t = self.model(x, time_step, condition)
        else:
            t_in = torch.cat([time_step] * 2)
            x_in = torch.cat([x] * 2)
            c_in = torch.cat([uncondition, condition])
            e_t, e_t_uncond = self.model(x_in, t_in, c_in).chuck(2)
            e_t = e_t_uncond + scale_factor * (e_t - e_t_uncond)
            return e_t

    @abstractmethod
    def sample(self,
               x_prev: torch.Tensor,
               condition: torch.Tensor,
               noise: torch.Tensor,
               temperature: float=1.,
               scale_factor: float=1.,
               uncondition: Optional[torch.Tensor]=None,
               skip_steps: int=0,
               ):
        raise NotImplementedError

    @abstractmethod
    def q_sample(self,
                    x_prev: torch.Tensor,
                    condition: torch.Tensor,
                    noise: torch.Tensor,
                    temperature: float=1.,
                    scale_factor: float=1.,
                    uncondition: Optional[torch.Tensor]=None,
                    skip_steps: int=0,
                    ):
        raise NotImplementedError



class DDPMSampler(nn.Module):
    def __init__(self, model: LatentDiffusion):
        super().__init__()
        self.model = model

    def sample(self,
               shape: List[int],
               cond: torch.Tensor,
               repeat_noise: bool = False,
               temperature: float = 1.,
               x_last: Optional[torch.Tensor] = None,
               uncond_scale: float = 1.,
               uncond_cond: Optional[torch.Tensor] = None,
               skip_steps: int = 0,
               ):
        """sample latent (x_0) from noise (x_T)"""
        device = self.model.device
        bs = shape[0]

        # Get $x_T$
        x = x_last if x_last is not None else torch.randn(shape, device=device)

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$
        time_steps = np.flip(self.time_steps)[skip_steps:]

        # Sampling loop
        for step in time_steps:
            # Time step $t$
            ts = x.new_full((bs,), step, dtype=torch.long)

            # Sample $x_{t-1}$
            x, pred_x0, e_t = self.p_sample(x, cond, ts, step,
                                            repeat_noise=repeat_noise,
                                            temperature=temperature,
                                            uncond_scale=uncond_scale,
                                            uncond_cond=uncond_cond)

        # Return $x_0$
        return x
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, c: torch.Tensor, t: torch.Tensor, step: int,
                 repeat_noise: bool = False,
                 temperature: float = 1.,
                 uncond_scale: float = 1., uncond_cond: Optional[torch.Tensor] = None):
        """
        Decode from x_t to get x_{t-1}
        ### Sample $x_{t-1}$ from $p_\theta(x_{t-1} | x_t)$

        :param x: is $x_t$ of shape `[batch_size, channels, height, width]`
        :param c: is the conditional embeddings $c$ of shape `[batch_size, emb_size]`
        :param t: is $t$ of shape `[batch_size]`
        :param step: is the step $t$ as an integer
        :repeat_noise: specified whether the noise should be same for all samples in the batch
        :param temperature: is the noise temperature (random noise gets multiplied by this)
        :param uncond_scale: is the unconditional guidance scale $s$. This is used for
            $\epsilon_\theta(x_t, c) = s\epsilon_\text{cond}(x_t, c) + (s - 1)\epsilon_\text{cond}(x_t, c_u)$
        :param uncond_cond: is the conditional embedding for empty prompt $c_u$
        """

        # Get $\epsilon_\theta$
        e_t = self.get_eps(x, t, c,
                           uncond_scale=uncond_scale,
                           uncond_cond=uncond_cond)

        # Get batch size
        bs = x.shape[0]

        # $\frac{1}{\sqrt{\bar\alpha_t}}$
        sqrt_recip_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_alpha_bar[step])
        # $\sqrt{\frac{1}{\bar\alpha_t} - 1}$
        sqrt_recip_m1_alpha_bar = x.new_full((bs, 1, 1, 1), self.sqrt_recip_m1_alpha_bar[step])

        # Calculate $x_0$ with current $\epsilon_\theta$
        #
        # $$x_0 = \frac{1}{\sqrt{\bar\alpha_t}} x_t -  \Big(\sqrt{\frac{1}{\bar\alpha_t} - 1}\Big)\epsilon_\theta$$
        x0 = sqrt_recip_alpha_bar * x - sqrt_recip_m1_alpha_bar * e_t

        # $\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}$
        mean_x0_coef = x.new_full((bs, 1, 1, 1), self.mean_x0_coef[step])
        # $\frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}$
        mean_xt_coef = x.new_full((bs, 1, 1, 1), self.mean_xt_coef[step])

        # Calculate $\mu_t(x_t, t)$
        #
        # $$\mu_t(x_t, t) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0
        #    + \frac{\sqrt{\alpha_t}(1 - \bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
        mean = mean_x0_coef * x0 + mean_xt_coef * x
        # $\log \tilde\beta_t$
        log_var = x.new_full((bs, 1, 1, 1), self.log_var[step])

        # Do not add noise when $t = 1$ (final step sampling process).
        # Note that `step` is `0` when $t = 1$)
        if step == 0:
            noise = 0
        # If same noise is used for all samples in the batch
        elif repeat_noise:
            noise = torch.randn((1, *x.shape[1:]))
        # Different noise for each sample
        else:
            noise = torch.randn(x.shape)

        # Multiply noise by the temperature
        noise = noise * temperature

        # Sample from,
        #
        # $$p_\theta(x_{t-1} | x_t) = \mathcal{N}\big(x_{t-1}; \mu_\theta(x_t, t), \tilde\beta_t \mathbf{I} \big)$$
        x_prev = mean + (0.5 * log_var).exp() * noise

        #
        return x_prev, x0, e_t

    @torch.no_grad()
    def q_sample(self, x0: torch.Tensor, index: int, noise: Optional[torch.Tensor] = None):
        """
        Add noise to $x_0$ to get $x_t$.
        ### Sample from $q(x_t|x_0)$
        $$q(x_t|x_0) = \mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$$

        :param x0: is $x_0$ of shape `[batch_size, channels, height, width]`
        :param index: is the time step $t$ index
        :param noise: is the noise, $\epsilon$
        """

        # Random noise, if noise is not specified
        if noise is None:
            noise = torch.randn_like(x0)

        # Sample from $\mathcal{N} \Big(x_t; \sqrt{\bar\alpha_t} x_0, (1-\bar\alpha_t) \mathbf{I} \Big)$
        return self.sqrt_alpha_bar[index] * x0 + self.sqrt_1m_alpha_bar[index] * noise
    

