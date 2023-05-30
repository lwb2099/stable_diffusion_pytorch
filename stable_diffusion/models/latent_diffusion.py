from typing import Optional
import torch
from torch import nn
from tqdm import tqdm

from stable_diffusion.models.autoencoder import AutoEncoderKL
from stable_diffusion.models.clip_model import CLIPModel
from stable_diffusion.models.scheduler import DDPMScheduler
from stable_diffusion.models.unet import UNetModel


class LatentDiffusion(nn.Module):
    def __init__(
        self,
        unet: UNetModel,
        autoencoder: AutoEncoderKL,
        text_encoder: CLIPModel,
        noise_scheduler: DDPMScheduler,
    ):
        """main class"""
        super().__init__()
        self.unet = unet
        self.autoencoder = autoencoder
        self.text_encoder = text_encoder
        self.noise_scheduler = noise_scheduler

    def pred_noise(
        self,
        noised_sample: torch.Tensor,
        time_step: torch.Tensor,
        context_emb: torch.Tensor,
        guidance_scale: float = 1.0,
    ):
        """
        predicts the noise added on latent vector at time step=t

        Args:
            - x (torch.Tensor):
                  noised latent vector, shape = `[batch, channels, height, width]`
            - time_steps (torch.Tensor):
                  time steps, shape = `[batch]`
            - context_emb (torch.Tensor):
                  conditional embedding, shape = `[batch, seq_len, d_model]`

        Returns:
            - pred noise (torch.Tensor):
                  predicted noise, shape = `[batch, channels, height, width]`
        """
        do_classifier_free_guidance = guidance_scale > 1
        if not do_classifier_free_guidance:
            return self.unet(noised_sample, time_step, context_emb)
        t_in = torch.cat([time_step] * 2)
        x_in = torch.cat([noised_sample] * 2)
        bsz = noised_sample.shape[0]
        tokenized_text = self.text_encoder.tokenize([""] * bsz).input_ids.to(
            noised_sample.device
        )
        uncond_emb = self.text_encoder.encode_text(
            tokenized_text  # adding `.to(self.weight_dtype)` causes error...
        )[0]
        c_in = torch.cat([uncond_emb, context_emb])
        pred_noise_cond, pred_noise_uncond = torch.chunk(
            self.unet(x_in, t_in, c_in), 2, dim=0
        )
        return pred_noise_cond + guidance_scale * (pred_noise_cond - pred_noise_uncond)

    def sample(
        self,
        noised_sample: torch.Tensor,
        context_emb: torch.Tensor,
        guidance_scale: float = 7.5,
        repeat_noise: bool = False,
        scale_factor: float = 1.0,
        time_steps: Optional[int] = None,
    ):
        """
        Sample loop to get x_0 given x_t and conditional embedding

        Args:
            - noised_sample (torch.Tensor):
                  original x_t of shape=`[batch, channels, height, width]`
            - context_emb (torch.Tensor):
                  conditional embedding of shape=`[batch, seq_len, context_dim]`
            - guidence_scale (float, optional):
                  scale used for classifer free guidance. Default: `7.5`.
            - repeat_noise (bool, optional):
                  whether use the same noise in a batch duuring each p_sample. Default: `False`.
            - scale_factor (float, optional):
                  scaling factor of noise. Default: `1.0`.

        Returns:
            - x_0 (torch.Tensor):
                  denoised latent x_0 of shape=`[batch, channels, height, width]`
        """
        bsz = noised_sample.shape[0]

        # Get x_T
        x = noised_sample

        # Time steps to sample at $T - t', T - t' - 1, \dots, 1$

        # Sampling loop
        if time_steps is not None:
            noise_time_steps = range(time_steps, 0, -1)
        else:
            noise_time_steps = self.noise_scheduler.noise_time_steps
        progress_bar = tqdm(reversed(noise_time_steps), desc="Sampling")
        for step in progress_bar:
            # fill time step t from int to tensor of shape=`[batch]`
            time_step = x.new_full((bsz,), step, dtype=torch.long)
            pred_noise = self.pred_noise(
                noised_sample=x,
                time_step=time_step,
                context_emb=context_emb,
                guidance_scale=guidance_scale,
            )
            # Sample x_{t-1}
            x, pred_x0 = self.noise_scheduler.step(
                pred_noise=pred_noise,
                x_t=x,
                time_step=step,
                repeat_noise=repeat_noise,
                scale_factor=scale_factor,
            )
        # Return $x_0$
        return x
