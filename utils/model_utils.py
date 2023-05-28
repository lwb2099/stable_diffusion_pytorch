#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   model_utils.py
@Time    :   2023/05/26 20:08:47
@Author  :   Wenbo Li
@Desc    :   util script to load and build models
"""

from stable_diffusion.models.autoencoder import AutoEncoderKL
from stable_diffusion.models.clip_model import CLIPModel
from stable_diffusion.models.latent_diffusion import LatentDiffusion

from stable_diffusion.models.scheduler import DDPMScheduler
from stable_diffusion.models.unet import UNetModel


# deprecated
def add_model_args(parser):
    model_group = parser.add_argument_group("model")
    UNetModel.add_unet_args(model_group)
    DDPMScheduler.add_ddpm_args(model_group)
    CLIPModel.add_clip_args(model_group)
    AutoEncoderKL.add_autoencoder_args(model_group)

    return model_group


def build_models(model_cfg, logger=None):
    noise_scheduler = DDPMScheduler(model_cfg.ddpm)
    unet = UNetModel(
        model_cfg.autoencoder.latent_channels,
        model_cfg.autoencoder.groups,
        model_cfg.unet,
    )
    text_encoder = CLIPModel(model_cfg.clip)
    text_encoder.requires_grad_(False)
    autoencoder = AutoEncoderKL(model_cfg.autoencoder)

    count_params(unet, logger=logger)
    count_params(text_encoder, logger=logger)
    count_params(autoencoder, logger=logger)

    return LatentDiffusion(
        unet,
        autoencoder,
        text_encoder,
        noise_scheduler,
    )


def count_params(model, trainable_only=True, logger=None):
    """
    Copied from original stable diffusion code [ldm/util.py](https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/util.py#L71)
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """

    def numel(p):
        return p.numel()

    total_params = sum(
        numel(p) for p in model.parameters() if not trainable_only or p.requires_grad
    )
    if logger is not None:
        logger.info(
            f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M {'trainable' if trainable_only else ''} params."
        )
    return total_params
