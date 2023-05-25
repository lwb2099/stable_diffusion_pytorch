from stable_diffusion.models.autoencoder import AutoEncoderKL
from stable_diffusion.models.clip_model import CLIPModel
from stable_diffusion.models.latent_diffusion import LatentDiffusion

from stable_diffusion.models.scheduler import DDPMScheduler
from stable_diffusion.models.unet import UNetModel


def add_model_args(parser):
    model_group = parser.add_argument_group("model")
    UNetModel.add_unet_args(model_group)
    DDPMScheduler.add_ddpm_args(model_group)
    CLIPModel.add_clip_args(model_group)
    AutoEncoderKL.add_autoencoder_args(model_group)

    return model_group


def build_models(model_cfg):
    noise_scheduler = DDPMScheduler(model_cfg.ddpm)
    unet = UNetModel(model_cfg.autoencoder.latent_channels, model_cfg.unet)
    text_encoder = CLIPModel(model_cfg.clip)
    autoencoder = AutoEncoderKL(model_cfg.autoencoder)
    # Freeze vae and text_encoder
    autoencoder.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return LatentDiffusion(
        unet,
        autoencoder,
        text_encoder,
        noise_scheduler,
    )


def get_model_param_count(model, trainable_only=False):
    """
    Calculate model's total param count. If trainable_only is True then count only those requiring grads
    """

    def numel(p):
        return p.numel()

    return sum(
        numel(p) for p in model.parameters() if not trainable_only or p.requires_grad
    )
