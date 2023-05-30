import torch
import os
import sys

# TODO: fix this
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from stable_diffusion.models.latent_diffusion import LatentDiffusion
from utils.model_utils import build_models
from utils.parse_args import load_config
from utils.prepare_dataset import detransform, to_img


def sample(
    model: LatentDiffusion,
    image_size=64,
    prompt: str = "",
    time_steps: int = 50,
    guidance_scale: float = 7.5,
    scale_factor=1.0,
    save_dir: str = "output",
    device: str = "cuda",
    weight_dtype=torch.float16,
):
    "Sample an image given prompt"
    model.to(device=device, dtype=weight_dtype)
    # random noise
    # to get the right latent size
    img_util = torch.randn(size=(1, 3, image_size, image_size)).to(
        device, dtype=weight_dtype
    )
    noise = model.autoencoder.encode(img_util).latent_dist.sample()
    noise = torch.rand_like(noise).to(device, dtype=weight_dtype)
    # tokenize prompt
    tokenized_prompt = model.text_encoder.tokenize([prompt]).input_ids.to(device)
    context_emb = model.text_encoder.encode_text(tokenized_prompt)[0].to(
        device, weight_dtype
    )
    x_0 = model.sample(
        noised_sample=noise,
        context_emb=context_emb,
        guidance_scale=guidance_scale,
        scale_factor=scale_factor,
        time_steps=time_steps,
    )
    sample = model.autoencoder.decode(x_0)
    sample = detransform(sample)
    to_img(sample, output_path=save_dir)


if __name__ == "__main__":
    args, cfg = load_config()
    model: LatentDiffusion = build_models(cfg.model)
    sample(model, prompt="a cat", image_size=64, time_steps=50, save_dir="output")
