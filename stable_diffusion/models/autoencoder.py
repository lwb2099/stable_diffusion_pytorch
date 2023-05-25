from typing import List
from torch import nn

from stable_diffusion.dataclass import BaseDataclass

from ..modules.distributions import GaussianDistribution
import torch

from .utils import (
    build_conv_in,
    build_input_blocks,
    build_bottleneck,
    build_output_blocks,
    build_final_output,
)


from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AutoencoderConfig(BaseDataclass):
    in_channels: int = field(
        default=3, metadata={"help": "Number of input channels of the input image."}
    )
    latent_channels: int = field(
        default=4, metadata={"help": "Embedding channels of the latent vector."}
    )
    out_channels: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of output channels of the decoded image. Should be the same as in_channels."
        },
    )
    autoencoder_channels_list: List[int] = field(
        default_factory=lambda: [64, 128],
        metadata={
            "help": "Comma-separated list of channel multipliers for each level."
        },
    )
    autoencoder_num_res_blocks: int = field(
        default=2, metadata={"help": "Number of residual blocks per level."}
    )


class AutoEncoderKL(nn.Module):
    @staticmethod
    def add_autoencoder_args(parser):
        autoencoder = parser.add_argument_group("autoencoder")
        autoencoder.add_argument(
            "--in_channels",
            type=int,
            default=3,
            help="number of input channels of input image",
        )
        autoencoder.add_argument(
            "--latent_channels",
            type=int,
            default=4,
            help="embedding channels of latent vector",
        )
        autoencoder.add_argument(
            "--out_channels",
            type=int,
            default=None,
            help="number of output channels of decoded image, should be the same with in_channels",
        )
        autoencoder.add_argument(
            "--autoencoder_channels_list",
            type=int,
            nargs="+",
            default=[64, 128],
            help="comma separated list of channels multipliers for each level",
        )
        autoencoder.add_argument(
            "--autoencoder_num_res_blocks",
            type=int,
            default=2,
            help="number of residual blocks per level",
        )

    def __init__(
        self,
        cfg,
    ):
        # check params
        assert (
            cfg.out_channels is None or cfg.out_channels == cfg.in_channels
        ), f"input channels({cfg.input_channels}) of image should be equal to output channels({cfg.out_channels})"
        super(AutoEncoderKL, self).__init__()
        self.latent_channels = latent_channels = cfg.latent_channels
        self.encoder = self.build_encoder(cfg)
        self.decoder = self.build_decoder(cfg)
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    @classmethod
    def build_encoder(cls, cfg):
        return Encoder(
            in_channels=cfg.in_channels,
            out_channels=cfg.latent_channels,
            channels_list=cfg.autoencoder_channels_list,
            num_res_blocks=cfg.autoencoder_num_res_blocks,
        )

    @staticmethod
    def build_decoder(cfg):
        return Decoder(
            in_channels=cfg.latent_channels,
            out_channels=cfg.out_channels or cfg.in_channels,
            channels_list=cfg.channels_list,
            num_res_blocks=cfg.num_res_blocks,
        )

    def encode(self, img: torch.Tensor) -> GaussianDistribution:
        z = self.encoder(img)
        # Get the moments in the quantized embedding space
        moments = self.quant_conv(z)
        # Return the distribution(posterior)
        return GaussianDistribution(moments)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode images from latent representation

        Args:
            - z(torch.Tensor):
                  latent representation with shape `[batch_size, emb_channels, z_height, z_height]`
        """
        # check params
        assert (
            z.shape[1] == self.latent_channels
        ), f"Expected latent representation to have {self.latent_channels} channels, got {z.shape[1]}"
        z = self.post_quant_conv(latent)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_list: List[int],
        num_res_blocks: int,
    ):
        super(Encoder, self).__init__()
        self.conv_in = build_conv_in(in_channels, channels_list[0])
        levels = len(channels_list)
        self.down, mid_ch = build_input_blocks(
            in_channels=in_channels,
            channels=channels_list[0],
            num_res_blocks=num_res_blocks,
            levels=levels,
            channels_list=channels_list,
        )
        self.bottleneck = build_bottleneck(mid_ch, d_head=mid_ch, use_attn_only=True)
        self.out = build_final_output(mid_ch, out_channels)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down(x)
        x = self.bottleneck(x)
        x = self.out(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels_list: List[int],
        num_res_blocks: int,
    ):
        super(Decoder, self).__init__()
        self.conv_in = build_conv_in(in_channels, channels_list[0])
        levels = len(channels_list)
        self.down, mid_ch = build_output_blocks(
            num_res_blocks=num_res_blocks,
            in_ch=in_channels,
            levels=levels,
            channels_list=channels_list,
        )
        self.bottleneck = build_bottleneck(mid_ch, d_head=mid_ch, use_attn_only=True)
        self.out = build_final_output(mid_ch, out_channels)
