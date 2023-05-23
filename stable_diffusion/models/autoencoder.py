
from torch import nn
from utils import build_conv_in, build_input_blocks, build_bottleneck, build_final_output, build_output_blocks
from modules.distributions import GaussianDistribution
import torch

class Encoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 channels: int,
                 out_channels: int,
                 channels_mult: list,
                 num_res_blocks: int,
                 ):
        super(Encoder, self).__init__()
        self.conv_in = build_conv_in(in_channels, channels)
        levels = len(channels_mult)
        channels_list = [channels * m for m in channels_mult]
        self.down, mid_ch = build_input_blocks(
            in_channels=in_channels,channels=channels,num_res_blocks=num_res_blocks,
            levels=levels, channels_list=channels_list
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
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 out_channels: int,
                 channels_mult: list,
                 num_res_blocks: int,
                 ):
        super(Decoder, self).__init__()
        self.conv_in = build_conv_in(in_channels, channels)
        levels = len(channels_mult)
        channels_list = [channels * m for m in channels_mult]
        self.down, mid_ch = build_output_blocks(
            num_res_blocks=num_res_blocks, in_ch=in_channels, levels=levels, channels_list=channels_list
        )
        self.bottleneck = build_bottleneck(mid_ch, d_head=mid_ch, use_attn_only=True)
        self.out = build_final_output(mid_ch, out_channels)



class AutoEncoderKL(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 emb_channels: int,
                 out_channels: int,
                 ):
        super(AutoEncoderKL, self).__init__()
        self.emb_channels = emb_channels
        self.encoder = encoder
        self.decoder = decoder
        # Convolution to map from embedding space to
        # quantized embedding space moments (mean and log variance)
        self.quant_conv = nn.Conv2d(2 * out_channels, 2 * emb_channels, 1)
        # Convolution to map from quantized embedding space back to
        # embedding space
        self.post_quant_conv = nn.Conv2d(emb_channels, out_channels, 1)

    def encode(self,img: torch.Tensor) -> GaussianDistribution:
        z = self.encode(img)
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
        assert z.shape[1] == self.emb_channels, f"Expected latent representation to have {self.emb_channels} channels, got {z.shape[1]}"
        z = self.post_quant_conv(latent)
        return self.decoder(z)

    #TODO: fill train step and val and log img
    def forward():
        pass


if __name__ == "__main__":
    x = torch.randn(1, 3, 32, 32)
    encoder = Encoder(in_channels=3, channels=64, out_channels=64, channels_mult=[1, 2, 4, 8], num_res_blocks=2)
    print(encoder(x).shape)
    decoder = Decoder(in_channels=64, channels=64, out_channels=3, channels_mult=[1, 2, 4, 8], num_res_blocks=2)
    print(decoder(x).shape)
    model = AutoEncoderKL(encoder=encoder, decoder=decoder, emb_channels=64, out_channels=64)
    print(model(x).shape)
