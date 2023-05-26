from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn

from dataclasses import dataclass, field
from typing import Optional

from stable_diffusion.dataclass import BaseDataclass


@dataclass
class ClipConfig(BaseDataclass):
    tokenizer: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "Tokenizer to use for text encoding."},
    )
    text_encoder: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={"help": "Text encoder model to use."},
    )
    max_seq_len: int = field(
        default=77, metadata={"help": "Maximum sequence length for tokenized text."}
    )
    model_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a directory to store the pretrained CLIP model."},
    )


class CLIPModel(nn.Module):
    @staticmethod
    def add_clip_args(model_parser):
        clip_group = model_parser.add_argument_group("clip")
        clip_group.add_argument(
            "--tokenizer",
            type=str,
            default="openai/clip-vit-base-patch32",
        )
        clip_group.add_argument(
            "--text_encoder",
            type=str,
            default="openai/clip-vit-base-patch32",
        )
        clip_group.add_argument(
            "--max_seq_len",
            type=int,
            default=77,
        )
        clip_group.add_argument(
            "--cache_dir",
            type=str,
            default=None,
            help="Path to a directory to store the pretrained clip model",
        )
        return clip_group

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.max_seq_len = cfg.max_seq_len
        self.text_encoder: CLIPTextModel = CLIPTextModel.from_pretrained(
            cfg.text_encoder, cache_dir=cfg.model_dir, subfolder="text_encoder"
        )
        self.tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(
            cfg.tokenizer,
            use_fast=False,
            cache_dir=cfg.model_dir,
            subfolder="tokenizer",
        )

    def tokenize(
        self,
        prompt: str = "",
        max_length: int = None,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        return self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
            max_length=(self.max_seq_len if max_length is None else max_length),
        )

    def encode_text(self, text):
        """Encode text to text embedding
        Args:
            - text (str):
                  text to encode, shape = [batch, seq_len]
        Returns:
            - text_embedding (torch.Tensor):
                  text embedding, shape = [batch, seq_len, d_model]
        """
        return self.text_encoder(text)
