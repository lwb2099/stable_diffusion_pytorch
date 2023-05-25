from transformers import CLIPTextModel, CLIPTokenizer
from torch import nn

from dataclasses import dataclass, field
from typing import Optional

from stable_diffusion.dataclass import BaseDataclass


@dataclass
class ClipConfig(BaseDataclass):
    tokenizer: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "Tokenizer to use for text encoding."},
    )
    text_encoder: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "Text encoder model to use."},
    )
    max_seq_len: int = field(
        default=77, metadata={"help": "Maximum sequence length for tokenized text."}
    )
    cache_dir: Optional[str] = field(
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
        """main class"""
        super().__init__()
        self.max_seq_len = cfg.max_seq_len
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.text_encoder, cache_dir=cfg.cache_dir
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.tokenizer, use_fast=False, cache_dir=cfg.cache_dir
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
        if isinstance(text, str):
            text = self.tokenize(text)
        return self.text_encoder(text.input_ids)[0]

    def forward(self, text):
        return self.text_encoder(text)
