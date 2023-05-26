#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   checkpointing_args.py
@Time    :   2023/05/26 20:09:15
@Author  :   Wenbo Li
@Desc    :   dataclass to store arguments for ckpt management
"""

from dataclasses import dataclass, field
from typing import Optional

from stable_diffusion.dataclass import BaseDataclass


@dataclass
class CheckpointConfig(BaseDataclass):
    output_dir: str = field(
        default="model",
        metadata={"help": "dir to save and load checkpoints"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "dir to load checkpoints from"},
    )
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps",
        },
    )


# deprecated


def add_checkpoint_args(parser):
    checkpoint_group = parser.add_argument_group("checkpoint")
    checkpoint_group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="dir to load checkpoints from",
    )
    checkpoint_group.add_argument(
        "--output_dir",
        type=str,
        default="dir to save and load checkpoints",
    )
