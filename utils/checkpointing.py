from dataclasses import MISSING, dataclass, field
from typing import Optional

from stable_diffusion.dataclass import BaseDataclass


@dataclass
class CheckpointConfig(BaseDataclass):
    output_dir: str = field(
        default="output",
        metadata={"help": "dir to save and load checkpoints"},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "dir to load checkpoints from"},
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
