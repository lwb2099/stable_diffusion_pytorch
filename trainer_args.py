#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   trainer_args.py
@Time    :   2023/05/26 20:11:49
@Author  :   Wenbo Li
@Desc    :   Args for logging, training, optimizer, scheduler, etc.
"""

from dataclasses import dataclass, field
from typing import Optional

from stable_diffusion.dataclass import BaseDataclass


@dataclass
class LogConfig(BaseDataclass):
    logging_dir: str = field(default="logs", metadata={"help": "log directory"})
    with_tracking: bool = field(
        default=False, metadata={"help": "whether enable tracker"}
    )
    report_to: str = field(
        default="wandb",
        metadata={"help": "tracker to use, only enabled when passed in --with_tracker"},
    )


@dataclass
class TrainConfig(BaseDataclass):
    seed: int = field(default=0, metadata={"help": "seed argument"})
    train_batch_size: int = field(
        default=1, metadata={"help": "train batch size per processor"}
    )
    max_train_steps: int = field(
        default=1000,
        metadata={"help": "total train steps, if provided, overrides max_train_epochs"},
    )
    max_train_epochs: int = field(default=100, metadata={"help": "max train epochs"})
    eval_batch_size: int = field(
        default=1, metadata={"help": "eval batch size per processor"}
    )
    log_interval: int = field(
        default=0,
        metadata={
            "help": "do evaluation every n steps, default 0 means no evaluation during training"
        },
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "gradient accumulation steps"}
    )
    use_deepspeed: bool = field(
        default=False, metadata={"help": "whether use deepspeed"}
    )


@dataclass
class OptimConfig(BaseDataclass):
    learning_rate: float = field(
        default=1e-4, metadata={"help": "learning rate argument"}
    )
    adam_weight_decay: float = field(
        default=0.1, metadata={"help": "Adam weight decay argument"}
    )
    use_8bit_adam: bool = field(
        default=False, metadata={"help": "Use 8-bit Adam argument"}
    )
    max_grad_norm: float = field(
        default=0.1, metadata={"help": "max grad norm argument"}
    )
    scheduler_type: str = field(
        default="linear", metadata={"help": "scheduler type argument"}
    )
    lr_warmup_steps: int = field(
        default=0, metadata={"help": "learning rate warm-up steps argument"}
    )


# below are deprecated, now we use dataclass


def add_distributed_training_args(parser):
    train_group = parser.add_argument_group("train")
    train_group.add_argument(
        "--logging_dir", type=str, default="logs", help="log directory"
    )
    train_group.add_argument(
        "--with_tracker",
        type=str,
        default=None,
    )
    train_group.add_argument("--report_to", type=int, default=0, help="seed argument")
    train_group.add_argument("--seed", type=int, default=0)
    train_group.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
    )
    train_group.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="total train steps, if provided, overrides max_train_epochs",
    )
    train_group.add_argument(
        "--max_train_epochs",
        type=int,
        default=100,
        help="max train epochs, orverides by max_training_steps",
    )
    train_group.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
    )
    train_group.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    return train_group


def add_optimization_args(parser):
    optim_group = parser.add_argument_group("optim")
    optim_group.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    optim_group.add_argument("--adam_weight_decay", type=float, default=0.1)
    optim_group.add_argument(
        "--use_8bit_adam",
        action="store_true",
        default=False,
    )
    return optim_group


def add_lr_scheduler_args(parser):
    lr_scheduler_group = parser.add_argument_group("lr_scheduler")
    lr_scheduler_group.add_argument(
        "--type",
        type=str,
        default="linear",
    )
    lr_scheduler_group.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
    )
    return lr_scheduler_group
