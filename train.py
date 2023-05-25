#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2023/05/12 19:24:02
@Author  :   Wenbo Li
@Desc    :   Main Class for training stable diffusion model
"""

import argparse

# from omegacli import OmegaConf

from omegaconf import OmegaConf
import math
import os

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
from diffusers.optimization import get_scheduler
from omegaconf import DictConfig
from stable_diffusion.models.latent_diffusion import LatentDiffusion
from utils.checkpointing import add_checkpoint_args
from utils.model_utils import add_model_args, build_models
from utils.parse_args import load_config
from utils.prepare_dataset import add_dataset_args, get_dataset
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# build environment
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
logger = get_logger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class StableDiffusionTrainer:
    def __init__(
        self,
        model: LatentDiffusion,
        args,
        train_dataset,
        eval_dataset,
        collate_fn,
    ):
        # check params
        assert isinstance(
            train_dataset, Dataset
        ), f"training requires a training dataset, but got train_dataset of type {type(train_dataset)}"
        assert eval_dataset is not None, "must specify an evaluation dataset"
        self.model: LatentDiffusion = model
        self.args = args
        self.train_dataset: Dataset = train_dataset
        self.eval_dataset: Dataset = eval_dataset
        # * 1. init accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.train.gradient_accumulation_steps,
            log_with=args.train.tracker,
            project_dir=args.train.logging_dir,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        # * 2. set seed
        if args.seed is not None:
            set_seed(args.seed)
        # * 4. build optimizer and lr_scheduler
        self.optimizer = self.__build_optimizer(args.optim)
        self.lr_scheduler = get_scheduler(
            args.lr_scheduler.type,
            optimizer=self.optimizer,
            num_warmup_steps=args.optim.lr_warmup_steps
            * args.train.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps
            * args.train.gradient_accumulation_steps,
        )
        # * 5. get dataset
        self.train_dataloader = self.get_dataloader(
            train_dataset=self.train_dataset,
            collate_fn=collate_fn,
            train_batch_size=args.train.train_batch_size,
            num_workers=self.accelerator.num_processes,
        )
        self.eval_dataloader = self.get_dataloader(
            eval_dataset=self.eval_dataset,
            collate_fn=collate_fn,
            train_batch_size=args.train.eval_batch_size,
            num_workers=self.accelerator.num_processes,
        )

        # * 5. Prepare everything with our `accelerator`.
        (
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_dataloader,
            self.eval_dataloader,
            self.lr_scheduler,
        )
        # * enable variable annotations so that we can easily debug
        self.model: LatentDiffusion = self.model
        self.train_dataloader: DataLoader = self.train_dataloader
        self.eval_dataloader: DataLoader = self.eval_dataloader
        # * 6. Move text_encoder and autoencoder to gpu and cast to weight_dtype
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        self.model.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.model.autoencoder.to(self.accelerator.device, dtype=self.weight_dtype)

    def get_dataloader(
        self, dataset, collate_fn, batch_size, num_workers
    ) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def __build_optimizer(self, optim_args):
        # Initialize the optimizer
        if optim_args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                ) from e

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        return optimizer_cls(
            # only train unet
            self.model.unet.parameters(),
            lr=optim_args.learning_rate,
            weight_decay=optim_args.adam_weight_decay,
        )

    def __resume_state(self, args):
        """
        Load checkpoints and resume train states
        """
        if args.resume_from_checkpoint:
            ckpt_path = os.path.basename(args.resume_from_checkpoint)
        else:
            # None, Get the most recent checkpoint or start from scratch
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(
                dirs, key=lambda x: int(x.split("-")[1])
            )  # checkpoint-100 => 100
            ckpt_path = dirs[-1] if len(dirs) > 0 else None
        if ckpt_path is None:
            self.accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            self.accelerator.print(f"Resuming from checkpoint {ckpt_path}")
            self.accelerator.load_state(os.path.join(args.output_dir, ckpt_path))
        # * Calculate train steps
        # total batch num / accumulate steps => actual update steps per epoch
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / args.gradient_accumulation_steps
        )
        if args.max_train_steps is None:
            args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
        else:
            # override max_train_epochs
            args.max_train_epochs = math.ceil(
                args.max_train_steps / num_update_steps_per_epoch
            )
        # actual updated steps
        self.global_step = int(ckpt_path.split("-")[1]) if ckpt_path else 0
        self.start_epoch = (
            self.global_step // num_update_steps_per_epoch if ckpt_path else 0
        )
        # * change diffusers implementation: 20 % 6 = 2 = 2*(10 % 3)
        self.resume_step = (
            self.global_step
            % (num_update_steps_per_epoch)
            * args.gradient_accumulation_steps
        )

    def train(self):
        # * 7. Resume training state
        self.__resume_state(args)
        total_batch_size = (
            args.train_batch_size
            * self.accelerator.num_processes
            * args.gradient_accumulation_steps
        )
        logger.info("****************Start Training******************")
        logger.info(f"Total training data: {len(self.train_dataloader)}")
        logger.info(f"Total update steps: {args.max_train_steps}")
        logger.info(f"Total Epochs: {args.max_train_epochs}")
        logger.info(f"Total Batch size: {total_batch_size}")
        logger.info(f"Resume from epoch={self.start_epoch}, step={self.resume_step}")
        logger.info("**********************************************")
        self.progress_bar = tqdm(
            range(self.resume_step, args.max_train_steps),
            disable=not self.accelerator.is_main_process,
            desc="Step",
        )
        for epoch in range(self.start_epoch, args.max_train_epochs):
            train_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                # * Skip steps until we reach the resumed step
                if (
                    args.resume_from_checkpoint
                    and epoch == self.first_epoch
                    and step < self.resume_step
                ):
                    if step % args.gradient_accumulation_steps == 0:
                        self.progress_bar.update(1)
                    continue
                with self.accelerator.accumulate(self.unet):
                    loss = self.__train_one_step(batch)
                # gather loss across processes for logging
                avg_loss = self.accelerator.gather(
                    loss.repeat(args.train_batch_size)
                ).mean()
                train_loss += avg_loss.item()
                # * 7. backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(
                        self.unet.parameters(), args.max_grad_norm
                    )
                # * 8. update
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
            if self.accelerator.sync_gradients:
                self.progress_bar.update(1)
                self.global_step += 1
                self.accelerator.log({"train_loss": train_loss}, step=self.global_step)
                train_loss = 0.0

                if self.global_step % args.checkpointing_steps == 0:
                    if self.accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{self.global_step}"
                        )
                        self.accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "loss": loss.detach().item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
            }
            self.progress_bar.set_postfix(**logs)
            if self.global_step >= args.max_train_steps:
                break

    def __train_one_step(self, batch):
        """
        __train_one_step: one diffusion backward step

        Args:
            - batch (_type_):
                  a batch of data, contains: #TODO:

        Returns:
            - torch.Tensor:
                  mse loss between sampled real noise and pred noise
        """
        # * 1. encode image
        latent_vector = (
            self.autoencoder.encode(batch["pixel_values"])
            .to(self.weight_dtype)
            .latent_dist.sample()
        )
        noise = torch.randn(latent_vector)
        # * 2. Sample a random timestep for each image
        timesteps = torch.randint(
            self.model.noise_scheduler.timesteps, (batch.shape[0],)
        )
        timesteps = timesteps.long()
        # * 3. add noise to latent vector
        x_t = self.model.noise_scheduler.add_noise(latent_vector, timesteps, noise)
        # * 4. get text encoding latent
        text_condition = self.text_encoder(batch["input_ids"])[0]
        # 90 % of the time we use the true text encoding, 10 % of the time we use an epmty string
        if np.random.random() < 0.1:
            text_condition = None
        # * 5. predict noise
        pred_noise = self.model.unet(x_t, timesteps, text_condition)
        return F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")

    def save_model(
        self,
    ):
        pass


if __name__ == "__main__":
    args = load_config()
    model = build_models(args.model)
    train_dataset = get_dataset(args.dataset)
    eval_dataset = get_dataset(args.dataset, split="validation")
    trainer = StableDiffusionTrainer(
        model,
        args,
        train_dataset,
        eval_dataset,
    )
    trainer.train()
