#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2023/05/12 19:24:02
@Author  :   Wenbo Li
@Desc    :   Main Class for training stable diffusion model
"""


import math
import os
import time

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    set_seed,
    DummyOptim,
    ProjectConfiguration,
    DummyScheduler,
    DeepSpeedPlugin,
)
from transformers import get_scheduler
import logging
from stable_diffusion.models.latent_diffusion import LatentDiffusion
from utils.model_utils import build_models
from utils.parse_args import load_config
from utils.prepare_dataset import collate_fn, get_dataset
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# build environment
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"
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
        assert train_dataset is not None, "must specify an training dataset"
        assert (
            eval_dataset is not None and args.train.log_interval > 0
        ), "if passed log_interval > 0, you must specify an evaluation dataset"
        self.model: LatentDiffusion = model
        self.args = args
        self.train_dataset: Dataset = train_dataset
        self.eval_dataset: Dataset = eval_dataset
        # * 1. init accelerator
        accelerator_log_kwargs = {}
        if args.log.with_tracking:
            try:
                accelerator_log_kwargs["log_with"] = args.log.report_to
            except AttributeError:
                print("need to specify report_to when passing in with_tracking=True")
            accelerator_log_kwargs["logging_dir"] = args.log.logging_dir

        accelerator_project_config = ProjectConfiguration()
        # check deepspeed
        if args.train.use_deepspeed is True:
            try:
                import deepspeed
            except ImportError as e:
                raise ImportError(
                    'You passed use_deepspeed=True, please install deepspeed by running `pip install deepspeed`, also deepspeed requies a matched cuda version, so you may need to run `conda install -c "nvidia/label/cuda-11.5.0" cuda-toolkit`, see https://anaconda.org/nvidia/cuda-toolkit for more options'
                ) from e

        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.train.gradient_accumulation_steps,
            **accelerator_log_kwargs,
            project_config=accelerator_project_config,
            deepspeed_plugin=DeepSpeedPlugin(
                zero_stage=2,
                gradient_accumulation_steps=args.train.gradient_accumulation_steps,
                gradient_clipping=args.optim.max_grad_norm,
                offload_optimizer_device="cpu",
                offload_param_device="cpu",
            )
            if args.train.use_deepspeed is True
            else None,
        )
        if args.log.with_tracking:
            if args.log.report_to != "wandb":
                raise NotImplementedError(
                    "Currently only support wandb, init trakcer for your platforms"
                )
            try:
                import wandb
            except ImportError as e:
                raise ImportError(
                    "You passed with_tracking and report_to `wandb`, please install wandb by running `pip install wandb`"
                ) from e

            self.accelerator.init_trackers(
                "stable_diffusion_pytorch",
                args,
                init_kwargs={
                    "wandb": {
                        "name": f"run_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}",
                        "tags": ["stable diffusion", "pytorch"],
                        "entity": "liwenbo2099",
                        # "resume": True,
                        # "id": "okoxasi7",
                        "save_code": True,
                    }
                },
            )

        logger.info(self.accelerator.state, main_process_only=False)
        # * 2. set seed
        if args.train.seed is not None:
            set_seed(args.train.seed)
        # * 4. build optimizer and lr_scheduler
        self.optimizer = self.__build_optimizer(args.optim)
        self.lr_scheduler = self.__build_lr_scheduler(args.optim)
        # * 5. get dataset
        self.train_dataloader = self.get_dataloader(
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            batch_size=args.train.train_batch_size,
            num_workers=args.dataset.dataloader_num_workers
            or self.accelerator.num_processes,
        )
        self.eval_dataloader = self.get_dataloader(
            dataset=self.eval_dataset,
            collate_fn=collate_fn,
            batch_size=args.train.eval_batch_size,
            num_workers=args.dataset.dataloader_num_workers
            or self.accelerator.num_processes,
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
        if self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        elif self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        self.model.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.model.autoencoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.model.unet.to(self.accelerator.device, dtype=self.weight_dtype)

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

        # * code change to fit deepspeed
        optimizer_cls = (
            optimizer_cls
            if self.accelerator.state.deepspeed_plugin is None
            or "optimizer"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )

        return optimizer_cls(
            # only train unet
            self.model.unet.parameters(),
            lr=optim_args.learning_rate,
            weight_decay=optim_args.adam_weight_decay,
        )

    def __build_lr_scheduler(self, optim_args):
        lr_scheduler = None
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            lr_scheduler = get_scheduler(
                optim_args.scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=optim_args.lr_warmup_steps
                * self.args.train.gradient_accumulation_steps,
                num_training_steps=self.args.train.max_train_steps
                * self.args.train.gradient_accumulation_steps,
            )
        else:
            lr_scheduler = DummyScheduler(
                self.optimizer,
                total_num_steps=self.args.train.max_train_steps,
                warmup_num_steps=optim_args.lr_warmup_steps,
            )
        return lr_scheduler

    def __resume_from_ckpt(self, ckpt_args):
        "resume from checkpoint or start a new train"
        if ckpt_args.resume_from_checkpoint:
            ckpt_path = os.path.basename(ckpt_args.resume_from_checkpoint)
        else:
            # None, Get the most recent checkpoint or start from scratch
            dirs = os.listdir(ckpt_args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(
                dirs, key=lambda x: int(x.split("-")[1])
            )  # checkpoint-100 => 100
            ckpt_path = dirs[-1] if len(dirs) > 0 else None
        if ckpt_path is None:
            self.accelerator.print(
                f"Checkpoint '{ckpt_args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            ckpt_args.resume_from_checkpoint = None
        else:
            self.accelerator.print(f"Resuming from checkpoint {ckpt_path}")
            self.accelerator.load_state(os.path.join(ckpt_args.output_dir, ckpt_path))
        self.ckpt_path = ckpt_path

    def __resume_train_state(self, train_args, ckpt_path):
        """
        resume train steps and epochs
        """
        # * Calculate train steps
        # total batch num / accumulate steps => actual update steps per epoch
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / train_args.gradient_accumulation_steps
        )
        if train_args.max_train_steps is None:
            train_args.max_train_steps = (
                train_args.max_train_epochs * num_update_steps_per_epoch
            )
        else:
            # override max_train_epochs
            train_args.max_train_epochs = math.ceil(
                train_args.max_train_steps / num_update_steps_per_epoch
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
            * train_args.gradient_accumulation_steps
        )

    def train(self):
        # * 7. Resume training state and ckpt
        self.__resume_from_ckpt(args.checkpoint)
        self.__resume_train_state(args.train, self.ckpt_path)
        # self.model.autoencoder.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        total_batch_size = (
            args.train.train_batch_size
            * self.accelerator.num_processes
            * args.train.gradient_accumulation_steps
        )
        self.checkpointing_steps = args.checkpoint.checkpointing_steps
        if self.checkpointing_steps is not None and self.checkpointing_steps.isdigit():
            self.checkpointing_steps = int(self.checkpointing_steps)
        logger.info("****************Start Training******************")
        logger.info(f"Total training data: {len(self.train_dataloader.dataset)}")
        if hasattr(self, "eval_dataloader") and self.eval_dataloader is not None:
            logger.info(f"Total eval data: {len(self.eval_dataloader.dataset)}")
        logger.info(f"Total update steps: {args.train.max_train_steps}")
        logger.info(f"Total Epochs: {args.train.max_train_epochs}")
        logger.info(f"Total Batch size: {total_batch_size}")
        logger.info(f"Resume from epoch={self.start_epoch}, step={self.resume_step}")
        logger.info("**********************************************")
        self.progress_bar = tqdm(
            range(self.global_step, args.train.max_train_steps),
            total=args.train.max_train_steps,
            disable=not self.accelerator.is_main_process,
            initial=self.global_step,  # @note: huggingface seemed to missed this, should first update to global step
            desc="Step",
        )
        self.model.train()
        for epoch in range(self.start_epoch, args.train.max_train_epochs):
            train_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                # * Skip steps until we reach the resumed step
                if (
                    self.ckpt_path is not None
                    and epoch == self.start_epoch
                    and step < self.resume_step
                ):
                    if step % args.train.gradient_accumulation_steps == 0:
                        # @note: huggingface seemed to missed this, global step should also be updated
                        self.global_step += 1
                        self.progress_bar.update(1)
                    continue
                with self.accelerator.accumulate(self.model.unet):
                    loss = self.__one_step(batch)
                    # gather loss across processes for logging
                    avg_loss = self.accelerator.gather(
                        loss.repeat(args.train.train_batch_size)
                    ).mean()
                    train_loss += avg_loss.item()
                    # * 7. backward
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.unet.parameters(), args.optim.max_grad_norm
                        )
                    # * 8. update
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.progress_bar.update(1)
                    self.global_step += 1
                    if self.args.log.with_tracking:
                        self.accelerator.log(
                            {
                                "train_loss": train_loss,
                                "lr": self.lr_scheduler.get_last_lr()[0],
                            },
                            step=self.global_step,
                        )
                    train_loss = 0.0
                    if (
                        isinstance(self.checkpointing_steps, int)
                        and self.global_step % self.checkpointing_steps == 0
                    ):
                        save_path = os.path.join(
                            args.checkpoint.output_dir,
                            f"checkpoint-{self.global_step}",
                        )
                        self.accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                logs = {
                    "loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                self.progress_bar.set_postfix(**logs)
                if self.global_step >= args.train.max_train_steps:
                    break
                # =======================Evaluation==========================
                if (
                    self.global_step > 0
                    and args.train.log_interval > 0
                    and self.global_step % args.train.log_interval == 0
                ):
                    logger.info(
                        f"Evaluate on eval dataset [len: {len(self.eval_dataset)}]"
                    )
                    model.eval()
                    losses = []
                    eval_bar = tqdm(
                        self.eval_dataloader,
                        disable=not self.accelerator.is_main_process,
                    )
                    for step, batch in enumerate(eval_bar):
                        with torch.no_grad():
                            loss = self.__one_step(batch)
                        losses.append(
                            self.accelerator.gather_for_metrics(
                                loss.repeat(args.train.eval_batch_size)
                            )
                        )
                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses)
                    logger.info(
                        f"global step {self.global_step}: eval_loss: {eval_loss}"
                    )
                    if args.log.with_tracking:
                        self.accelerator.log(
                            {
                                "eval_loss": eval_loss,
                                "epoch": epoch,
                            },
                            step=self.global_step,
                        )
                    self.model.train()  # back to train mode
            # save ckpt for each epoch
            if self.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if args.checkpoint.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                logger.info(f"Saved state to {output_dir}")
                self.accelerator.save_state(output_dir)

        # end training
        self.accelerator.wait_for_everyone()
        if self.args.log.with_tracking:
            self.accelerator.end_training()

        if self.accelerator.is_main_process:
            output_dir = os.path.join(self.args.output_dir, self.global_step)
            self.accelerator.save_state()

    def __one_step(self, batch: dict):
        """
        __train_one_step: one diffusion backward step

        Args:
            - batch (dict):
                  a batch of data, contains: pixel_values and input_ids

        Returns:
            - torch.Tensor:
                  mse loss between sampled real noise and pred noise
        """
        # * 1. encode image
        latent_vector = self.model.autoencoder.encode(
            batch["pixel_values"].to(self.weight_dtype)
        ).sample()
        noise = torch.randn(latent_vector.shape).to(self.accelerator.device)
        # * 2. Sample a random timestep for each image
        timesteps = torch.randint(
            self.model.noise_scheduler.noise_steps, (batch["pixel_values"].shape[0],)
        ).to(self.accelerator.device, dtype=torch.long)
        # timesteps = timesteps.long()
        # * 3. add noise to latent vector
        x_t = self.model.noise_scheduler.add_noise(
            original_samples=latent_vector, timesteps=timesteps, noise=noise
        ).to(dtype=self.weight_dtype)
        # * 4. get text encoding latent
        tokenized_text = batch["input_ids"]
        # 90 % of the time we use the true text encoding, 10 % of the time we use an empty string
        if np.random.random() < 0.1:
            tokenized_text = self.model.text_encoder.tokenize(
                [""] * len(tokenized_text)
            ).input_ids.to(self.accelerator.device)
        text_condition = self.model.text_encoder.encode_text(
            tokenized_text  # adding `.to(self.weight_dtype)` causes error...
        )[0].to(self.weight_dtype)
        # * 5. predict noise
        pred_noise = self.model.pred_noise(x_t, timesteps, text_condition)
        return F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")


if __name__ == "__main__":
    args = load_config()
    model = build_models(args.model, logger)
    train_dataset = get_dataset(
        args.dataset,
        split="train",
        tokenizer=model.text_encoder.tokenizer,
        logger=logger,
    )
    eval_dataset = get_dataset(
        args.dataset,
        split="validation",
        tokenizer=model.text_encoder.tokenizer,
        logger=logger,
    )
    trainer = StableDiffusionTrainer(
        model,
        args,
        train_dataset,
        eval_dataset,
        collate_fn=collate_fn,
    )
    trainer.train()
