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
import shutil
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
from stable_diffusion.models.autoencoder import AutoEncoderKL
from stable_diffusion.models.latent_diffusion import LatentDiffusion
from stable_diffusion.modules.distributions import GaussianDistribution
from utils.model_utils import build_models
from utils.parse_args import load_config
from utils.prepare_dataset import (
    collate_fn,
    detransform,
    get_dataset,
    get_transform,
    sample_test_image,
    to_img,
)
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.distributed.elastic.multiprocessing.errors import record
from PIL import Image
from transformers import CLIPTokenizer

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


class AutoencoderKLTrainer:
    def __init__(
        self,
        model: AutoEncoderKL,
        args,
        cfg,
        train_dataset,
        eval_dataset,
        test_images,
        collate_fn,
    ):
        # check params
        assert train_dataset is not None, "must specify an training dataset"
        assert (
            eval_dataset is not None and cfg.train.log_interval > 0
        ), "if passed log_interval > 0, you must specify an evaluation dataset"
        self.model: AutoEncoderKL = model
        # make sure model is in train mode
        self.model.requires_grad_(True)
        self.cfg = cfg
        self.train_dataset: Dataset = train_dataset
        self.eval_dataset: Dataset = eval_dataset
        self.test_images = test_images
        self.last_ckpt = None
        # * 1. init accelerator
        accelerator_log_kwcfg = {}
        if cfg.log.with_tracking:
            try:
                accelerator_log_kwcfg["log_with"] = cfg.log.report_to
            except AttributeError:
                print("need to specify report_to when passing in with_tracking=True")
            accelerator_log_kwcfg["logging_dir"] = cfg.log.logging_dir

        accelerator_project_config = ProjectConfiguration()
        # check deepspeed
        if cfg.train.use_deepspeed is True:
            try:
                import deepspeed
            except ImportError as e:
                raise ImportError(
                    'You passed use_deepspeed=True, please install deepspeed by running `pip install deepspeed`, also deepspeed requies a matched cuda version, so you may need to run `conda install -c "nvidia/label/cuda-11.5.0" cuda-toolkit`, see https://anaconda.org/nvidia/cuda-toolkit for more options'
                ) from e

        self.accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
            **accelerator_log_kwcfg,
            project_config=accelerator_project_config,
            deepspeed_plugin=DeepSpeedPlugin(
                zero_stage=2,
                gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
                gradient_clipping=cfg.optim.max_grad_norm,
                offload_optimizer_device="cpu",
                offload_param_device="cpu",
            )
            if cfg.train.use_deepspeed is True
            else None,
        )
        if cfg.log.with_tracking:
            if cfg.log.report_to != "wandb":
                raise NotImplementedError(
                    "Currently only support wandb, init trakcer for your platforms"
                )
            if self.accelerator.is_main_process:
                try:
                    import wandb
                except ImportError as e:
                    raise ImportError(
                        "You passed with_tracking and report_to `wandb`, please install wandb by running `pip install wandb`"
                    ) from e

                wandb_kwargs = {
                    "name": f"run_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}",
                    "notes": "train autoencoder",
                    "group": "train autoencoder",
                    "tags": ["stable diffusion", "pytorch"],
                    "entity": "liwenbo2099",
                    "resume": cfg.log.resume,
                    # "id": ,
                    "save_code": True,
                    "allow_val_change": True,
                }

                self.accelerator.init_trackers(
                    "stable_diffusion_pytorch",
                    args,
                    init_kwargs={"wandb": wandb_kwargs},
                )
                wandb.config.update(
                    args, allow_val_change=True  # wandb_kwargs["allow_val_change"]
                )

        logger.info(self.accelerator.state, main_process_only=False)
        # * 2. set seed
        if cfg.train.seed is not None:
            set_seed(cfg.train.seed)
        # * 4. build optimizer and lr_scheduler
        self.optimizer = self.__build_optimizer(cfg.optim)
        self.lr_scheduler = self.__build_lr_scheduler(cfg.optim)
        # * 5. get dataset
        self.train_dataloader = self.get_dataloader(
            train=True,
            dataset=self.train_dataset,
            collate_fn=collate_fn,
            batch_size=cfg.train.train_batch_size,
            num_workers=cfg.dataset.dataloader_num_workers
            or self.accelerator.num_processes,
        )
        self.eval_dataloader = self.get_dataloader(
            train=False,
            dataset=self.eval_dataset,
            collate_fn=collate_fn,
            batch_size=cfg.train.eval_batch_size,
            num_workers=cfg.dataset.dataloader_num_workers
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
        self.model: AutoEncoderKL = self.model
        self.train_dataloader: DataLoader = self.train_dataloader
        self.eval_dataloader: DataLoader = self.eval_dataloader
        # * 6. Move text_encoder and autoencoder to gpu and cast to weight_dtype
        self.weight_dtype = torch.float32
        if self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        elif self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        self.model.to(self.accelerator.device, dtype=self.weight_dtype)

    def get_dataloader(
        self, train, dataset, collate_fn, batch_size, num_workers
    ) -> DataLoader:
        return DataLoader(
            dataset,
            shuffle=train,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    def __build_optimizer(self, optim_cfg):
        # Initialize the optimizer
        if optim_cfg.use_8bit_adam:
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
            self.model.parameters(),
            lr=optim_cfg.learning_rate,
            weight_decay=optim_cfg.adam_weight_decay,
        )

    def __build_lr_scheduler(self, optim_cfg):
        lr_scheduler = None
        if (
            self.accelerator.state.deepspeed_plugin is None
            or "scheduler"
            not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            lr_scheduler = get_scheduler(
                optim_cfg.scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=optim_cfg.lr_warmup_steps
                * self.cfg.train.gradient_accumulation_steps,
                num_training_steps=self.cfg.train.max_train_steps
                * self.cfg.train.gradient_accumulation_steps,
            )
        else:
            lr_scheduler = DummyScheduler(
                self.optimizer,
                total_num_steps=self.cfg.train.max_train_steps,
                warmup_num_steps=optim_cfg.lr_warmup_steps,
            )
        return lr_scheduler

    def __resume_from_ckpt(self, ckpt_cfg):
        "resume from checkpoint or start a new train"
        ckpt_path = None
        if ckpt_cfg.resume_from_checkpoint:
            ckpt_path = os.path.basename(ckpt_cfg.resume_from_checkpoint)
            if ckpt_cfg.resume_from_checkpoint == "latest":
                # None, Get the most recent checkpoint or start from scratch
                dirs = os.listdir(ckpt_cfg.ckpt_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(
                    dirs, key=lambda x: int(x.split("-")[1])
                )  # checkpoint-100 => 100
                ckpt_path = dirs[-1] if len(dirs) > 0 else None
        if ckpt_path is None:
            self.accelerator.print(
                f"Checkpoint '{ckpt_cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            ckpt_cfg.resume_from_checkpoint = None
        else:
            self.accelerator.print(f"Resuming from checkpoint {ckpt_path}")
            self.accelerator.load_state(os.path.join(ckpt_cfg.ckpt_dir, ckpt_path))
        self.ckpt_path = ckpt_path

    def __resume_train_state(self, train_cfg, ckpt_path):
        """
        resume train steps and epochs
        """
        # * Calculate train steps
        # total batch num / accumulate steps => actual update steps per epoch
        num_update_steps_per_epoch = math.ceil(
            len(self.train_dataloader) / train_cfg.gradient_accumulation_steps
        )
        if train_cfg.max_train_steps is None:
            train_cfg.max_train_steps = (
                train_cfg.max_train_epochs * num_update_steps_per_epoch
            )
        else:
            # override max_train_epochs
            train_cfg.max_train_epochs = math.ceil(
                train_cfg.max_train_steps / num_update_steps_per_epoch
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
            * train_cfg.gradient_accumulation_steps
        )

    def train(self):
        cfg = self.cfg
        # * 7. Resume training state and ckpt
        self.__resume_from_ckpt(cfg.checkpoint)
        self.__resume_train_state(cfg.train, self.ckpt_path)

        total_batch_size = (
            cfg.train.train_batch_size
            * self.accelerator.num_processes
            * cfg.train.gradient_accumulation_steps
        )
        self.checkpointing_steps = cfg.checkpoint.checkpointing_steps
        if self.checkpointing_steps is not None and self.checkpointing_steps.isdigit():
            self.checkpointing_steps = int(self.checkpointing_steps)
        logger.info("****************Start Training******************")
        logger.info(f"Total training data: {len(self.train_dataloader.dataset)}")
        if hasattr(self, "eval_dataloader") and self.eval_dataloader is not None:
            logger.info(f"Total eval data: {len(self.eval_dataloader.dataset)}")
        logger.info(f"Total update steps: {cfg.train.max_train_steps}")
        logger.info(f"Total Epochs: {cfg.train.max_train_epochs}")
        logger.info(f"Total Batch size: {total_batch_size}")
        logger.info(f"Resume from epoch={self.start_epoch}, step={self.resume_step}")
        logger.info("**********************************************")
        self.progress_bar = tqdm(
            range(self.global_step, cfg.train.max_train_steps),
            total=cfg.train.max_train_steps,
            disable=not self.accelerator.is_main_process,
            initial=self.global_step,  # @note: huggingface seemed to missed this, should first update to global step
            desc="Step",
        )
        self.model.train()
        for epoch in range(self.start_epoch, cfg.train.max_train_epochs):
            train_loss = 0
            for step, batch in enumerate(self.train_dataloader):
                # * Skip steps until we reach the resumed step
                if (
                    self.ckpt_path is not None
                    and epoch == self.start_epoch
                    and step < self.resume_step
                ):
                    if step % cfg.train.gradient_accumulation_steps == 0:
                        # @note: huggingface seemed to missed this, global step should also be updated
                        self.global_step += 1
                        self.progress_bar.update(1)
                    continue
                with self.accelerator.accumulate(self.model):
                    loss = self.__one_step(batch)
                    # gather loss across processes for logging
                    avg_loss = self.accelerator.gather(
                        loss.repeat(cfg.train.train_batch_size)
                    ).mean()
                    train_loss += avg_loss.item()
                    # * 7. backward
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), cfg.optim.max_grad_norm
                        )
                    # * 8. update
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if self.accelerator.sync_gradients:
                    self.progress_bar.update(1)
                    self.global_step += 1
                    if self.cfg.log.with_tracking:
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
                        ckpt_path = os.path.join(
                            cfg.checkpoint.ckpt_dir, f"checkpoint-{self.global_step}"
                        )
                        if (
                            self.cfg.checkpoint.keep_last_only
                            and self.accelerator.is_main_process
                        ):
                            # del last save path
                            if self.last_ckpt is not None:
                                shutil.rmtree(self.last_ckpt)
                            self.last_ckpt = ckpt_path
                            logger.info(f"self.savepath={ckpt_path}")
                        # wait main process handle dir del and create
                        self.accelerator.wait_for_everyone()
                        # @note: when using deepspeed, we can't use is_main_process, or it will get stucked
                        self.accelerator.save_state(ckpt_path)
                        logger.info(f"Saved state to {ckpt_path}")

                logs = {
                    "loss": loss.detach().item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                self.progress_bar.set_postfix(**logs)
                if self.global_step >= cfg.train.max_train_steps:
                    break
                # =======================Evaluation==========================
                if (
                    self.global_step > 0
                    and cfg.train.log_interval > 0
                    and (self.global_step + 1) % cfg.train.log_interval == 0
                ):
                    logger.info(
                        f"Evaluate on eval dataset [len: {len(self.eval_dataset)}]"
                    )
                    self.model.eval()
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
                                loss.repeat(cfg.train.eval_batch_size)
                            )
                        )
                    losses = torch.cat(losses)
                    eval_loss = torch.mean(losses)
                    logger.info(
                        f"global step {self.global_step}: eval_loss: {eval_loss}"
                    )
                    if cfg.log.with_tracking:
                        self.accelerator.log(
                            {
                                "eval_loss": eval_loss,
                            },
                            step=self.global_step,
                        )
                    # log image
                    if cfg.log.log_image:
                        self.log_image()
                    self.model.train()  # back to train mode
            # save ckpt for each epoch
            if self.checkpointing_steps == "epoch":
                ckpt_path = f"epoch_{epoch}"
                if cfg.checkpoint.ckpt_dir is not None:
                    ckpt_path = os.path.join(cfg.ckpt_dir, ckpt_path, "autoencoder")
                if (
                    self.cfg.checkpoint.keep_last_only
                    and self.accelerator.is_main_process
                ):
                    # del last save path
                    if self.last_ckpt is not None:
                        shutil.rmtree(self.ckpt_path)
                    self.last_ckpt = ckpt_path
                # @note: when using deepspeed, we can't use is_main_process, or it will get stucked
                self.accelerator.save_state(ckpt_path)
                logger.info(f"Saved state to {ckpt_path}")

        # end training
        self.accelerator.wait_for_everyone()
        if self.cfg.log.with_tracking:
            self.accelerator.end_training()

    def __one_step(self, batch: dict):
        """
        __train_one_step: one diffusion backward step

        Args:
            - batch (dict):
                  a batch of data, contains: pixel_values and input_ids

        Returns:
            - torch.Tensor:
                  recon loss + kl loss
        """
        # * 1. encode image
        img = batch["pixel_values"].to(self.weight_dtype)
        dist: GaussianDistribution = self.model.encode(img).latent_dist
        latent_vector = dist.sample()
        recon_image = self.model.decode(latent_vector)
        recon_loss = F.mse_loss(img.float(), recon_image.float(), reduction="mean")
        kl_loss = dist.kl()[0].to(dtype=torch.float32)  # recon loss is float32
        # @ recon loss is float32, pass float16 loss will raise bias correction error in deepspeed cpu adam
        return recon_loss + self.cfg.model.autoencoder.kl_weight * kl_loss

    def recon(self, image):
        image = image.unsqueeze(0).to(
            self.accelerator.device, dtype=self.weight_dtype
        )  # [1, ...]
        latent_vector = self.model.encode(image).latent_dist.sample()
        recon_latent = self.model.decode(latent_vector)
        recon_digit = detransform(recon_latent)
        return to_img(recon_digit, output_path="output", name="autoencoder")

    def log_image(self):
        recons = [self.recon(img) for img in self.test_images]
        if self.cfg.log.with_tracking:
            import wandb

            self.accelerator.log(
                {
                    "original_imgs": [wandb.Image(img) for img in self.test_images],
                    "recon_imgs": [wandb.Image(recon) for recon in recons],
                },
                step=self.global_step,
            )


@record
def main():
    args, cfg = load_config()
    model = AutoEncoderKL(cfg.model.autoencoder)
    tokenizer = CLIPTokenizer.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="tokenizer",
        use_fast=False,
        cache_dir="data/pretrained",
    )
    train_dataset = get_dataset(
        cfg.dataset,
        split="train",
        tokenizer=tokenizer,
        logger=logger,
    )
    eval_dataset = get_dataset(
        cfg.dataset,
        split="validation",
        tokenizer=tokenizer,
        logger=logger,
    )
    test_images = sample_test_image(
        cfg.dataset,
        split="test",
        tokenizer=tokenizer,
        logger=logger,
        num=10,
    )
    trainer = AutoencoderKLTrainer(
        model,
        args,
        cfg,
        train_dataset,
        eval_dataset,
        test_images=test_images,
        collate_fn=collate_fn,
    )
    trainer.train()


if __name__ == "__main__":
    main()


# to run without debug:
# accelerate launch --config_file stable_diffusion/config/accelerate_config/deepspeed.yaml --main_process_port 29511 train_autoencoder.py --use-deepspeed --with-tracking --log-image --max-train-steps 10000 --max-train-samples 700 --max-val-samples 50 --max-test-samples 50 --resume-from-checkpoint latest --learning-rate 1e-3
