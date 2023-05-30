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
from utils.prepare_dataset import collate_fn, detransform, get_dataset, to_img
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKL
from torch.distributed.elastic.multiprocessing.errors import record

# build environment
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
        cfg,
        train_dataset,
        eval_dataset,
        collate_fn,
    ):
        # check params
        assert train_dataset is not None, "must specify an training dataset"
        assert (
            eval_dataset is not None and cfg.train.log_interval > 0
        ), "if passed log_interval > 0, you must specify an evaluation dataset"
        self.model: LatentDiffusion = model
        # test autoencoder
        self.model.autoencoder = AutoencoderKL.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            subfolder="vae",
            cache_dir="data/pretrained",
        )
        self.model.autoencoder.requires_grad_(False)
        self.cfg = cfg
        self.train_dataset: Dataset = train_dataset
        self.eval_dataset: Dataset = eval_dataset
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
                    "notes": "train unet only",
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
            self.model.unet.parameters(),
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
        elif ckpt_cfg.resume_from_checkpoint == "latest":
            # None, Get the most recent checkpoint or start from scratch
            dirs = os.listdir(ckpt_cfg.output_dir)
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
            self.accelerator.load_state(os.path.join(ckpt_cfg.output_dir, ckpt_path))
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
                with self.accelerator.accumulate(self.model.unet):
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
                            self.model.unet.parameters(), cfg.optim.max_grad_norm
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
                        save_path = os.path.join(
                            cfg.checkpoint.output_dir,
                            f"checkpoint-{self.global_step}",
                        )
                        self.accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

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
                    and self.global_step % cfg.train.log_interval == 0
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
                    self.model.train()  # back to train mode
            # save ckpt for each epoch
            if self.checkpointing_steps == "epoch":
                output_dir = f"epoch_{epoch}"
                if cfg.checkpoint.output_dir is not None:
                    output_dir = os.path.join(cfg.checkpoint.output_dir, output_dir)
                logger.info(f"Saved state to {output_dir}")
                self.accelerator.save_state(output_dir)

        # end training
        self.accelerator.wait_for_everyone()
        if self.cfg.log.with_tracking:
            self.accelerator.end_training()

        if self.accelerator.is_main_process:
            output_dir = os.path.join(
                self.cfg.checkpoint.output_dir, f"checkpoint-{self.global_step}"
            )
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
        ).latent_dist.sample()
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
        pred_noise = self.model.pred_noise(
            x_t, timesteps, text_condition, guidance_scale=self.cfg.train.guidance_scale
        )
        return F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")

    def sample(
        self,
        image_size=64,
        prompt: str = "",
        guidance_scale: float = 7.5,
        scale_factor=1.0,
        save_dir: str = "output",
    ):
        "Sample an image given prompt"
        # random noise
        # to get the right latent size
        img_util = torch.randn(size=(1, 3, image_size, image_size)).to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        noise = self.model.autoencoder.encode(img_util).latent_dist.sample()
        noise = torch.rand_like(noise).to(
            self.accelerator.device, dtype=self.weight_dtype
        )
        # tokenize prompt
        tokenized_prompt = self.model.text_encoder.tokenize([prompt]).input_ids.to(
            self.accelerator.device
        )
        context_emb = self.model.text_encoder.encode_text(tokenized_prompt)[0].to(
            self.weight_dtype
        )
        x_0 = self.model.sample(
            noised_sample=noise,
            context_emb=context_emb,
            guidance_scale=guidance_scale,
            scale_factor=scale_factor,
        )
        sample = self.model.autoencoder.decode(x_0)
        sample = detransform(sample)
        to_img(sample, output_path=save_dir, name="unet_sample")


@record
def main():
    args, cfg = load_config()
    model = build_models(cfg.model, logger)
    train_dataset = get_dataset(
        cfg.dataset,
        split="train",
        tokenizer=model.text_encoder.tokenizer,
        logger=logger,
    )
    eval_dataset = get_dataset(
        cfg.dataset,
        split="validation",
        tokenizer=model.text_encoder.tokenizer,
        logger=logger,
    )
    trainer = StableDiffusionTrainer(
        model,
        args,
        cfg,
        train_dataset,
        eval_dataset,
        collate_fn=collate_fn,
    )
    trainer.train()
    trainer.sample(prompt="a cat sat on the mat")


if __name__ == "__main__":
    main()


# to run without debug:
# accelerate launch --config_file stable_diffusion/config/accelerate_config/deepspeed.yaml --main_process_port 29511 train_unet.py --use-deepspeed
