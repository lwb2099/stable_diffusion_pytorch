#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   train.py
@Time    :   2023/05/12 19:24:02
@Author  :   Wenbo Li
@Desc    :   Main Class for training stable diffusion model
'''

import argparse
import math
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import logging
from diffusers.optimization import get_scheduler
from utils.load_datasets import get_dataset
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm import tqdm
from torchvision import transforms

# build environment
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a stable diffusion training script.")
    #TODO: Add args
    # utils
    parser.add_argument(
        "--logging_dir", 
        type=str,
        default="logs",
        help="log directory"
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=None,
    )
    # loading and saving
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="openai/clip-vit-base-patch16"
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        default="openai/clip-vit-base-patch16",
        help="CLIP model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Norod78/simpsons-blip-captions",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="dir to load checkpoints from"
    )
  
    parser.add_argument(
        "--resolution",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--center_crop",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--random_flip",
        type=bool,
        default=True,
    )
    # training
    parser.add_argument(
        "--seed",
        type=int,
        default=0
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="linear",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="total train steps, if provided, overrides max_train_epochs"
    )
    parser.add_argument(
        "--max_train_epochs",
        type=int,
        default=100,
        help="max train epochs, orverides by max_training_steps"
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.1
    )
    # distributed training
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        default=False,
    )
    

    args = parser.parse_args()
    
    return args

class StableDiffusionPipeline():

    def __init__(self):
        pass
    
    def __build_model(self,tokenizer, text_encoder):
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer, use_fast=False)
        # ***************** test ******************
        from diffusers import DDPMScheduler, UNet2DConditionModel,AutoencoderKL
        self.noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")  # DDPMScheduler()
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")  # UNet2DConditonal()
        self.text_encoder = CLIPTextModel.from_pretrained(text_encoder)
        self.autoencoder = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")  # VQVAE()
        # Freeze vae and text_encoder
        self.autoencoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
    
    def __build_optimizer(self, args):
         # Initialize the optimizer
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                raise ImportError(
                    "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
                ) from e

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW

        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=args.learning_rate,
            weight_decay=args.adam_weight_decay,
        )

    def __resume_from_checkpoint(self, args):
        """
        Load checkpoints, copy from diffusers, examples/train_text_to_image, line 750 or so
        slightly modified
        """
        if args.resume_from_checkpoint:
            ckpt_path = os.path.basename(args.resume_from_checkpoint)
        else:
            # None, Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))  # checkpoint-100 => 100
            ckpt_path = dirs[-1] if len(dirs) > 0 else None
        if ckpt_path is None:
            self.accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            self.accelerator.print(f"Resuming from checkpoint {ckpt_path}")
            self.accelerator.load_state(os.path.join(args.output_dir, ckpt_path))
        #* Calculate train steps
        # total batch num / accumulate steps => actual update steps per epoch
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.max_train_epochs * num_update_steps_per_epoch
        else:
            # override max_train_epochs
            args.max_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
        # actual updated steps
        self.global_step = int(ckpt_path.split("-")[1]) if ckpt_path else 0
        self.start_epoch = self.global_step // num_update_steps_per_epoch if ckpt_path else 0
        #* change diffusers implementation: 20 % 6 = 2 = 2*(10 % 3)
        self.resume_step = self.global_step  % (num_update_steps_per_epoch) * args.gradient_accumulation_steps

    def prepare(self, args):
        #* 1. init accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            log_with=args.tracker,
            project_dir=args.logging_dir,
        )
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)
        #* 2. set seed
        if args.seed is not None:
            set_seed(args.seed)
        #* 3. import tokenizers, models 
        self.__build_model(args.tokenizer, args.text_encoder)
        #* 4. build optimizer
        self.__build_optimizer(args)
        #* 5. get dataset
        train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
        self.train_dataloader = get_dataset(args.dataset, args.data_dir, self.tokenizer, train_transforms, args.train_batch_size, self.accelerator.num_processes)
        self.lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
                num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
            )
        #* 5. Prepare everything with our `accelerator`.
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        #* 6. Move text_encode and vae to gpu and cast to weight_dtype
        self.text_encoder.to(self.accelerator.device, dtype=self.accelerator.mixed_precision)
        self.autoencoder.to(self.accelerator.device, dtype=self.accelerator.mixed_precision)
        #* 7. Resume from checkpoint 
        self.__resume_from_checkpoint(args)
        total_batch_size = args.batch_size * self.accelerator.num_processes * args.gradient_accumulation_steps
        logger.info("**********************************************")
        logger.info(f"Total training data: {len(self.train_dataloader)}")
        logger.info(f"Total update steps: {args.max_train_steps}")
        logger.info(f"Total Epochs: {args.max_train_epochs}")
        logger.info(f"Total Batch size: {total_batch_size}")
        logger.info(f"Resume from epoch={self.start_epoch}, step={self.resume_step}")
        logger.info("**********************************************")

    def fit(self, args):
        logger.info("Start Training")
        self.progress_bar = tqdm(
                        range(self.resume_step, args.total_train_update_steps),
                        disable= not self.accelerator.is_main_process,
                        desc="Step"
                    )
        for epoch in range(self.start_epoch, args.total_train_epochs):
            train_loss = 0
            for step, batch in enumerate(self.train_dataloader):
            #* Skip steps until we reach the resumed step
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
                avg_loss = self.accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item()   
                #* 7. backward
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.unet.parameters(), args.max_grad_norm)
                #* 8. update
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
                        save_path = os.path.join(args.output_dir, f"checkpoint-{self.global_step}")
                        self.accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0]}
            self.progress_bar.set_postfix(**logs)

            if self.global_step >= args.max_train_steps:
                break
    
    def __train_one_step(self, batch):
        """
        __train_one_step: one diffusion backward step

        Args:
            - batch (_type_):   
                  _description_

        Returns:
            - _type_:   
                  mse loss between sampled real noise and pred noise
        """
        #* 1. encode image
        latent_vector = self.autoencoder.encode(batch["pixel_values"]).to(self.accelerator.mixed_precision)
        noise = torch.randn(latent_vector)
        #* 2. Sample a random timestep for each image
        bsz = latent_vector.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.noise_timesteps, (bsz,), device=latent_vector.device)
        timesteps = timesteps.long()
        #* 3. add noise to latent vector
        x_t = self.noise_scheduler.add_noise(latent_vector, timesteps, noise)
        #* 4. get text encoding latent
        text_condition = self.text_encoder(batch["input_ids"])[0]
        # 90 % of the time we use the true text encoding, 10 % of the time we use an epmty string
        if np.random.random() < 0.1:
                text_condition = None
        #* 5. predict noise
        pred_noise = self.unet(x_t, timesteps, text_condition)
        return F.mse_loss(pred_noise.float(), noise.float(), reduction="mean")


if __name__ == '__main__':
    args = parse_args()
    stablediffusion = StableDiffusion()
    stablediffusion.prepare(args)
    stablediffusion.fit(args)
