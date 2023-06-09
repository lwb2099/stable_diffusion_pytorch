#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   prepare_dataset.py
@Time    :   2023/05/26 20:07:10
@Author  :   Wenbo Li
@Desc    :   code for loading and transform txt2img dataset
"""

import os
import random
from typing import Optional
import numpy as np
import torch
from torchvision import transforms
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import CLIPTokenizer

from dataclasses import dataclass, field

from stable_diffusion.dataclass import BaseDataclass
from PIL import Image


@dataclass
class DatasetConfig(BaseDataclass):
    dataset: str = field(
        default="poloclub/diffusiondb",
        metadata={"help": "name of the dataset to use."},
    )
    subset: Optional[str] = field(
        default=None,  # "2m_first_10k",
        metadata={"help": "subset of the dataset to use."},
    )
    # dataset: str = field(
    #     default="lambdalabs/pokemon-blip-captions",
    # )
    data_dir: str = field(
        default="data/dataset",
        metadata={"help": "Cache directory to store loaded dataset."},
    )
    dataloader_num_workers: int = field(
        default=4, metadata={"help": "number of workers for the dataloaders."}
    )
    resolution: int = field(default=64, metadata={"help": "resolution of the images."})
    center_crop: bool = field(
        default=True, metadata={"help": "whether to apply center cropping."}
    )
    random_flip: bool = field(
        default=False, metadata={"help": "whether to apply random flipping."}
    )
    max_train_samples: Optional[int] = field(
        default=9000, metadata={"help": "max number of training samples to load."}
    )
    max_val_samples: Optional[int] = field(
        default=500, metadata={"help": "max number of validation samples to load."}
    )
    max_test_samples: Optional[int] = field(
        default=500, metadata={"help": "max number of test samples to load."}
    )


def add_dataset_args(parser):
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="poloclub/diffusiondb",
    )
    dataset_group.add_argument(
        "--subset",
        type=str,
        default="2m_first_100k",
    )
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        default="data",
    )
    dataset_group.add_argument(
        "--resolution",
        type=int,
        default=64,
    )
    dataset_group.add_argument(
        "--center_crop",
        type=bool,
        default=True,
    )
    dataset_group.add_argument(
        "--random_flip",
        type=bool,
        default=True,
    )


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def tokenize_captions(examples, tokenizer, is_train=True):
    captions = []
    for caption in examples:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                "Caption column should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


def get_transform(resolution, random_flip, center_crop):
    return transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(resolution)
            if center_crop
            else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip()
            if random_flip
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )


def detransform(latent: torch.Tensor):
    latent = latent.squeeze().cpu().detach().numpy()
    latent = np.transpose(latent, (1, 2, 0))  # [c,h,w] -> [h,w,c]
    latent = (latent + 1) / 2
    latent = np.clip(latent, 0, 1)
    return (latent * 255).astype(np.uint8)


def to_img(digit_img, output_path: str = "", name="sample"):
    img = Image.fromarray(digit_img.astype(np.uint8))
    img.save(os.path.join(output_path, f"{name}.png"))
    return img


def get_dataset(
    args,
    split: str = "train",
    tokenizer: CLIPTokenizer = None,
    logger=None,
):
    # check params
    assert tokenizer is not None, "you need to specify a tokenizer"

    assert split in {
        "train",
        "validation",
        "test",
    }, "split should be one of train, validation, test"

    # most of the txt2img datasets are not splited into train, validation and test, manually split it
    dataset = load_dataset(
        args.dataset,
        args.subset,
        cache_dir=os.path.join(args.data_dir, args.dataset),
    )["train"]

    if args.max_train_samples is not None and split == "train":
        if args.max_train_samples < len(dataset):
            dataset = dataset.select(range(args.max_train_samples))
        elif logger is not None:
            logger.info(
                f"max_train_samples({args.max_train_samples}) is larger than the number of train samples({len(dataset)})"
            )
    if args.max_val_samples is not None and split == "validation":
        if args.max_train_samples + args.max_val_samples < len(dataset):
            dataset = dataset.select(
                range(
                    args.max_train_samples,
                    args.max_train_samples + args.max_val_samples,
                )
            )
        elif logger is not None:
            logger.info(
                f"max_val_samples({args.max_val_samples}) is larger than the number of val samples({len(dataset)})"
            )
    if args.max_test_samples is not None and split == "test":
        if args.max_train_samples + args.max_val_samples + args.max_test_samples < len(
            dataset
        ):
            dataset = dataset.select(
                range(
                    args.max_train_samples + args.max_val_samples,
                    args.max_train_samples
                    + args.max_val_samples
                    + args.max_test_samples,
                )
            )
        elif logger is not None:
            logger.info(
                f"max_test_samples({args.max_test_samples}) is larger than the number of test samples({len(dataset)})"
            )

    image_column = [col for col in ["image", "img"] if col in dataset.column_names][0]
    caption_colum = [
        col for col in ["text", "caption", "prompt"] if col in dataset.column_names
    ][0]

    transform = get_transform(args.resolution, args.random_flip, args.center_crop)

    def preprocess_train(examples):
        """tokenize captions and convert images to pixel values"""
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [transform(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples[caption_colum], tokenizer)
        return examples

    if logger is not None:
        logger.info(
            f"Loaded {len(dataset)} {split} samples from dataset:{args.dataset}"
        )

    return dataset.with_transform(preprocess_train)


def sample_test_image(args, split, tokenizer, logger, num: int = 10):
    test_data = get_dataset(args, split=split, tokenizer=tokenizer, logger=logger)
    images = []
    for _ in range(num):
        idx = np.random.randint(0, len(test_data))
        images.append(test_data[idx]["pixel_values"])
    return images
