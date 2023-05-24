#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torchvision import transforms
from huggingface_hub import snapshot_download
from datasets import load_dataset
from transformers import CLIPTokenizer


def add_dataset_args(parser):
    dataset_group = parser.add_argument_group("dataset")
    dataset_group.add_argument(
        "--dataset",
        type=str,
        default="Norod78/simpsons-blip-captions",
    )
    dataset_group.add_argument(
        "--data_dir",
        type=str,
        default="datasets",
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


def get_dataset(
    args,
    split: str = "train",
    tokenizer: CLIPTokenizer = None,
):
    # check params
    assert tokenizer is not None, "you need to specify a tokenizer"
    assert split in {
        "train",
        "validation",
        "test",
    }, "split should be one of train, validation, test"
    # load dataset
    dataset = load_dataset(
        args.dataset_name,
        cache_dir=os.path.join(args.cache_dir, args.dataset_name),
        split=split,
    )

    image_column = [col for col in ["image", "img"] if col in dataset.column_names][0]
    caption_colum = [col for col in ["text", "caption"] if col in dataset.column_names][
        0
    ]

    transform = get_transform(args.resolution, args.random_flip, args.center_crop)

    def preprocess_train(examples):
        """tokenize captions and convert images to pixel values"""
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [transform(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples[caption_colum], tokenizer)
        return examples

    return dataset.with_transform(preprocess_train)
