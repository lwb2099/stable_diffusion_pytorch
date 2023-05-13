#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torchvision import transforms
from huggingface_hub import snapshot_download
from datasets import load_dataset


def collate_fn(examples):
    pass

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
        captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids


def get_dataset(dataset_name, save_dir, tokenizer, transform, train_batch_size, num_workers):
    # load dataset
    dataset = load_dataset(
        os.path.join(save_dir, dataset_name),
        )
    
    #TODO: fit all datasets to the same format
    image_column = [col for col in ["image", "img"] if col in dataset["train"].column_names][0] 
    caption_colum = [col for col in ["text", "caption"] if col in dataset["train"].column_names][0]
    def preprocess_train(examples):
        """tokenize captions and convert images to pixel values"""
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [transform(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples[caption_colum], tokenizer)
        return examples
    
    train_dataset = dataset["train"].with_transform(preprocess_train)
    # create dataloader
    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_batch_size,
        num_workers=num_workers,
    )