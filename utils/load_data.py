#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   load_data.py
@Time    :   2023/05/27 12:20:44
@Author  :   Wenbo Li
@Desc    :   If load dataset failed, try this script
"""

from datasets import load_dataset
import os
from huggingface_hub import snapshot_download

load_dataset(
    "poloclub/diffusiondb",
    "2m_first_100k",
    cache_dir=os.path.join("data/datasets", "poloclub/diffusiondb"),
)

snapshot_download(
    "poloclub/diffusiondb",
    "2m_first_100k",
    repo_type="dataset",
    local_dir=os.path.join("data/datasets", "poloclub/diffusiondb"),
)
