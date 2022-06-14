import copy
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


def create_dataframe(txt_path):
    df = pd.read_csv(txt_path, sep="|", names=["cyrillic", "bichig"])
    train, rem = train_test_split(df, test_size=0.3)
    val, test = train_test_split(rem, test_size=0.5)

    train["split"] = "train"
    val["split"] = "val"
    test["split"] = "test"

    df = pd.concat([train, val, test], ignore_index=True)

    with open("data/lyrics.csv", "w") as fp:
        df.to_csv(fp, index=False)


def set_seed_everywhere(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def handle_dirs(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def clones(module, N):
    "Produce N identical layers."

    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
