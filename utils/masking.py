import numpy as np
import torch
from torch.autograd import Variable


def subsequent_mask(size):
    "Mask out subsequent positions."

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")

    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(tgt, pad):
    "Create mask and hide padding and future words"

    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

    return tgt_mask
