import numpy as np
import torch
from torch.autograd import Variable


class Batch:
  "Object for holding a batch of data with mask during training"

  def __init__(self, src, tgt=None, pad=0):
    self.src = src
    self.src_mask = (src != pad).unsqueeze(-2)  # 在倒数第二个维度上添加一个维度
    if tgt is not None:
      self.tgt = tgt[:, :-1]
      self.tgt_y = tgt[:, 1:]
      self.tgt_mask = \
          self._make_std_mask(self.tgt, pad)
      self.ntokens = (self.tgt_y != pad).data.sum()

  def _subsequent_mask(self, size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

  def _make_std_mask(self, tgt, pad):
    "Create mask and hide padding and future words"
    tgt_mask = (tgt != pad).unsqueeze(-2)
    tgt_mask = tgt_mask & Variable(
        self._subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

    return tgt_mask
