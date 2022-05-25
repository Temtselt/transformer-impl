import copy
import time

import torch.nn as nn

from models.embeddings import Embeddings
from models.encoder_decoder import (Decoder, DecoderLayer, Encoder,
                                    EncoderDecoder, EncoderLayer)
from models.generator import Generator
from models.multi_head_attention import MultiHeadedAttention
from models.positionwise_feed_forward import PositionwiseFeedForward
from models.postional_encoding import PositionalEncoding


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
  c = copy.deepcopy
  attn = MultiHeadedAttention(h, d_model)
  ff = PositionwiseFeedForward(d_model, d_ff, dropout)
  position = PositionalEncoding(d_model, dropout, 512)

  model = EncoderDecoder(
      Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
      Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
      nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
      nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
      Generator(d_model, tgt_vocab)
  )

  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)

  return model


def run_epoch(data_iter, model, loss_compute):
  start = time.time()
  total_loss = 0
  total_tokens = 0
  tokens = 0
  for i, batch in enumerate(data_iter):
    out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
    loss = loss_compute(out, batch.tgt_y, batch.ntokens)
    total_loss += loss
    total_tokens += batch.ntokens
    tokens = batch.ntokens

    if i % 50 == 1:
      elapsed = time.time() - start
      print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
            (i, loss / batch.ntokens, tokens / elapsed))
      start = time.time()
      tokens = 0

  return total_loss / total_tokens

global max_src_in_batch, max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch,  len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


if __name__ == '__main__':
  tmp_model = make_model(10, 10, 2)
