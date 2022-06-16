import copy
import os
import time
from argparse import Namespace

import torch
import torch.nn as nn

from model.layers.decoder import Decoder, DecoderLayer
from model.layers.embedding import Embedding
from model.layers.encoder import Encoder, EncoderLayer
from model.layers.generator import Generator
from model.layers.multi_head_attention import MultiHeadAttention
from model.layers.positionwise_feed_forward import PositionwiseFeedForward
from model.layers.postional_encoding import PositionalEncoding
from model.transformer import Transformer
from utils.bookkeeping import make_train_state
from utils.logger import Logger


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embedding(d_model, src_vocab), c(position)),
        nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    Logger.logi(__name__, "Made model")
    return model


def load_checkpoints(model):
    if args.reload_from_files and os.path.exists(args.model_state_file):
        model.load_state_dict(torch.load(args.model_state_file))
        Logger.logi(__name__, f"Reload model state from {args.model_state_file}")
    else:
        Logger.logi(__name__, "New model")


def run_epoch(
    data_iter,
    model,
    loss_compute,
    optimizer,
    scheduler,
    mode="train",
    accum_iter=1,
    train_state=[],
):
    """Train a single epoch"""

    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    running_loss = 0
    running_acc = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(
            batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
        )  # compute the output
        loss, loss_node = loss_compute(
            out, batch.tgt_y, batch.ntokens
        )  # compute the loss
        if mode == "train" or mode == "train+log":
            loss_node.backward()  # use loss to produce gradients
            train_state["step"] += 1
            train_state["samples"] += batch.src.shape[0]
            train_state["tokens"] += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()  # use optimizer to take gradient step
                optimizer.zero_grad(set_to_none=True)  # reset gradients
                n_accum += 1
                train_state["accum_step"] += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                    "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                    + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


if __name__ == "__main__":
    args = Namespace(
        N=6,
        d_model=512,
        d_ff=2048,
        h=8,
        dataset_csv="data/lyrics.csv",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir="data/model.storage",
        reload_from_files=True,
        expand_filepaths_to_save_dir=True,
        cuda=False,
        seed=1337,
        learning_rate=5e-4,
        batch_size=16,
        num_epochs=10,
        early_stopping_criteria=5,
    )
    model = make_model(10, 10)
    load_checkpoints(model)
    train_state = make_train_state(args)
