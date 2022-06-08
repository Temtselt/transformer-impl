import copy
import os
import time
from argparse import Namespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from transformer.embeddings import Embeddings
from transformer.encoder_decoder import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
)
from transformer.generator import Generator
from transformer.multi_head_attention import MultiHeadedAttention
from transformer.positionwise_feed_forward import PositionwiseFeedForward
from transformer.postional_encoding import PositionalEncoding
from utils.helpers import handle_dirs, set_seed_everywhere
from utils.logger import Logger

args = Namespace(
    dataset_csv="data/lyrics_lite.csv",
    vectorizer_file="vectorizer.json",
    model_state_file="model.pth",
    save_dir="model.storage/cyrillc_to_mongolian",
    reload_from_files=True,
    expand_filepaths_to_save_dir=True,
    cuda=False,
    seed=1337,
    learning_rate=5e-4,
    batch_size=8,
    num_epochs=100,
    early_stopping_criteria=5,
    source_embedding_size=32,
    target_embedding_size=32,
    encoding_size=32,
    catch_keyboard_interrupt=True,
)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout, 512)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
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
            print(
                "Epoch Step: %d Loss: %f Tokens per Sec: %f"
                % (i, loss / batch.ntokens, tokens / elapsed)
            )
            start = time.time()
            tokens = 0

    return total_loss / total_tokens


global max_src_in_batch, max_tgt_in_batch


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    """A generator function which wraps the PyTorch DataLoader."""
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )

    for data_dict in dataloader:
        lengths = data_dict["x_source_length"].numpy()
        sorted_length_indices = lengths.argsort()[::-1].tolist()

        out_data_dict = {}
        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name][sorted_length_indices].to(device)
        yield out_data_dict


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    global max_src_in_batch, max_tgt_in_batch
    if count == 1:
        max_src_in_batch = 0
        max_tgt_in_batch = 0
    max_src_in_batch = max(max_src_in_batch, len(new.src))
    max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
    src_elements = count * max_src_in_batch
    tgt_elements = count * max_tgt_in_batch
    return max(src_elements, tgt_elements)


if __name__ == "__main__":
    if args.expand_filepaths_to_save_dir:
        args.vectorizer_file = os.path.join(args.save_dir, args.vectorizer_file)
        args.model_state_file = os.path.join(args.save_dir, args.model_state_file)
        Logger.logi(
            __name__,
            "Expanded filepaths: \n"
            f"\t{args.vectorizer_file}\n"
            f"\t{args.model_state_file}",
        )

    if not torch.cuda.is_available():
        args.cuda = False
        Logger.logi(__name__, "CUDA is not available")

    args.device = torch.device("cuda" if args.cuda else "cpu")
    Logger.logi(__name__, f"Using CUDA: {args.cuda}")

    set_seed_everywhere(args.seed, args.cuda)
    handle_dirs(args.save_dir)
    tmp_model = make_model(10, 10, 2)
