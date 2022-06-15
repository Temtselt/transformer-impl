import os
import copy
from argparse import Namespace

import torch.nn as nn

from models.transformer import Decoder
from models.transformer import Encoder
from models.transformer import Generator
from models.transformer import Transformer
from models.transformer.blocks.decoder_layer import DecoderLayer
from models.transformer.blocks.encoder_layer import EncoderLayer
from models.transformer.embeddings.embedding import Embedding
from models.transformer.embeddings.postional_encoding import PositionalEncoding
from models.transformer.layers.positionwise_feed_forward import PositionwiseFeedForward


class Trainer:
    def __init__(self, args):
        self.args = args

    def make_model(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
        c = copy.deepcopy
        attn = MultiheadAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = Transformer(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(Embedding(d_model, src_vocab), c(position)),
            nn.Sequential(Embedding(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab)
        )

        # TODO
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return model

    def load_checkpoints(self):
        if args.reload_from_files and os.path.exists(args.model_state_file):
            model.load_state_dict(torch.load(args.model_state_file))
            Logger.logi(__class__, f"Reload model state from {args.model_state_file}")
        else:
            Logger.logi(__class__, "New model")

    def run_epoch(self):
        pass
        # TODO


if __name__ == "__main__":
    args = Namespace(
        N=6,
        d_model=512,
        d_ff=2048,
        h=8,
        reload_from_files=True,
        dataset_csv="data/lyrics.csv",
        vectorizer_file="vectorizer.json",
        model_state_file="model.pth",
        save_dir="data/model.storage",
    )
    trainer = Trainer(args)
    trainer.make_model(10, 10, 2)
