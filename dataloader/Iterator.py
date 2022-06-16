from os.path import exists

import torch
from torch.utils.data import DataLoader


def collate_batch(
    batch,
    vocab_src,
    vocab_tgt,
    tokenizer_src,
    tokenizer_tgt,
    device,
    max_padding=128,
    pad_id=2,
):
    bos_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)
    src_list, tgt_list = [], []

    for (_src, _tgt) in batch:
        pass
        # TODO


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    tokenizer_src,
    tokenizer_tgt,
    batch_size=12000,
    max_padding=128,
):
    def collate_fn(batch):
        return collate_batch(
            batch,
            vocab_src,
            vocab_tgt,
            tokenizer_src,
            tokenizer_tgt,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_dataloader = DataLoader(
        train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    valid_dataloader = DataLoader(
        valid_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    return train_dataloader, valid_dataloader
