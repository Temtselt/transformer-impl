import torch
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from utils.logger import Logger
from os.path import exists


def load_tokenizers(tokenizer_src_filepath, tokenizer_tgt_filepath):
    try:
        tokenizer_src = Tokenizer.from_file(tokenizer_src_filepath)
    except IOError:
        Logger.logw(__name__, f"Tokenizer file not found at {tokenizer_src_filepath}")

    try:
        tokenizer_tgt = Tokenizer.from_file(tokenizer_tgt_filepath)
    except IOError:
        Logger.logw(__name__, f"Tokenizer file not found at {tokenizer_tgt_filepath}")

    return tokenizer_src, tokenizer_tgt


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(tokenizer_src, tokenizer_tgt):
    def tokenize_src(text):
        return tokenize(text, tokenizer_src)

    def tokenize_mn(text):
        return tokenize(text, tokenizer_tgt)

    Logger.logi(__name__, "Building vocabulary from source text...")
    # TODO
    # train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    # vocab_src = build_vocab_from_iterator(
    #     yield_tokens(train + val + test, tokenize_de, index=0),
    #     min_freq=2,
    #     specials=["<s>", "</s>", "<blank>", "<unk>"],
    # )

    Logger.logi(__name__, "Building vocabulary from target text...")
    # TODO
    # train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    # vocab_tgt = build_vocab_from_iterator(
    #     yield_tokens(train + val + test, tokenize_en, index=1),
    #     min_freq=2,
    #     specials=["<s>", "</s>", "<blank>", "<unk>"],
    # )

    # vocab_src.set_default_index(vocab_src["<unk>"])
    # vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    # return vocab_src, vocab_tgt


def load_vocab(tokenizer_src, tokenizer_tgt):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(tokenizer_src, tokenizer_tgt)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")

    Logger.logi(
        __name__,
        f"Finished.\nVocabulary sizes:\nSource Vocabulary: \t{len(vocab_src)}\nTarget Vocabulary: \t{len(vocab_tgt)}",
    )

    return vocab_src, vocab_tgt


# def collate_batch(
#     batch,
#     vocab_src,
#     vocab_tgt,
#     tokenizer_src,
#     tokenizer_tgt,
#     device,
#     max_padding=128,
#     pad_id=2,
# ):
#     bos_id = torch.tensor([0], device=device)
#     eos_id = torch.tensor([1], device=device)
#     src_list, tgt_list = [], []

#     for (_src, _tgt) in batch:
#         pass
#         # TODO


# def create_dataloaders(
#     device,
#     vocab_src,
#     vocab_tgt,
#     tokenizer_src,
#     tokenizer_tgt,
#     batch_size=12000,
#     max_padding=128,
# ):
#     def collate_fn(batch):
#         return collate_batch(
#             batch,
#             vocab_src,
#             vocab_tgt,
#             tokenizer_src,
#             tokenizer_tgt,
#             max_padding=max_padding,
#             pad_id=vocab_src.get_stoi()["<blank>"],
#         )

#     train_dataloader = DataLoader(
#         train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
#     )

#     valid_dataloader = DataLoader(
#         valid_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
#     )
#     return train_dataloader, valid_dataloader
