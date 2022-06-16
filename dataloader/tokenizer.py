from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from torch.utils.data import DataLoader
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator
from utils.logger import Logger

from dataloader.dataset import NMTDataset


def make_tokenizer_train(files, save_path):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.NFD()
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    tokenizer.train(files, trainer)
    tokenizer.save(save_path)

    return tokenizer


def load_saved_tokenizer(tokenizer_path):
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except IOError:
        Logger.logw(__name__, f"Tokenizer file not found at {tokenizer_path}")

    return tokenizer


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.encode(text)]


def yield_tokens(data_iter, tokenizer, index):
    for from_to_tuple in data_iter:
        yield tokenizer(from_to_tuple[index])


def build_vocabulary(tokenizer_src, tokenizer_tgt):
    def tokenize_src(text):
        return tokenize(text, tokenizer_src)

    def tokenize_tgt(text):
        return tokenize(text, tokenizer_tgt)

    dataset = NMTDataset.load_dataset("data/lyrics_lite.csv")
    dataset.set_split("train")

    src_vocab = build_vocab_from_iterator(
        yield_tokens(dataset._target_df["cyrillic"], tokenize_src, 0),
        min_freq=1,
    )

    tgt_vocab = build_vocab_from_iterator(
        yield_tokens(dataset._target_df["bichig"], tokenize_tgt, 1), min_freq=1
    )

    return src_vocab, tgt_vocab


if __name__ == "__main__":
    data = [
        "авааль нөхөр алтансүх минь"
        "авааль нөхөр алтансүхтэйгээ"
        "авааль нөхөрт чинь ховлохгүйэ"
        "аваачаад өгье гэсэн чинь"
        "ав ав ав түүнийг ав сав сав сав түүнтэй сав"
    ]
    # nmt_tokenizer = NMTTokenizer()
    # files = ["data/lyrics_lite.txt"]
    # nmt_tokenizer.train_tokenizer(files, "data/lyrics_lite.json")

    tokenizer = Tokenizer.from_file("data/lyrics_lite.json")
    src_vocab, tgt_vocab = build_vocabulary(tokenizer, tokenizer)
