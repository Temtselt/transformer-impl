from utils.logger import Logger

from tokenizers import Tokenizer


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
