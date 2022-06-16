import pandas as pd
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from utils.logger import Logger


class NMTDataset(Dataset):
    """
    Dataset class for the data.
    """

    def __init__(self, text_df, tokenizer):

        self.text_df = text_df
        self._tokenizer = tokenizer

        self.train_df = self.text_df[self.text_df["split"] == "train"]
        self.train_size = len(self.train_df)

        self.val_df = self.text_df[self.text_df["split"] == "val"]
        self.val_size = len(self.val_df)

        self.test_df = self.text_df[self.text_df["split"] == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.val_size),
            "test": (self.test_df, self.test_size),
        }

        Logger.logi(
            __class__,
            f"Initialized dataset: train_size={self.train_size}, val_size={self.val_size}, test_size={self.test_size}",
        )

        self.set_split("train")

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    def get_tokenizer(self):
        return self._tokenizer

    @classmethod
    def load_dataset_and_load_tokenizer(cls, dataset_csv, tokenizer_filepath):
        """
        Load dataset and the corresponding vectorizer.
        Used in the case in the vectorizer has been cached for re-use
        """

        text_df = pd.read_csv(dataset_csv)
        tokenizer = cls.load_tokenizer_only(tokenizer_filepath)

        return cls(text_df, tokenizer)

    @classmethod
    def load_dataset_and_make_tokenizer(cls, dataset_csv):
        """Load dataset and make a new vectorizer from scratch"""
        NotImplemented  # TODO

    @staticmethod
    def load_tokenizer_only(tokenizer_filepath):
        """a static method for loading the tokenizer from file"""

        return Tokenizer.from_file(tokenizer_filepath)

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        NotImplementedError  # TODO

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size


if __name__ == "__main__":
    dataset = NMTDataset.load_dataset_and_load_tokenizer(
        "data/lyrics_lite.csv", "data/lyrics_lite.json"
    )
    print(dataset._target_df)
