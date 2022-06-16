import pandas as pd
from torch.utils.data import Dataset
from utils.logger import Logger


class NMTDataset(Dataset):
    """
    Dataset class for the data.
    """

    def __init__(self, text_df):

        self.text_df = text_df

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

    @classmethod
    def load_dataset(cls, dataset_csv):
        """
        Loads the data from the csv file.
        """
        text_df = pd.read_csv(dataset_csv)
        return cls(text_df)

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        NotImplementedError  # TODO


if __name__ == "__main__":
    dataset = NMTDataset.load_dataset("data/lyrics_lite.csv")
    print(dataset._target_df)
