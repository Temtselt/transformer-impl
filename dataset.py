import json
import pandas as pd
from torch.utils.data import Dataset

from vectorizer import Vectorizer


class Dataset(Dataset):
    def __init__(self, text_df, vectorizer):
        self.text_df = text_df
        self._vectorizer = vectorizer

        self.train_df = text_df[text_df.split == "train"]
        self.train_size = len(self.train_df)

        self.val_df = text_df[text_df.split == "val"]
        self.validation_size = len(self.val_df)

        self.test_df = text_df[text_df.split == "test"]
        self.test_size = len(self.test_df)

        self._lookup_dict = {
            "train": (self.train_df, self.train_size),
            "val": (self.val_df, self.validation_size),
            "test": (self.test_df, self.test_size),
        }

        self.set_split("train")

    def set_split(self, split="train"):
        self._target_split = split
        self._target_df, self._target_size = self._lookup_dict[split]

    @classmethod
    def load_dataset_and_make_vectorizer(cls, dataset_csv):
        text_df = pd.read_csv(dataset_csv)
        train_subset = text_df[text_df.split == "train"]

        return cls(text_df, Vectorizer.from_dataframe(train_subset))

    @classmethod
    def load_dataset_and_load_vectorizer(cls, dataset_csv, vectorizer_filepath):
        text_df = pd.read_csv(dataset_csv)
        vectorizer = cls.load_vectorizer_only(vectorizer_filepath)

        return cls(text_df, vectorizer)

    @staticmethod
    def load_vectorizer_only(vectorizer_filepath):
        with open(vectorizer_filepath) as fp:
            Vectorizer.from_serializable(json.load(fp))

    def save_vectorizer(self, vectorizer_filepath):
        with open(vectorizer_filepath, "w") as fp:
            json.dump(Vectorizer.to_serializable(), fp)

    def get_vectorizer(self):
        return self._vectorizer

    def __len__(self):
        return self._target_size

    def __getitem__(self, index):
        """the primary entry point method for PyTorch datasets

        Args:
            index (int): the index to the data point
        Returns:
            a dictionary holding the data point: (x_data, y_target, class_index)
        """
        row = self._target_df.iloc[index]

        vector_dict = self._vectorizer.vectorize(row.source_text, row.target_text)

        return {
            "x_source": vector_dict["source_vector"],
            "x_target": vector_dict["target_x_vector"],
            "y_target": vector_dict["target_y_vector"],
            "x_source_length": vector_dict["source_length"],
        }

    def get_num_batches(self, batch_size):
        """Given a batch size, return the number of batches in the dataset

        Args:
            batch_size (int)
        Returns:
            number of batches in the dataset
        """
        return len(self) // batch_size
