import numpy as np

from vocabulary import SequenceVocabulary


class Vectorizer(object):
    """The Vectorizer which coordinates the Vocabularies and puts them to use"""

    def __init__(
        self, source_vocab, target_vocab, max_source_length, max_target_length
    ) -> None:
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def _get_source_indices(self, source_text):
        indices = [self.source_vocab.begin_seq_index]
        indices.extend(
            self.source_vocab.lookup_token(token) for token in source_text.split(" ")
        )
        indices.append(self.source_vocab.end_seq_index)

        return indices

    def _get_target_indices(self, target_text):
        indices = [
            self.source_vocab.lookup_token(token) for token in target_text.split(" ")
        ]
        x_indices = [self.source_vocab.begin_seq_indices] + indices
        y_indices = indices + [self.target_vocab.end_seq_indices]

        return x_indices, y_indices

    def _vectorize(self, indices, vector_length, mask_index):
        if vector_length < 0:
            vector_length = len(indices)

        vector = np.zeros(vector_length, dtype=np.int64)
        vector[: len(indices)] = indices
        vector[len(indices) :] = mask_index

        return vector

    def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
        source_vector_length = 0
        target_vector_length = 0

        if use_dataset_max_lengths:
            source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_source_length + 1

        source_indices = self._get_source_indices(source_text)
        source_vector = self._vectorize(
            source_indices,
            vector_length=source_vector_length,
            mask_index=self.source_vocab.mask_index,
        )

        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(
            target_x_indices,
            vector_length=target_vector_length,
            mask_index=self.target_vocab.mask_index,
        )
        target_y_vector = self._vectorize(
            target_y_indices,
            vector_length=target_vector_length,
            mask_index=self.target_vocab.mask_index,
        )

        return {
            "source_vector": source_vector,
            "target_x_vector": target_x_vector,
            "target_y_vector": target_y_vector,
            "source_length": len(source_indices),
        }

    @classmethod
    def from_dataframe(cls, bitext_df):
        """Instantiate the vectorizer from the dataset dataframe

        Args:
            bitext_df (pandas.DataFrame): the parallel text dataset
        Returns:
            an instance of the NMTVectorizer
        """
        source_vocab = SequenceVocabulary()
        target_vocab = SequenceVocabulary()

        max_source_length = 0
        max_target_length = 0

        for _, row in bitext_df.iterrows():
            source_tokens = row["source_text"].split(" ")
            target_tokens = row["target_text"].split(" ")

            if len(source_tokens) > max_source_length:
                max_source_length = len(source_tokens)

            if len(target_tokens) > max_target_length:
                max_target_length = len(target_tokens)

            source_vocab.add_token(token for token in source_tokens)
            target_vocab.add_token(token for token in target_tokens)

        return cls(source_vocab, target_vocab, max_source_length, max_target_length)

    @classmethod
    def from_serializable(cls, contents):
        source_vocab = SequenceVocabulary.from_serializable(contents["source_vocab"])
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])

        return cls(
            source_vocab=source_vocab,
            target_vocab=target_vocab,
            max_source_length=contents["max_source_length"],
            max_target_length=contents["max_target_length"],
        )

    def to_serializable(self):

        return {
            "source_vocab": self.source_vocab.to_serializable(),
            "target_vocab": self.target_vocab.to_serializable(),
            "max_source_length": self.max_source_length,
            "max_target_length": self.max_target_length,
        }
