class Vocabulary(object):
    """Class to process text and extract vocabulary for mapping"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>") -> None:
        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}

    def to_serializable(self):
        return {
            "token_to_index": self._token_to_idx,
        }

    @classmethod
    def from_serializable(cls, content):
        return cls(**content)

    def add_token(self, token):
        try:
            index = self._token_to_idx[token]
        except KeyError:
            index = len(self._token_to_idx)

            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    def add_many(self, tokens):
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Retrieve the index associated with the token

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        """
        return self._token_to_idx[token]

    def lookup_index(self, index):
        """Return the token associated with the index

        Args:
            index (int): the index to look up
        Returns:
            token (str): the token corresponding to the index
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError(f"the index {index} is not in the Vocabulary")

        return self._idx_to_token[index]

    def __str__(self) -> str:
        return f"<Vocabular(size={len(self)}>"

    def __len__(self):
        return len(self._token_to_idx)


class SequenceVocabulary(Vocabulary):
    def __init__(
        self,
        token_to_index=None,
        unk_token="<UNK>",
        mask_token="<MASK>",
        begin_seq_token="<BEGIN>",
        end_seq_token="<END>",
    ):
        super(SequenceVocabulary, self).__init__(token_to_index)

        self._unk_token = unk_token
        self._mask_token = mask_token
        self._begin_seq_token = begin_seq_token
        self._end_seq_token = end_seq_token

        self.unk_index = self.add_token(self._unk_token)
        self.mask_index = self.add_token(self._mask_token)
        self.begin_seq_index = self.add_token(self._begin_seq_token)
        self.end_seq_index = self.add_token(self._end_seq_token)

    def to_serializable(self):
        contents = super(SequenceVocabulary, self).to_serializable()
        contents.update(
            {
                "unk_token": self._unk_token,
                "mask_token": self._mask_token,
                "begin_seq_token": self._begin_seq_token,
                "end_seq_token": self._end_seq_token,
            }
        )

        return contents

    def lookup_token(self, token):
        """Retrieve the index associated with the token
          or the UNK index if token isn't present.

        Args:
            token (str): the token to look up
        Returns:
            index (int): the index corresponding to the token
        Notes:
            `unk_index` needs to be >=0 (having been added into the Vocabulary)
              for the UNK functionality
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
