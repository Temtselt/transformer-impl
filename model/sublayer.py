import torch.nn as nn

from model.norm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout) -> None:
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
