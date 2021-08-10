import torch.nn as nn
import torch


class SkipBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
        )

    def forward(self, x):
        out = self.layer(x)
        return out + x


class ChessValueModel(nn.Module):
    def __init__(self, hidden_size=128, num_blocks=8):
        super().__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(901, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.8),
        )
        self.blocks = nn.Sequential(
            *[SkipBlock(hidden_size) for _ in range(num_blocks)]
        )
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.blocks(x)
        x = self.out_layer(x)
        return torch.tanh(x)
