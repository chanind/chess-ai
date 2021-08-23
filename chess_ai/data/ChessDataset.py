from chess_ai.data.TrainingSample import TrainingSample
from typing import Sequence
import numpy as np
from torch.utils.data import Dataset
import os
from chess.pgn import read_game
from pathlib import Path
import torch


DATA_DIR = (Path(__file__) / ".." / ".." / "data").resolve()
MIN_ELO = 2300


class ChessDataset(Dataset):
    training_samples: Sequence[TrainingSample]

    def __init__(self, training_samples: Sequence[TrainingSample]):
        self.training_samples = training_samples

    def __len__(self):
        return len(self.training_samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_tensor = sample.input.to_tensor()
        move_tensor = sample.action.to_tensor()
        return (input_tensor, move_tensor)
