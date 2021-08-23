from chess_ai.data.TrainingSample import TrainingSample
from typing import Sequence
from torch.utils.data import Dataset
from pathlib import Path


DATA_DIR = (Path(__file__) / ".." / ".." / "data").resolve()
MIN_ELO = 2300


class ChessDataset(Dataset):
    samples: Sequence[TrainingSample]

    def __init__(self, samples: Sequence[TrainingSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_tensor = sample.input.to_tensor()
        move_tensor = sample.action.to_tensor()
        return (input_tensor, move_tensor)
