import numpy as np
from torch.utils.data import Dataset
import os
from chess.pgn import read_game
from pathlib import Path
import torch

from .State import State, serialization_to_tensor

DATA_DIR = (Path(__file__) / ".." / ".." / "data").resolve()
MIN_ELO = 2000


class ChessDataset(Dataset):
    def __init__(self, max_samples=None):
        self.X = []
        self.Y = []
        games_counter = 0
        values = {"1/2-1/2": 0, "0-1": -1, "1-0": 1}
        # pgn files in the data folder
        for games_file in os.listdir(DATA_DIR):
            pgn = open(DATA_DIR / games_file)
            while True:
                game = read_game(pgn)
                if game is None:
                    break
                if (
                    int(game.headers["BlackElo"]) < MIN_ELO
                    or int(game.headers["WhiteElo"]) < MIN_ELO
                ):
                    continue
                res = game.headers["Result"]
                if res not in values:
                    continue
                value = values[res]
                board = game.board()
                for move in game.mainline_moves():
                    board.push(move)
                    ser = State(board).serialize()
                    self.X.append(ser)
                    self.Y.append(value)
                if max_samples is not None and len(self.X) > max_samples:
                    break
                games_counter += 1
                if games_counter % 50 == 0:
                    print(f"\r{len(self.X)} samples from {games_counter} games", end="")
        print("loaded", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_tensor = serialization_to_tensor(self.X[idx])
        y_tensor = torch.tensor(self.Y[idx]).unsqueeze(-1).float()
        return (x_tensor, y_tensor)
