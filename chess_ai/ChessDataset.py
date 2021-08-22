import numpy as np
from torch.utils.data import Dataset
import os
from chess.pgn import read_game
from pathlib import Path
import torch

from .State import State, process_move_coords, serialization_to_tensor

DATA_DIR = (Path(__file__) / ".." / ".." / "data").resolve()
MIN_ELO = 2500


class ChessDataset(Dataset):
    def __init__(self, max_samples=None):
        self.X = []
        self.Y = []
        games_counter = 0
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
                board = game.board()
                for move in game.mainline_moves():
                    serialized_board = State(board).serialize()
                    self.X.append(serialized_board)
                    self.Y.append([move, board.turn])
                    board.push(move)
                if max_samples is not None and len(self.X) > max_samples:
                    break
                games_counter += 1
                if games_counter % 50 == 0:
                    print(f"\r{len(self.X)} samples from {games_counter} games", end="")
        print("loaded", len(self.X))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        input_tensor = serialization_to_tensor(self.X[idx])
        move, turn = self.Y[idx]
        move_coords = process_move_coords(move, turn)
        with torch.no_grad():
            target_move_tensor = torch.zeros(73, 8, 8)
            target_move_tensor[move_coords] = 1.0
        return (input_tensor, target_move_tensor)
