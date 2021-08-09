import numpy as np
from torch.utils.data import Dataset
import os
from chess.pgn import read_game
from pathlib import Path
import torch

from .State import State

DATA_DIR = (Path(__file__) / ".." / ".." / "data").resolve()

class ChessDataset(Dataset):
  def __init__(self, max_samples = None):
    X = []
    Y = []
    gn = 0
    values = {'1/2-1/2':0, '0-1':-1, '1-0':1}
    # pgn files in the data folder
    for games_file in os.listdir(DATA_DIR):
        pgn = open(DATA_DIR / games_file)
        while True:
            game = read_game(pgn)
            if game is None:
                break
            res = game.headers['Result']
            if res not in values:
                continue
            value = values[res]
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                ser = State(board).serialize()
                X.append(ser)
                Y.append(value)
            if max_samples is not None and len(X) > max_samples:
                break
            gn += 1
            if gn % 100 == 0:
                print('.')
    self.X = torch.stack(X)
    self.Y = torch.Tensor(Y)
    print("loaded", self.X.shape, self.Y.shape)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, idx):
    return (self.X[idx], self.Y[idx])