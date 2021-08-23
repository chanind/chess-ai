from typing import List
import numpy as np
from torch.utils.data import Dataset
import os
from chess.pgn import read_game
from pathlib import Path

from .TrainingSample import TrainingSample
from chess_ai.translation.InputState import InputState
from chess_ai.translation.Action import Action


DATA_DIR = (Path(__file__) / ".." / ".." / ".." / "data").resolve()
MIN_ELO = 2300


def parse_pgn_training_samples(
    max_samples=None, min_elo=MIN_ELO, progress_indicator=True
) -> List[TrainingSample]:
    samples = []
    games_counter = 0
    # pgn files in the data folder
    for games_file in os.listdir(DATA_DIR):
        pgn = open(DATA_DIR / games_file)
        while True:
            game = read_game(pgn)
            if game is None:
                break
            if (
                int(game.headers["BlackElo"]) < min_elo
                or int(game.headers["WhiteElo"]) < min_elo
            ):
                continue
            board = game.board()
            for move in game.mainline_moves():
                sample = TrainingSample(
                    input=InputState(board), action=Action(move, board.turn)
                )
                samples.append(sample)
                board.push(move)
            if max_samples is not None and len(samples) >= max_samples:
                break
            games_counter += 1
            if progress_indicator and games_counter % 50 == 0:
                print(
                    f"\r{len(samples)} samples from {games_counter} games",
                    end="",
                    flush=True,
                )
    return samples
