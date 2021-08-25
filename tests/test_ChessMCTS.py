import numpy as np
import torch
import chess
import pytest
from random import random

from chess_ai.ChessMCTS import ChessMCTS
from chess_ai.translation.Action import Action, ACTION_CHANNELS


def test_ChessMCTS_runs():
    # just return random scores
    model = lambda _: (
        torch.rand(ACTION_CHANNELS, 8, 8).unsqueeze(0),
        torch.tensor(2 * random() - 1).unsqueeze(0),
    )
    board = chess.Board()
    mcts = ChessMCTS(model, num_simulations=10, cpuct=1.0)

    probs = mcts.get_action_probabilities(board)
    assert probs.shape == (ACTION_CHANNELS, 8, 8)
    assert np.sum(probs) == pytest.approx(1.0)
    assert np.max(probs) <= 1.0
    assert np.min(probs) >= 0.0

    valid_action_coords = [
        Action(move, board.turn).coords for move in board.legal_moves
    ]

    for action_index in range(np.prod(probs.shape)):
        action_coord = np.unravel_index(action_index, probs.shape)
        if action_coord not in valid_action_coords:
            assert probs[action_coord] == 0
