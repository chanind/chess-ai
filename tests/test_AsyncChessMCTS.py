from chess_ai.AsyncPredictDataLoader import AsyncPredictDataLoader
import numpy as np
import torch
import chess
import pytest
from random import random

from chess_ai.AsyncChessMCTS import AsyncChessMCTS
from chess_ai.translation.Action import Action, ACTION_CHANNELS
from chess_ai.translation.BoardWrapper import BoardWrapper


class MockModel:
    def predict(self, input):
        return (
            torch.rand(input.shape[0], ACTION_CHANNELS, 8, 8),
            torch.ones(input.shape[0]) * (2 * random() - 1),
        )


@pytest.mark.asyncio
async def test_AsyncChessMCTS_runs():
    # just return random scores
    model = MockModel()
    board = chess.Board()
    mcts = AsyncChessMCTS(
        AsyncPredictDataLoader(model), "cpu", num_simulations=10, cpuct=1.0
    )

    probs = await mcts.get_action_probabilities(BoardWrapper(board))
    assert probs.shape == (ACTION_CHANNELS, 8, 8)
    assert probs.sum() == pytest.approx(1.0)
    assert probs.max() <= 1.0
    assert probs.min() >= 0.0

    valid_action_coords = [
        Action(move, board.turn).coords for move in board.legal_moves
    ]

    for action_index in range(np.prod(probs.shape)):
        action_coord = np.unravel_index(action_index, probs.shape)
        if action_coord not in valid_action_coords:
            assert probs[action_coord] == 0
