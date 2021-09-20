from chess_ai.ModelPredictActor import InitModelPredictActorMessage, ModelPredictActor
import numpy as np
import torch
import chess
import pytest
from random import random
from thespian.actors import ActorExitRequest, ActorSystem

from chess_ai.ChessMCTS import ChessMCTS
from chess_ai.translation.Action import Action, ACTION_CHANNELS
from chess_ai.translation.BoardWrapper import BoardWrapper


class MockModel:
    def predict(self, input):
        return (
            torch.rand(input.shape[0], ACTION_CHANNELS, 8, 8),
            torch.ones(input.shape[0]) * (2 * random() - 1),
        )


def test_ChessMCTS_runs():
    # just return random scores
    model = MockModel()
    board = chess.Board()
    loader = ActorSystem().createActor(ModelPredictActor)
    ActorSystem().tell(loader, InitModelPredictActorMessage(model=model))
    mcts = ChessMCTS(loader, "cpu", num_simulations=10, cpuct=1.0)
    probs = mcts.get_action_probabilities(BoardWrapper(board))
    ActorSystem().tell(loader, ActorExitRequest())

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
