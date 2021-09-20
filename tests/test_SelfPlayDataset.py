from chess_ai.translation.Action import ACTION_CHANNELS
import torch
from random import random

from chess_ai.data.SelfPlayDataset import SelfPlayDataset


class MockModel:
    def predict(self, input):
        return (
            torch.rand(input.shape[0], ACTION_CHANNELS, 8, 8),
            torch.ones(input.shape[0]) * (2 * random() - 1),
        )


def test_SelfPlayDataset_inputs_and_outputs():
    dataset = SelfPlayDataset("cpu", mcts_simulations=2, games_per_iteration=3)
    # just return random scores
    model = MockModel()
    dataset.generate_self_play_data(model)
    assert len(dataset.train_examples_history) == 3
