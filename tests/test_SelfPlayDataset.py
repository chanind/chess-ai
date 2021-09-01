from chess_ai.translation.Action import ACTION_PROBS_SHAPE
import torch
from random import random
import numpy as np

from chess_ai.data.SelfPlayDataset import SelfPlayDataset


def test_SelfPlayDataset_inputs_and_outputs():
    dataset = SelfPlayDataset("cpu", mcts_simulations=2, games_per_iteration=3)
    # just return random scores
    model = lambda _: (
        torch.rand(*ACTION_PROBS_SHAPE).unsqueeze(0),
        torch.tensor(2 * random() - 1).unsqueeze(0),
    )
    dataset.generate_self_play_data(model)
    assert len(dataset.train_examples_history) == 3
