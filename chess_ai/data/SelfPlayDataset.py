from chess_ai.data.SelfPlayGamesManagerActor import (
    PlayGamesMessage,
    SelfPlayGamesManagerActor,
)
from typing import Tuple
from thespian.actors import ActorExitRequest, ActorSystem
from torch.utils.data import Dataset
import numpy as np
import torch

# import logging

from chess_ai.translation.InputState import InputState
from chess_ai.ChessModel import ChessModel
from chess_ai.ModelPredictActor import InitModelPredictActorMessage, ModelPredictActor

# based on https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py


class SelfPlayDataset(Dataset):

    current_training_examples: Tuple[InputState, np.ndarray, int]

    def __init__(
        self,
        device: torch.device,
        mcts_simulations: int = 50,
        games_per_iteration: int = 100,
        max_recent_training_games: int = 200000,
        temp_threshold: int = 15,
    ):
        super().__init__()
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.temp_threshold = temp_threshold
        self.max_recent_training_games = max_recent_training_games
        self.train_examples_history = []
        self.current_training_examples = []
        self.device = device

    def generate_self_play_data(self, model: ChessModel):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        actor_sys = ActorSystem()
        model_predict_loader = actor_sys.createActor(ModelPredictActor)
        actor_sys.tell(model_predict_loader, InitModelPredictActorMessage(model=model))
        self_play_manager = actor_sys.createActor(SelfPlayGamesManagerActor)
        print("INIT THE LOADER", model_predict_loader)
        games_training_examples = actor_sys.ask(
            self_play_manager,
            PlayGamesMessage(
                mcts_simulations=self.mcts_simulations,
                temp_threshold=self.temp_threshold,
                device=self.device,
                model_predict_actor_addr=model_predict_loader,
                num_games=self.games_per_iteration,
            ),
        )
        actor_sys.tell(self_play_manager, ActorExitRequest())
        actor_sys.tell(model_predict_loader, ActorExitRequest())
        self.train_examples_history = (
            self.train_examples_history + games_training_examples
        )
        if len(self.train_examples_history) > self.max_recent_training_games:
            self.train_examples_history = self.train_examples_history[
                -1 * self.max_recent_training_games :
            ]

        self.current_training_examples = []
        for examples in self.train_examples_history:
            self.current_training_examples.extend(examples)
        return self.current_training_examples

    def __len__(self):
        return len(self.current_training_examples)

    def __getitem__(self, idx):
        input_state, pi, outcome = self.current_training_examples[idx]
        return (
            input_state.to_tensor(),
            torch.from_numpy(pi),
            torch.tensor(outcome, dtype=torch.float),
        )
