from promise import Promise
from collections import deque
from random import shuffle
from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
import chess
import torch
import logging

from chess_ai.translation.Action import ACTION_PROBS_SHAPE, Action
from chess_ai.translation.InputState import InputState
from chess_ai.ChessModel import ChessModel
from chess_ai.AsyncChessMCTS import AsyncChessMCTS
from chess_ai.AsyncPredictDataLoader import AsyncPredictDataLoader

# based on https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py

log = logging.getLogger(__name__)


def find_move_from_action_coord(
    action_coord: np.ndarray, board: chess.Board
) -> chess.Move:
    for move in board.legal_moves:
        if Action(move, board.turn).coords == action_coord:
            return move
    raise Exception(f"No legal move found for action coord: {action_coord}")


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

    async def generate_self_play_data(self, model: ChessModel):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        loader = AsyncPredictDataLoader(model)

        await Promise.all(
            [
                self.generate_single_selfplay_game_data(loader, i)
                for i in range(self.games_per_iteration)
            ]
        )

    async def generate_single_selfplay_game_data(
        self, loader: AsyncPredictDataLoader, gamenum: int
    ):
        # bookkeeping
        log.info(f"Starting game {gamenum} ...")

        mcts = AsyncChessMCTS(
            loader, self.device, self.mcts_simulations
        )  # reset search tree
        train_examples = await self.selfplay_game(mcts)

        # save the iteration examples to the history
        self.train_examples_history.append(train_examples)

        if len(self.train_examples_history) > self.max_recent_training_games:
            log.warning(
                f"Removing the oldest entry in train_examples_history. len(train_examples_history) = {len(self.train_examples_history)}"
            )
            self.train_examples_history.pop(0)
        # backup history to a file
        # NB! the examples were collected using the model from the previous iteration, so (i-1)
        # self.saveTrainExamples(i - 1)

        # shuffle examples before training
        self.current_training_examples = []
        for e in self.train_examples_history:
            self.current_training_examples.extend(e)
        shuffle(self.current_training_examples)

        log.info(f"completed game {gamenum}")

    async def selfplay_game(self, mcts: AsyncChessMCTS):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        train_examples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in train_examples.
        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.
        Returns:
            train_examples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        train_examples = []
        board = chess.Board()
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.temp_threshold)

            pi = await mcts.get_action_probabilities(board, temp=temp)
            train_examples.append((InputState(board), pi, 0))

            action_index = np.random.choice(pi.size, p=pi.flatten())
            action_coord = np.unravel_index(action_index, ACTION_PROBS_SHAPE)

            move = find_move_from_action_coord(action_coord, board)

            board.push(move)

            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner is None:
                    return train_examples
                result = 1 if outcome.winner == chess.WHITE else -1

                adjusted_train_examples = []
                for train_example in train_examples:
                    state = train_example[0]
                    pi = train_example[1]
                    adjusted_train_examples.append(
                        (
                            state,
                            pi,
                            result * ((-1) ** (state.turn == chess.BLACK)),
                        )
                    )

                return adjusted_train_examples

    def __len__(self):
        return len(self.current_training_examples)

    def __getitem__(self, idx):
        input_state, pi, outcome = self.current_training_examples[idx]
        return (input_state.to_tensor(), torch.from_numpy(pi), torch.tensor(outcome))
