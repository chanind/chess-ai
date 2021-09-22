from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import chess
import chess.pgn
import torch

# import logging
from thespian.actors import Actor

from chess_ai.translation.find_move_from_action_coord import (
    find_move_from_action_coord,
)
from chess_ai.translation.Action import unravel_action_index
from chess_ai.translation.InputState import InputState
from chess_ai.translation.BoardWrapper import BoardWrapper, get_next_board_wrapper
from chess_ai.ChessMCTS import ChessMCTS

# based on https://github.com/suragnair/alpha-zero-general/blob/master/Coach.py

# log = logging.getLogger(__name__)


@dataclass
class PlayGameMessage:
    mcts_simulations: int
    temp_threshold: int
    device: torch.device
    model_predict_actor_addr: any


@dataclass
class PlayGameResult:
    train_examples: List[Tuple[InputState, torch.Tensor, int]]
    pgn: chess.pgn.Game


class SelfPlayGameActor(Actor):
    def receiveMessage(self, msg, sender):
        if isinstance(msg, PlayGameMessage):
            mcts = ChessMCTS(
                device=msg.device,
                loader=msg.model_predict_actor_addr,
                num_simulations=msg.mcts_simulations,
            )
            train_examples, pgn = self.selfplay_game(mcts, msg.temp_threshold)
            self.send(sender, PlayGameResult(train_examples=train_examples, pgn=pgn))

    def selfplay_game(self, mcts: ChessMCTS, temp_threshold: int):
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
        board_wrapper = BoardWrapper(chess.Board())
        episode_step = 0

        game = chess.pgn.Game()
        node = game

        while True:
            episode_step += 1
            temp = int(episode_step < temp_threshold)

            pi = mcts.get_action_probabilities(board_wrapper, temp=temp)

            action_index = np.random.choice(pi.size, p=pi.flatten())
            action_coord = unravel_action_index(action_index)
            train_examples.append((InputState(board_wrapper.board), action_index, 0))

            move = find_move_from_action_coord(action_coord, board_wrapper.board)

            board_wrapper = get_next_board_wrapper(board_wrapper, move)
            node = node.add_variation(move)

            if board_wrapper.board.is_game_over():
                game.headers["Result"] = board_wrapper.board.result()
                outcome = board_wrapper.board.outcome()
                if outcome.winner is None:
                    return train_examples
                result = 1 if outcome.winner == chess.WHITE else -1

                adjusted_train_examples = []
                for train_example in train_examples:
                    example_state = train_example[0]
                    example_action = train_example[1]
                    adjusted_train_examples.append(
                        (
                            example_state,
                            example_action,
                            result * ((-1) ** (example_state.turn == chess.BLACK)),
                        )
                    )
                return adjusted_train_examples, game
