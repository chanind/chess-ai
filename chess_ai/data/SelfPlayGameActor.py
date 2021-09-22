from chess_ai.ModelPredictActor import PredictionResultMessage
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import chess
import chess.pgn
import torch
from promise import Promise

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
    pending_predictions = {}

    def receiveMessage(self, msg, sender):
        if isinstance(msg, PlayGameMessage):

            def load_prediction(key, input):
                def resolver(resolve, _reject):
                    self.pending_predictions[key] = resolve

                res_promise = Promise(resolver)
                self.send(msg.model_predict_actor_addr, input)
                return res_promise

            mcts = ChessMCTS(
                device=msg.device,
                loader=load_prediction,
                num_simulations=msg.mcts_simulations,
            )
            self.selfplay_game(mcts, msg.temp_threshold).then(
                lambda res: self.send(
                    sender, PlayGameResult(train_examples=res[0], pgn=res[1])
                )
            )

        if isinstance(msg, PredictionResultMessage):
            key = msg.key
            self.pending_predictions[key](msg.output)
            del self.pending_predictions[key]

    def selfplay_game(self, mcts: ChessMCTS, temp_threshold: int) -> Promise:
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

        game = chess.pgn.Game()
        node = game

        def move_until_over() -> Promise:
            move_num = len(train_examples)
            temp = int(move_num < temp_threshold)

            def on_get_action_probabilities(pi):
                nonlocal board_wrapper
                nonlocal node

                action_index = np.random.choice(pi.size, p=pi.flatten())
                action_coord = unravel_action_index(action_index)
                train_examples.append(
                    (InputState(board_wrapper.board), action_index, 0)
                )

                move = find_move_from_action_coord(action_coord, board_wrapper.board)
                print("Made a move!", move)

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
                else:
                    return move_until_over()

            return mcts.get_action_probabilities(board_wrapper, temp=temp).then(
                on_get_action_probabilities
            )

        return move_until_over()
