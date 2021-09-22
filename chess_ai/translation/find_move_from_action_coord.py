import chess
import numpy as np
from .Action import Action
from .BoardWrapper import BoardWrapper


class InvalidMoveException(Exception):
    pass


def find_move_from_action_coord(
    action_coord: np.ndarray, board_wrapper: BoardWrapper
) -> chess.Move:
    for action in board_wrapper.legal_actions:
        if action.coords == action_coord:
            return action.move
    raise InvalidMoveException(
        f"No legal move found for action coord: {action_coord} and fen: {board_wrapper.board.fen()}"
    )
