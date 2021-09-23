import chess
import numpy as np
from .Action import Action
from .board_helpers import get_legal_actions


class InvalidMoveException(Exception):
    pass


def find_move_from_action_coord(
    action_coord: np.ndarray, board_hash: str, board: chess.Board
) -> chess.Move:
    for action in get_legal_actions(board_hash, board):
        if action.coords == action_coord:
            return action.move
    raise InvalidMoveException(
        f"No legal move found for action coord: {action_coord} and fen: {board.fen()}"
    )
