import chess
import numpy as np
from .Action import Action


class InvalidMoveException(Exception):
    pass


def find_move_from_action_coord(
    action_coord: np.ndarray, board: chess.Board
) -> chess.Move:
    for move in board.legal_moves:
        if Action(move, board.turn).coords == action_coord:
            return move
    raise InvalidMoveException(
        f"No legal move found for action coord: {action_coord} and fen: {board.fen()}"
    )
