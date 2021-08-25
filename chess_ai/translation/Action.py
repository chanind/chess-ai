from typing import Tuple
import chess
import torch
import numpy as np

from .utils import get_coords, transform_board_index


ACTION_CHANNELS = 73


QUEEN_DIRECTIONAL_MAPPING = {
    (-1, 0): 0,
    (-1, -1): 1,
    (-1, 1): 2,
    (0, -1): 3,
    (0, 1): 4,
    (1, 1): 5,
    (1, 0): 6,
    (1, -1): 7,
}

KNIGHT_DIRECTIONAL_MAPPING = {
    (1, 2): 0,
    (1, -2): 1,
    (2, 1): 2,
    (2, -1): 3,
    (-1, 2): 4,
    (-1, -2): 5,
    (-2, 1): 6,
    (-2, -1): 7,
}

UNDERPROMOTION_TYPE_MAPPING = {
    chess.KNIGHT: 0,
    chess.ROOK: 1,
    chess.BISHOP: 2,
}
UNDERPROMOTION_DIRECTIONAL_MAPPING = {
    0: 0,
    1: 1,
    -1: 2,
}


def func_compare(val1, val2) -> int:
    if val1 == val2:
        return 0
    return 1 if val1 > val2 else -1


def model_output_to_chess_move(board: chess.Board, action: torch.Tensor) -> chess.Move:
    best_move = None
    best_move_score = -np.Inf
    for move in board.legal_moves:
        move_coords = Action(move, board.turn).coords
        move_score = action[move_coords].item()
        if best_move_score < move_score:
            best_move = move
            best_move_score = move_score
    return best_move


class Action:
    coords: Tuple[int, int, int]
    move: chess.Move

    def __init__(self, move: chess.Move, turn: chess.Color):
        is_white = turn == chess.WHITE
        x_coord, y_coord = get_coords(transform_board_index(move.from_square, is_white))

        to_coords = get_coords(transform_board_index(move.to_square, is_white))

        key_x = func_compare(to_coords[0], x_coord)
        key_y = func_compare(to_coords[1], y_coord)

        delta_x = to_coords[0] - x_coord
        delta_y = to_coords[1] - y_coord

        is_knight_move = (
            min(abs(delta_x), abs(delta_y)) == 1
            and max(abs(delta_x), abs(delta_y)) == 2
        )
        is_underpromotion = move.promotion is not None and move.promotion != chess.QUEEN

        if is_underpromotion:
            # underpromotion values are 64-72
            direction_key = key_x
            underpromotion_value = UNDERPROMOTION_TYPE_MAPPING[move.promotion]
            underpromotion_direction = UNDERPROMOTION_DIRECTIONAL_MAPPING[direction_key]
            action_type = underpromotion_direction + (underpromotion_value * 3) + 64

        elif is_knight_move:
            # knight moves are 56-63
            direction_key = (delta_x, delta_y)
            action_type = KNIGHT_DIRECTIONAL_MAPPING[direction_key] + 56
        else:
            # queen moves are 0-55
            direction_key = (key_x, key_y)
            magnitude = max(abs(delta_x), abs(delta_y))
            action_type = QUEEN_DIRECTIONAL_MAPPING[direction_key] + (magnitude - 1) * 8

        self.coords = (action_type, x_coord, y_coord)
        self.move = move

    def to_tensor(self) -> torch.Tensor:
        with torch.no_grad():
            action_tensor = torch.zeros(ACTION_CHANNELS, 8, 8)
            action_tensor[self.coords] = 1.0
        return action_tensor
