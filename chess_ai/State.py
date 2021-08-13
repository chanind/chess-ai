# based on https://github.com/geohot/twitchchess/blob/master/state.py

from typing import Optional, Tuple
import chess
import torch
from torch.nn.functional import one_hot
import numpy as np


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


def serialization_to_tensor(
    serialization: Tuple[np.array, np.array, Optional[int], int]
) -> torch.Tensor:
    (pieces_state, castling_state, en_passant_state, turn, move_number) = serialization
    one_hot_pieces_state = one_hot(
        torch.from_numpy(pieces_state).type(torch.int64), num_classes=13
    ).view((8, 8, -1))
    one_hot_en_passant_state = torch.zeros((8, 8, 1), dtype=torch.float)
    if en_passant_state is not None:
        one_hot_en_passant_state = one_hot(
            torch.tensor(en_passant_state), num_classes=64
        ).view((8, 8, 1))
    castling_state_tensor = torch.from_numpy(castling_state).expand(8, 8, -1)
    bstate = torch.cat(
        [
            one_hot_pieces_state,
            castling_state_tensor,
            one_hot_en_passant_state,
            torch.tensor(turn * 1.0).expand(8, 8, 1),
            torch.tensor(move_number * 1.0).expand(8, 8, 1),
        ],
        dim=-1,
    ).transpose(0, 2)
    return bstate.float()


def get_coords(square) -> Tuple[int, int]:
    x_coord = square % 8
    y_coord = square // 8
    return (x_coord, y_coord)


def func_compare(val1, val2) -> int:
    if val1 == val2:
        return 0
    return 1 if val1 > val2 else -1


def action_tensor_to_chess_move(board: chess.Board, action: torch.Tensor) -> chess.Move:
    best_move = None
    best_move_score = -np.Inf
    for move in board.legal_moves:
        move_coords = process_move_coords(move)
        move_score = action[move_coords].item()
        if best_move_score < move_score:
            best_move = move
            best_move_score = move_score
    return best_move


def process_move_coords(move) -> Tuple[int, int, int]:
    """
    given a move, output the x, y, and move_type for the move
    This represents the coordinate of the output that will be 1, the rest will be 0
    """
    x_coord, y_coord = get_coords(move.from_square)
    to_coords = get_coords(move.to_square)

    key_x = func_compare(to_coords[0], x_coord)
    key_y = func_compare(to_coords[1], y_coord)

    delta_x = to_coords[0] - x_coord
    delta_y = to_coords[1] - y_coord

    is_knight_move = (
        min(abs(delta_x), abs(delta_y)) == 1 and max(abs(delta_x), abs(delta_y)) == 2
    )
    is_underpromotion = move.promotion is not None and move.promotion != chess.QUEEN

    if is_underpromotion:
        # underpromotion values are 64-72
        direction_key = key_x
        underpromotion_value = UNDERPROMOTION_TYPE_MAPPING[move.promotion]
        underpromotion_direction = UNDERPROMOTION_DIRECTIONAL_MAPPING[direction_key]
        return (
            x_coord,
            y_coord,
            underpromotion_direction + (underpromotion_value * 3) + 64,
        )

    elif is_knight_move:
        # knight moves are 56-63
        direction_key = (delta_x, delta_y)
        return (x_coord, y_coord, KNIGHT_DIRECTIONAL_MAPPING[direction_key] + 56)
    else:
        # queen moves are 0-55
        direction_key = (key_x, key_y)
        magnitude = max(abs(delta_x), abs(delta_y))
        return (
            x_coord,
            y_coord,
            QUEEN_DIRECTIONAL_MAPPING[direction_key] + (magnitude - 1) * 8,
        )


class State(object):
    def __init__(self, board=None):
        if board is None:
            self.board = chess.Board()
        else:
            self.board = board

    def key(self):
        return (
            self.board.board_fen(),
            self.board.turn,
            self.board.castling_rights,
            self.board.ep_square,
        )

    def serialize(self) -> Tuple[np.array, np.array, Optional[int], int]:
        assert self.board.is_valid()

        pieces_state_linear = np.zeros(64, np.uint8)
        for i in range(64):
            pp = self.board.piece_at(i)
            if pp is not None:
                pieces_state_linear[i] = {
                    "P": 1,
                    "N": 2,
                    "B": 3,
                    "R": 4,
                    "Q": 5,
                    "K": 6,
                    "p": 7,
                    "n": 8,
                    "b": 9,
                    "r": 10,
                    "q": 11,
                    "k": 12,
                }[pp.symbol()]
        pieces_state = pieces_state_linear.reshape((8, 8))

        castling_state = np.zeros(4, np.uint8)
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling_state[0] = 1
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling_state[1] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling_state[2] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling_state[3] = 1

        en_passant_state = self.board.ep_square

        turn = self.board.turn
        move_number = min(len(self.board.move_stack) / 250.0, 1)
        # bstate = torch.cat([pieces_state, castling_state, en_passant_state, turn])

        # returning these partially processed parts as a tuple to have lower memory usage in dataloader
        return (pieces_state, castling_state, en_passant_state, turn, move_number)
