from typing import Optional
import chess
import torch
from torch.nn.functional import one_hot
import numpy as np

from .utils import transform_board_index


PIECES_MAP_WHITE = {
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
}
# # if it's black's turn, invert the board and piece locations
PIECES_MAP_BLACK = {
    "p": 1,
    "n": 2,
    "b": 3,
    "r": 4,
    "q": 5,
    "k": 6,
    "P": 7,
    "N": 8,
    "B": 9,
    "R": 10,
    "Q": 11,
    "K": 12,
}


class InputState(object):
    pieces_state: np.ndarray
    castling_state: np.ndarray
    en_passant_state: Optional[int]
    turn: chess.Color
    move_number_scaled: float

    def __init__(self, board: chess.Board):
        pieces_state_linear = np.zeros(64, np.uint8)
        is_white = board.turn == chess.WHITE
        for i in range(64):
            pos = transform_board_index(i, is_white)
            pp = board.piece_at(i)
            if pp is not None:
                pieces_map = PIECES_MAP_WHITE if is_white else PIECES_MAP_BLACK
                pieces_state_linear[pos] = pieces_map[pp.symbol()]
        pieces_state = pieces_state_linear.reshape((8, 8))

        castling_state = np.zeros(4, np.uint8)
        if board.has_queenside_castling_rights(board.turn):
            castling_state[0] = 1
        if board.has_kingside_castling_rights(board.turn):
            castling_state[1] = 1
        if board.has_queenside_castling_rights(not board.turn):
            castling_state[2] = 1
        if board.has_kingside_castling_rights(not board.turn):
            castling_state[3] = 1

        en_passant_state = None
        if board.ep_square is not None:
            en_passant_state = transform_board_index(board.ep_square, is_white)

        turn = board.turn
        move_number_scaled = min(len(board.move_stack) / 250.0, 1)

        # store these partially processed parts to have lower memory usage in dataloader
        self.pieces_state = pieces_state
        self.castling_state = castling_state
        self.en_passant_state = en_passant_state
        self.turn = turn
        self.move_number_scaled = move_number_scaled

    def to_tensor(self) -> torch.Tensor:
        one_hot_pieces_state = one_hot(
            torch.from_numpy(self.pieces_state).type(torch.int64), num_classes=13
        ).view((8, 8, -1))
        one_hot_en_passant_state = torch.zeros((8, 8, 1), dtype=torch.float)
        if self.en_passant_state is not None:
            one_hot_en_passant_state = one_hot(
                torch.tensor(self.en_passant_state), num_classes=64
            ).view((8, 8, 1))
        castling_state_tensor = torch.from_numpy(self.castling_state).expand(8, 8, -1)
        bstate = torch.cat(
            [
                one_hot_pieces_state,
                castling_state_tensor,
                one_hot_en_passant_state,
                torch.tensor(self.turn * 1.0).expand(8, 8, 1),
                torch.tensor(self.move_number_scaled * 1.0).expand(8, 8, 1),
            ],
            dim=-1,
        ).transpose(0, 2)
        return bstate.float()
