# based on https://github.com/geohot/twitchchess/blob/master/state.py

from typing import Optional, Tuple
import chess
import torch
from torch.nn.functional import one_hot
import numpy as np


def serialization_to_tensor(
    serialization: Tuple[np.array, np.array, Optional[int], int]
) -> torch.Tensor:
    (pieces_state, castling_state, en_passant_state, turn) = serialization
    one_hot_pieces_state = one_hot(
        torch.from_numpy(pieces_state).type(torch.int64), num_classes=13
    ).view((-1))
    one_hot_en_passant_state = torch.zeros(64, dtype=torch.float)
    if en_passant_state is not None:
        en_passant_state = one_hot(torch.tensor(en_passant_state), num_classes=64)
    castling_state_tensor = torch.from_numpy(castling_state)
    bstate = torch.cat(
        [
            one_hot_pieces_state,
            castling_state_tensor,
            one_hot_en_passant_state,
            torch.tensor(turn * 1.0).unsqueeze(0),
        ]
    )
    return bstate.float()


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

        pieces_state = np.zeros(64, np.uint8)
        for i in range(64):
            pp = self.board.piece_at(i)
            if pp is not None:
                pieces_state[i] = {
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
        # bstate = torch.cat([pieces_state, castling_state, en_passant_state, turn])

        # returning these partially processed parts as a tuple to have lower memory usage in dataloader
        return (pieces_state, castling_state, en_passant_state, turn)
