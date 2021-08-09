# based on https://github.com/geohot/twitchchess/blob/master/state.py

import chess
import torch
from torch.nn.functional import one_hot


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

    def serialize(self) -> torch.Tensor:
        assert self.board.is_valid()

        raw_pieces_state = torch.zeros(64, dtype=torch.int64)
        for i in range(64):
            pp = self.board.piece_at(i)
            if pp is not None:
                raw_pieces_state[i] = {
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
        one_hot_pieces_state = one_hot(raw_pieces_state, num_classes=13)
        pieces_state = one_hot_pieces_state.view((-1))

        castling_state = torch.zeros(4, dtype=torch.int64)
        if self.board.has_queenside_castling_rights(chess.WHITE):
            castling_state[0] = 1
        if self.board.has_kingside_castling_rights(chess.WHITE):
            castling_state[1] = 1
        if self.board.has_queenside_castling_rights(chess.BLACK):
            castling_state[2] = 1
        if self.board.has_kingside_castling_rights(chess.BLACK):
            castling_state[3] = 1

        en_passant_state = torch.zeros(64, dtype=torch.int64)
        if self.board.ep_square is not None:
            en_passant_state = one_hot(
                torch.tensor(self.board.ep_square, dtype=torch.int64), num_classes=64
            )

        turn = torch.tensor(self.board.turn * 1.0, dtype=torch.int64).unsqueeze(0)
        bstate = torch.cat([pieces_state, castling_state, en_passant_state, turn])

        return bstate
