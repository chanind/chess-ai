import chess
import torch
from chess_ai.State import State

def test_serialize_state_dims():
    board = chess.Board()
    state = State(board)
    assert state.serialize().shape == torch.Size([901])
