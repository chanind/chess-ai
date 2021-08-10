import chess
import torch
from chess_ai.State import State, serialization_to_tensor


def test_serialize_state_to_tensor_dims():
    board = chess.Board()
    state = State(board)
    serialization = state.serialize()

    assert serialization_to_tensor(serialization).shape == torch.Size([901])
