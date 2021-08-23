import chess
from chess_ai.translation.InputState import InputState


def test_serialize_state_to_tensor_dims():
    board = chess.Board()
    state = InputState(board)
    assert state.to_tensor().shape == (20, 8, 8)


def test_serialization_as_black():
    board = chess.Board()
    original_state = InputState(board)

    board.turn = not board.turn
    inverted_state = InputState(board)

    # queen should be on the left in the original state
    assert original_state.pieces_state[0, 3] == 5

    # queen should be on the right in the inverted state
    assert inverted_state.pieces_state[0, 3] == 6

    # both boards should be identical except for queen / king flipped
    assert inverted_state.pieces_state[0, 0] == original_state.pieces_state[0, 0]
