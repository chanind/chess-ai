import chess
from chess_ai.State import State, serialization_to_tensor, process_move_coords


def test_serialize_state_to_tensor_dims():
    board = chess.Board()
    state = State(board)
    serialization = state.serialize()

    assert serialization_to_tensor(serialization).shape == (20, 8, 8)


def test_process_move_coords():
    # queen move right 1 square
    move = chess.Move(from_square=4, to_square=5)
    assert process_move_coords(move) == (4, 0, 6)

    # queen move left 1 square
    move = chess.Move(from_square=4, to_square=3)
    assert process_move_coords(move) == (4, 0, 0)

    # queen move left 2 square
    move = chess.Move(from_square=4, to_square=2)
    assert process_move_coords(move) == (4, 0, 8)

    # knight move
    move = chess.Move(from_square=4, to_square=21)
    assert process_move_coords(move) == (4, 0, 56)

    # knight reversed
    move = chess.Move(from_square=21, to_square=4)
    assert process_move_coords(move) == (5, 2, 61)

    # normal pawn promotion
    move = chess.Move(from_square=53, to_square=61, promotion=chess.QUEEN)
    assert process_move_coords(move) == (5, 6, 4)

    # pawn forward promotion to knight
    move = chess.Move(from_square=53, to_square=61, promotion=chess.KNIGHT)
    assert process_move_coords(move) == (5, 6, 64)

    # pawn diagonal promotion to bishop
    move = chess.Move(from_square=53, to_square=62, promotion=chess.BISHOP)
    assert process_move_coords(move) == (5, 6, 71)
