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
    assert process_move_coords(move) == (6, 4, 0)

    # queen move left 1 square
    move = chess.Move(from_square=4, to_square=3)
    assert process_move_coords(move) == (0, 4, 0)

    # queen move left 2 square
    move = chess.Move(from_square=4, to_square=2)
    assert process_move_coords(move) == (8, 4, 0)

    # knight move
    move = chess.Move(from_square=4, to_square=21)
    assert process_move_coords(move) == (56, 4, 0)

    # knight reversed
    move = chess.Move(from_square=21, to_square=4)
    assert process_move_coords(move) == (61, 5, 2)

    # normal pawn promotion
    move = chess.Move(from_square=53, to_square=61, promotion=chess.QUEEN)
    assert process_move_coords(move) == (4, 5, 6)

    # pawn forward promotion to knight
    move = chess.Move(from_square=53, to_square=61, promotion=chess.KNIGHT)
    assert process_move_coords(move) == (64, 5, 6)

    # pawn diagonal promotion to bishop
    move = chess.Move(from_square=53, to_square=62, promotion=chess.BISHOP)
    assert process_move_coords(move) == (71, 5, 6)


def test_serialization_as_black():
    board = chess.Board()
    original_state = State(board)
    original_serialization = original_state.serialize()

    board.turn = not board.turn
    inverted_state = State(board)
    inverted_serialization = inverted_state.serialize()

    # queen should be on the left in the original state
    assert original_serialization[0][0, 3] == 5

    # queen should be on the right in the inverted state
    assert inverted_serialization[0][0, 3] == 6

    # both boards should be identical except for queen / king flipped
    assert inverted_serialization[0][0, 0] == original_serialization[0][0, 0]
