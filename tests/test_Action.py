import chess

from chess_ai.translation.Action import Action


def test_Action_coords():
    # queen move right 1 square
    move = chess.Move(from_square=4, to_square=5)
    assert Action(move, chess.WHITE).coords == (6, 4, 0)

    # queen move left 1 square
    move = chess.Move(from_square=4, to_square=3)
    assert Action(move, chess.WHITE).coords == (0, 4, 0)

    # queen move left 2 square
    move = chess.Move(from_square=4, to_square=2)
    assert Action(move, chess.WHITE).coords == (8, 4, 0)

    # knight move
    move = chess.Move(from_square=4, to_square=21)
    assert Action(move, chess.WHITE).coords == (56, 4, 0)

    # knight reversed
    move = chess.Move(from_square=21, to_square=4)
    assert Action(move, chess.WHITE).coords == (61, 5, 2)

    # normal pawn promotion
    move = chess.Move(from_square=53, to_square=61, promotion=chess.QUEEN)
    assert Action(move, chess.WHITE).coords == (4, 5, 6)

    # pawn forward promotion to knight
    move = chess.Move(from_square=53, to_square=61, promotion=chess.KNIGHT)
    assert Action(move, chess.WHITE).coords == (64, 5, 6)

    # pawn diagonal promotion to bishop
    move = chess.Move(from_square=53, to_square=62, promotion=chess.BISHOP)
    assert Action(move, chess.WHITE).coords == (71, 5, 6)
