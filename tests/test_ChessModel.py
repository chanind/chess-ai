import chess
from chess_ai.translation.InputState import InputState
from chess_ai.ChessModel import ChessModel


def test_ChessModel_output_sizes():
    board = chess.Board()
    input = InputState(board)
    model = ChessModel()
    model.eval()
    action_probs, value = model(input.to_tensor().unsqueeze(0))

    assert action_probs.shape == (1, 73, 8, 8)
    assert value.shape == (1, 1)
