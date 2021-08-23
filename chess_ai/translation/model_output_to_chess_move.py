import chess
import numpy as np
import torch

from .Action import Action


def model_output_to_chess_move(
    board: chess.Board, action_probabilities: torch.Tensor
) -> chess.Move:
    """
    Translate model output probabilities to a playable chess move
    """

    best_move = None
    best_move_score = -np.Inf
    for move in board.legal_moves:
        move_coords = Action(move, board.turn).coords
        move_score = action_probabilities[move_coords].item()
        if best_move_score < move_score:
            best_move = move
            best_move_score = move_score
    return best_move
