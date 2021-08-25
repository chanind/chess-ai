import chess
import numpy as np
import torch
from typing import Tuple

from .Action import Action


def model_output_to_chess_move(
    board: chess.Board, action_probabilities: torch.Tensor
) -> Tuple[chess.Move, Tuple[int, int, int]]: 
    """
    Translate model output probabilities to a playable chess move
    """

    best_move, best_move_coords = None, None
    best_move_score = -np.Inf
    for move in board.legal_moves:
        move_coords = Action(move, board.turn).coords
        move_score = action_probabilities[move_coords].item()
        if best_move_score < move_score:
            best_move, best_move_coords = move, move_coords
            best_move_score = move_score
    return best_move, best_move_coords

def model_output_to_move_distribution(
    board: chess.Board, action_probabilites: torch.Tensor):

    moves = [move for move in board.legal_moves]
    scores = torch.zeros(len(moves))
    for i, move in enumerate(moves):
        move_coords = Action(move, board.turn).coords
        scores[i] = action_probabilites[move_coords]
    return scores, moves


