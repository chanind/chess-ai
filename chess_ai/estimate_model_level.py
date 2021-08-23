import torch
import chess
import chess.engine

from .play_vs_stockfish import play_vs_stockfish
from .ChessModel import ChessModel


def estimate_model_level(
    model: ChessModel,
    device: torch.device,
    rounds_per_level: int = 10,
    stockfish_binary=None,
):
    """
    estimate model level by playing against various levels of stockfish
    Higher is better
    """
    score_per_level = []
    for stockfish_level in range(20):
        for _ in range(rounds_per_level):
            score = 0
            for color in [chess.WHITE, chess.BLACK]:
                result = play_vs_stockfish(
                    model,
                    device,
                    color,
                    stockfish_level,
                    stockfish_binary=stockfish_binary,
                )
                outcome = result.outcome()
                score_delta = 0.5
                if not result.is_stalemate():
                    score_delta = 1 if outcome.winner == color else 0
                score += score_delta
        avg_score = score / rounds_per_level
        score_per_level.append(avg_score)
        if avg_score == 0:
            break
    total_score = sum(
        [(level + 1) * avg_score for level, avg_score in enumerate(score_per_level)]
    )
    return total_score
