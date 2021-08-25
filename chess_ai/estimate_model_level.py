import torch
import chess
import chess.engine
import collections

from .play_vs_stockfish import play_vs_stockfish
from .ChessModel import ChessModel
from .chess_players import ChessPlayer, MinmaxPlayer, StockfishPlayer, AlphaChess, AlphaChessSP
from torch.distributions import Categorical

def one_game(player1: ChessPlayer, player2: ChessPlayer) -> int:
    board = chess.Board()
    while not board.is_game_over():
        if board.turn:
            move = player1.make_move(board)
        else:
            move = player2.make_move(board)
        board.push(move)
    outcome = board.outcome()      
    if outcome.winner is None:
        return 0
    else:
        return 1 if outcome.winner == chess.WHITE else -1

def play_against_others(player: ChessPlayer):

    adversaries = [

        (StockfishPlayer("stockfish", 0, move_timeout = 0.05), 5),
        (StockfishPlayer("stockfish", 1, move_timeout = 0.05), 5),
        (StockfishPlayer("stockfish", 2, move_timeout = 0.05), 5),
        (StockfishPlayer("stockfish", 3, move_timeout = 0.05), 5),
    ]

    for adversary, n in adversaries:
        result_map = {
            chess.WHITE: {1: "wins", 0: "stales", -1: "defeats"},
            chess.BLACK: {-1: "wins", 0: "stales", 1: "defeats"}
        }
        for color in [chess.WHITE, chess.BLACK]:
            results = collections.defaultdict(int)
            for i in range(n):
                if color:
                    r = one_game(player, adversary)
                else:
                    r = one_game(adversary, player)
                results[result_map[color][r]] += 1
            print("Played {} times with color {} against {}, for a total of:".format(
                n, "white" if color else "black", adversary
            ))
            for k, v in results.items():
                print("{} {}, ".format(v, k), end = " ")
            print("")
            print("+++++++++++++++++++++++++++++++++++++++++++")
        adversary.quit()
    
                
            

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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessModel()
    model.load_state_dict(
        torch.load("chess.pth", map_location=torch.device(device))
    )
    player = AlphaChessSP(model, device)
    play_against_others(player)