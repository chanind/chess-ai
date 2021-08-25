import chess
import torch
import torch.nn as nn


from .translation.InputState import InputState
from .ChessModel import ChessModel
from .translation.model_output_to_chess_move import model_output_to_chess_move, model_output_to_move_distribution
from torch.distributions import Categorical

class ChessPlayer:

    def __init__(self):
        pass

    def make_move(self, board: chess.Board) -> chess.Move:
        pass

    def __str__(self):
        return "A chess player"

    def quit(self):
        pass

class StockfishPlayer(ChessPlayer):

    def __init__(self, stockfish_binary, stockfish_kill_level, move_timeout = 0.01):
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_binary)
        self.kill_level = stockfish_kill_level
        self.engine.configure({"Skill Level": self.kill_level})
        self.move_timeout = move_timeout
        

    def make_move(self, board: chess.Board) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=self.move_timeout))
        return result.move

    def quit(self):
        self.engine.quit()

    def __str__(self):
        return "Stockfish Player Level {}, with timeout = {}".format(self.kill_level, self.move_timeout)


class MinmaxPlayer(ChessPlayer):

    def __init__(self, depth):
        self.depth = depth
        self.center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        self.score_map = None

    def get_score_map(self):
        return {
            "r": -5, "n": -3, "b": -3, "q": -9, "p": -1,
            "R": 5, "N": 3, "B": 3, "Q": 9, "P": 1
        } if self.this_turn == chess.WHITE else {
            "r": 5, "n": 3, "b": 3, "q": 9, "p": 1,
            "R": -5, "N": -3,"B": -3, "Q": -9, "P": -1
        }

    def score_board(self, board):
        score = 0
        for x in str(board).split():
            if x in self.score_map:
                score += 15*self.score_map[x]
        for square in chess.SQUARES:
            mult = 3 if square in self.center_squares else 1
            if board.piece_at(square):
                mult += 3
            score += mult*(len(board.attackers(chess.BLACK, square))-len(board.attackers(chess.WHITE, square)))
        return score

    def minmax_search(self, board, depth, prev = None, maxi = True):
        if board.is_game_over():
            winner = board.outcome().winner
            if winner is None:
                return -200, None
            elif winner == self.this_turn:
                return 1000, None
            else:
                return -1000, None

        if depth == 0:
            return self.score_board(board), None
        else:
            best, next_move = float("-inf") if maxi else float("inf"), None
            moves = reversed([move for move in board.legal_moves])
            cmp = lambda a,b: a>b if maxi else lambda a,b: a<b
            for move in moves:
                board.push_san(board.san(move))
                score, _ = self.minmax_search(board, depth-1, prev = best, maxi = not maxi)
                if cmp(score, best):
                    best, next_move = score, move
                    if prev and cmp(best, prev):
                        board.pop()
                        break
                board.pop()
            return best, next_move

    def make_move(self, board)->chess.Move:
        self.this_turn = board.turn
        self.score_map = self.get_score_map()
        score, move = self.minmax_search(board, self.depth)
        return move

    def __str__(self):
        return "Minmax player with depth {}".format(self.depth)

class AlphaChess(ChessPlayer):

    def __init__(self, chess_model: ChessModel, device):
        self.chess_model = chess_model
        self.chess_model.eval()
        self.device = device
    
    def make_move(self, board: chess.Board)->chess.Move:
        input = InputState(board).to_tensor().unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.chess_model(input)
        move, _ = model_output_to_chess_move(board, result[0])
        return move

class AlphaChessSP(ChessPlayer):
    def __init__(self, chess_model: ChessModel, device):
        self.chess_model = chess_model
        self.chess_model.eval()
        self.device = device
    
    def make_move(self, board: chess.Board)->chess.Move:
        input = InputState(board).to_tensor().unsqueeze(0).to(self.device)
        with torch.no_grad():
            result = self.chess_model(input)
        scores, moves = model_output_to_move_distribution(board, result[0])
        m = Categorical(logits=scores)
        return moves[m.sample()]