import torch
import torch.nn.functional as F
import chess
import collections
import random
import time
from torch.distributions import Categorical

from .ChessModel import ChessModel
from .chess_players import ChessPlayer, MinmaxPlayer, StockfishPlayer, AlphaChess, AlphaChessSP
from .translation.model_output_to_chess_move import model_output_to_chess_move, model_output_to_move_distribution
from .translation.InputState import InputState
from .estimate_model_level import play_against_others

def train_one_game(model: ChessModel, adversary: ChessPlayer, color = chess.WHITE, device = "cpu"):
    board = chess.Board()
    loss = 0
    while not board.is_game_over():
        if board.turn == color:
            input = InputState(board).to_tensor().unsqueeze(0).to(device)
            result = model(input)
            scores, moves = model_output_to_move_distribution(board, result[0])
            m = Categorical(logits=scores)
            action = m.sample()
            loss -= m.log_prob(action)
            move = moves[action]
        else:
            move = adversary.make_move(board)
        board.push(move)
    outcome = board.outcome()
    r = 0     
    if outcome.winner is not None:
        r = 1 if outcome.winner == chess.WHITE else -1
    return r, loss*r

def train_self_play(
    model: ChessModel,
    device: str,
    adversary_pool = [],
    max_adversaries = 5,
    train_interval = 10,
    change_adversary_interval = 500,
    iterations = 50000,
    lr = 0.0001,
    stockfish: str = "stockfish"):

    optim = torch.optim.SGD(model.parameters(), lr = lr, momentum = 0.9)
    adversary, loss, t0 = None, [], time.time()
    for i in range(iterations):
        if i%change_adversary_interval == 0:
            new_model = ChessModel()
            new_model.load_state_dict(model.state_dict())
            new_model = new_model.to(device)
            adversary_pool.append(AlphaChessSP(new_model, device))
            adversary = random.sample(adversary_pool, 1)[0]
        if i%85==0:
            torch.save(model.state_dict(), "rchess.pth")
            play_against_others(AlphaChessSP(model, device), stockfish=stockfish)
        r, game_loss = train_one_game(model, adversary, color = i%2)
        loss.append(game_loss)
        print("Iteration no {}, r = {}, loss = {}, t = {}".format(i, r, game_loss, time.time() - t0))
        if i%train_interval == 0:
            optim.zero_grad()
            loss_sum = sum(loss)
            loss_sum.backward()
            print("Training, loss_sum = {}".format(loss_sum.item()))
            optim.step()
            loss = []


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessModel()
    model.load_state_dict(
        torch.load("chess.pth", map_location=torch.device(device))
    )
    train_self_play(model, device)
    
        
        
        



