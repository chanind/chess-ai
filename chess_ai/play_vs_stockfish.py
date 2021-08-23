import torch
import chess
import chess.engine
import argparse

from .State import State, action_tensor_to_chess_move, serialization_to_tensor
from .ChessModel import ChessModel


def play_vs_stockfish(
    model: ChessModel,
    device: torch.device,
    color: bool = chess.WHITE,
    stockfish_kill_level: int = 5,
    stockfish_move_timeout: float = 0.01,
    stockfish_binary="stockfish",
):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_binary)
    engine.configure({"Skill Level": stockfish_kill_level})

    board = chess.Board()

    while not board.is_game_over():
        if board.turn == color:
            input = (
                serialization_to_tensor(State(board).serialize())
                .unsqueeze(0)
                .to(device)
            )
            result = model(input)
            move = action_tensor_to_chess_move(board, result[0])
        else:
            result = engine.play(board, chess.engine.Limit(time=stockfish_move_timeout))
            move = result.move
        board.push(move)

    engine.quit()
    return board


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file")
    parser.add_argument("--skill-level", type=int)
    parser.add_argument("--as-black", action="store_true")
    parser.add_argument("--stockfish-binary", default="stockfish")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessModel()
    model.load_state_dict(
        torch.load(args.model_file, map_location=torch.device(device))
    )
    model.to(device)
    model.eval()

    color = chess.BLACK if args.as_black else chess.WHITE
    print(f"playing stockfish level {args.skill_level}")

    res = play_vs_stockfish(
        model,
        device,
        color,
        stockfish_kill_level=args.skill_level,
        stockfish_binary=args.stockfish_binary,
    )

    print("Done!")
    outcome = res.outcome()
    if res.is_stalemate():
        print("Stalemate!")
    else:
        print("White wins" if outcome.winner == chess.WHITE else "Black wins")
    print(res)
