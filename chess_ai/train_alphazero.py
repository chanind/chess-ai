import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from thespian.actors import ActorSystem
import argparse

from .ChessModel import ChessModel
from .data.SelfPlayDataset import SelfPlayDataset
from .estimate_model_level import estimate_model_level
from .logcfg import logcfg


ActorSystem("multiprocQueueBase", logDefs=logcfg)


# def criterion_pi(output_log_probs, target_pis):
#     return -torch.sum(target_pis * output_log_probs) / target_pis.shape[0]

criterion_pi = nn.CrossEntropyLoss()
criterion_value = nn.MSELoss()


def train_alphazero(
    device: torch.device,
    model: ChessModel,
    epochs: int = 100,
    batch_size: int = 256,
    evaluate_after_batch=True,
    stockfish_binary=None,
    mcts_simulations=5,
    num_workers: int = 2,
    games_per_iteration: int = 10,
    max_recent_training_games=10000,
    model_file: str = "chess_alphazero_model.pth",
):
    selfplay_dataset = SelfPlayDataset(
        device,
        mcts_simulations=mcts_simulations,
        games_per_iteration=games_per_iteration,
        max_recent_training_games=max_recent_training_games,
    )
    optimizer = optim.Adam(model.parameters())

    model.to(device)

    for epoch in range(epochs):
        train_loss = 0
        num_train_batches = 0
        model.eval()
        selfplay_dataset.generate_self_play_data(model)

        train_loader = DataLoader(
            selfplay_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        model.train()
        with tqdm(
            total=len(selfplay_dataset),
            desc=f"Epoch {epoch + 1}",
        ) as pbar:
            for (inputs, target_pis, target_value) in train_loader:
                batch_size = inputs.shape[0]
                optimizer.zero_grad()
                output_pis, output_value = model(inputs.to(device))
                loss_pi = criterion_pi(
                    # F.log_softmax(output_pis.view((batch_size, -1))),
                    # target_pis.to(device).view((batch_size, -1)),
                    output_pis.view((batch_size, -1)),
                    # this is hacky, the target shouldn't be one-hot, so this is undoing the one hot encoding
                    # we should just directly output the one-hot pos in the dataloader rather than doing this
                    target_pis.to(device).view((batch_size, -1)).max(dim=1)[1],
                )
                loss_value = criterion_value(
                    target_value.to(device), output_value.squeeze(-1)
                )
                loss = loss_pi + loss_value
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_train_batches += 1
                pbar.update(inputs.shape[0])
                pbar.set_postfix(
                    **{
                        "loss": train_loss / num_train_batches,
                    }
                )

            if evaluate_after_batch:
                model.eval()
                model_score = estimate_model_level(
                    model, device, stockfish_binary=stockfish_binary
                )
                print(f"model score: {model_score}")

            torch.save(model.state_dict(), model_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default="chess_alphazero_model.pth")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--games-per-iteration", type=int, default=100)
    parser.add_argument("--mcts-simulations", type=int, default=15)
    parser.add_argument("--max-recent-training-games", type=int, default=1000)
    parser.add_argument("--evaluate-after-batch", action="store_true")
    parser.add_argument("--load-model-from-file", action="store_true")
    parser.add_argument("--stockfish-binary", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessModel()
    if args.load_model_from_file:
        model.load_state_dict(
            torch.load(args.model_file, map_location=torch.device(device))
        )
    train_alphazero(
        device,
        model,
        batch_size=args.batch_size,
        mcts_simulations=args.mcts_simulations,
        games_per_iteration=args.games_per_iteration,
        evaluate_after_batch=args.evaluate_after_batch,
        stockfish_binary=args.stockfish_binary,
        max_recent_training_games=args.max_recent_training_games,
        model_file=args.model_file,
    )
