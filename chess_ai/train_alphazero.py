import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

from .ChessModel import ChessModel
from .data.SelfPlayDataset import SelfPlayDataset
from .estimate_model_level import estimate_model_level


loss_pi = nn.CrossEntropyLoss()
loss_value = nn.MSELoss()


def train(
    device: torch.device,
    model: ChessModel,
    epochs: int = 100,
    batch_size: int = 256,
    evaluate_after_batch=True,
    stockfish_binary=None,
    mcts_simulations=50,
    num_workers: int = 2,
    games_per_iteration: int = 100,
    max_recent_training_games=10000,
):
    selfplay_dataset = SelfPlayDataset(
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
            model.train()
            for (inputs, target_pis, target_value) in train_loader:
                batch_size = inputs.shape[0]
                optimizer.zero_grad()
                output_pis, output_value = model(inputs.to(device))
                loss_pi = loss_pi(target_pis, output_pis)
                loss_value = loss_value(target_value, output_value)
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessModel()
    train(device, model)
    torch.save(model.state_dict(), "chess_value_model.pth")
