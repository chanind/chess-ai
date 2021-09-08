import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import asyncio
import torch.nn.functional as F

from .ChessModel import ChessModel
from .data.SelfPlayDataset import SelfPlayDataset
from .estimate_model_level import estimate_model_level


def criterion_pi(output_log_probs, target_pis):
    return -torch.sum(target_pis * output_log_probs) / target_pis.shape[0]


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
        loop = asyncio.get_event_loop()
        loop.run_until_complete(
            selfplay_dataset.generate_self_play_data(model, batch_size=batch_size)
        )
        loop.close()

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
                    F.log_softmax(output_pis.view((batch_size, -1))),
                    target_pis.view((batch_size, -1)),
                )
                loss_value = criterion_value(target_value, output_value.squeeze(-1))
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
    train_alphazero(device, model)
    torch.save(model.state_dict(), "chess_value_model.pth")
