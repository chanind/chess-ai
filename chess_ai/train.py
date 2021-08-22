from chess_ai.estimate_model_level import estimate_model_level
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .State import process_move_coords
from .ChessModel import ChessModel
from .ChessDataset import ChessDataset


criterion = nn.CrossEntropyLoss()


def train(
    device: torch.device,
    model: ChessModel,
    epochs: int = 100,
    batch_size: int = 256,
    max_samples=None,
    evaluate_after_batch=True,
    min_elo=2300,
):
    chess_dataset = ChessDataset(max_samples, min_elo=min_elo)
    train_loader = torch.utils.data.DataLoader(
        chess_dataset, batch_size=batch_size, shuffle=True
    )
    optimizer = optim.Adam(model.parameters())

    model.to(device)

    for epoch in range(epochs):
        all_loss = 0
        num_loss = 0
        with tqdm(
            total=len(chess_dataset),
            desc=f"Epoch {epoch + 1}",
            unit="img",
        ) as pbar:
            model.train()
            for (data, target) in train_loader:
                batch_size = data.shape[0]
                optimizer.zero_grad()
                output = model(data.to(device))
                loss = criterion(
                    output.view((batch_size, -1)),
                    # this is hacky, the target shouldn't be one-hot, so this is undoing the one hot encoding
                    # we should just directly output the one-hot pos in the dataloader rather than doing this
                    target.to(device).view((batch_size, -1)).max(dim=1)[1],
                )
                loss.backward()
                optimizer.step()

                all_loss += loss.item()
                num_loss += 1
                pbar.update(data.shape[0])
                pbar.set_postfix(
                    **{
                        "loss": all_loss / num_loss,
                    }
                )

            if evaluate_after_batch:
                model.eval()
                model_score = estimate_model_level(model, device)
                print(f"model score: {model_score}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessModel()
    train(device, model, max_samples=1000)
    torch.save(model.state_dict(), "chess_value_model.pth")
