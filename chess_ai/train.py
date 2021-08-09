import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .ChessValueModel import ChessValueModel
from .ChessDataset import ChessDataset


def train(
    device: torch.device,
    model: ChessValueModel,
    epochs: int = 100,
    batch_size: int = 256,
    max_samples=None,
):
    chess_dataset = ChessDataset(max_samples)
    train_loader = torch.utils.data.DataLoader(
        chess_dataset, batch_size=batch_size, shuffle=True
    )
    optimizer = optim.Adam(model.parameters())
    floss = nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(epochs):
        all_loss = 0
        num_loss = 0
        with tqdm(
            total=len(chess_dataset),
            desc=f"Epoch {epoch + 1}",
            unit="img",
        ) as pbar:
            for (data, target) in train_loader:
                target = target.unsqueeze(-1)
                data, target = data.to(device), target.to(device)
                data = data.float()
                target = target.float()

                optimizer.zero_grad()
                output = model(data)
                loss = floss(output, target)
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessValueModel()
    train(device, model, max_samples=1000)
    torch.save(model.state_dict(), "chess_value_model.pth")
