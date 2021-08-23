import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from .ChessModel import ChessModel
from .data.ChessDataset import ChessDataset
from .estimate_model_level import estimate_model_level


criterion = nn.CrossEntropyLoss()


def train(
    device: torch.device,
    model: ChessModel,
    train_dataset: ChessDataset,
    validation_dataset: ChessDataset = None,
    epochs: int = 100,
    batch_size: int = 256,
    evaluate_after_batch=True,
    stockfish_binary=None,
):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = None
    if validation_dataset:
        validation_loader = torch.utils.data.DataLoader(
            validation_dataset, batch_size=batch_size, shuffle=False
        )
    optimizer = optim.Adam(model.parameters())

    model.to(device)

    for epoch in range(epochs):
        train_loss = 0
        num_train_batches = 0
        with tqdm(
            total=len(train_dataset),
            desc=f"Epoch {epoch + 1}",
            unit="img",
        ) as pbar:
            model.train()
            for (inputs, target) in train_loader:
                batch_size = inputs.shape[0]
                optimizer.zero_grad()
                output = model(inputs.to(device))
                loss = criterion(
                    output.view((batch_size, -1)),
                    # this is hacky, the target shouldn't be one-hot, so this is undoing the one hot encoding
                    # we should just directly output the one-hot pos in the dataloader rather than doing this
                    target.to(device).view((batch_size, -1)).max(dim=1)[1],
                )
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

            if validation_loader:
                model.eval()
                validation_loss = 0
                num_validation_batches = 0
                with torch.no_grad():
                    for (inputs, target) in validation_loader:
                        output = model(inputs.to(device))
                        loss = criterion(
                            output.view((batch_size, -1)),
                            # this is hacky, the target shouldn't be one-hot, so this is undoing the one hot encoding
                            # we should just directly output the one-hot pos in the dataloader rather than doing this
                            target.to(device).view((batch_size, -1)).max(dim=1)[1],
                        )

                        validation_loss += loss.item()

                        pbar.set_postfix(
                            **{
                                "Validation Loss": validation_loss
                                / num_validation_batches,
                            }
                        )
                        pbar.update(inputs.shape[0])


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChessModel()
    train(device, model, max_samples=1000)
    torch.save(model.state_dict(), "chess_value_model.pth")
