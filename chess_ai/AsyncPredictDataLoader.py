from typing import Sequence
from promise import Promise
from promise.dataloader import DataLoader
import torch

from chess_ai.ChessModel import ChessModel


class AsyncPredictDataLoader(DataLoader):
    def __init__(self, model: ChessModel):
        super().__init__()
        self.model = model

    def batch_load_fn(self, inputs: Sequence[torch.Tensor]):
        pis, values = self.model.predict(torch.cat(inputs, dim=0))
        return Promise.resolve(
            [
                (pis[i, :, :].unsqueeze(0), values[i].unsqueeze(0))
                for i in range(len(inputs))
            ]
        )
