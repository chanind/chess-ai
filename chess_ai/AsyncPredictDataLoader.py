from typing import Sequence
from aiodataloader import DataLoader
import torch

from chess_ai.ChessModel import ChessModel


class AsyncPredictDataLoader(DataLoader):
    cache = False

    def __init__(self, model: ChessModel, max_batch_size=None):
        super().__init__(max_batch_size=max_batch_size)
        self.model = model

    async def batch_load_fn(self, inputs: Sequence[torch.Tensor]):
        with torch.no_grad():
            pis, values = self.model.predict(torch.cat(inputs, dim=0))
        return [
            (pis[i, :, :].unsqueeze(0), values[i].unsqueeze(0))
            for i in range(len(inputs))
        ]
