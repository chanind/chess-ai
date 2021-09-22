from typing import Any, Tuple
from thespian.actors import Actor
from dataclasses import dataclass
import torch

from chess_ai.ChessModel import ChessModel


@dataclass
class InitModelPredictActorMessage:
    model: ChessModel


@dataclass
class PredictMessage:
    input: torch.Tensor
    key: Any


@dataclass
class PredictionResultMessage:
    output: Tuple[torch.Tensor, torch.Tensor]
    key: Any


class ModelPredictActor(Actor):
    """
    Actor to handle efficiently bulk querying the model to effectively make use of the GPU
    NOTE: Must pass InitModelPredictActorMessage before using
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bulk_load_queue = []

    def receiveMessage(self, message, sender):
        if isinstance(message, InitModelPredictActorMessage):
            self.model = message.model
        if isinstance(message, PredictMessage):
            print("Prediction plz")
            if self.model is None:
                raise Exception(
                    "Must first send this actor a InitModelPredictActorMessage to load the current model"
                )
            self.bulk_load_queue.append((message, sender))
            self.send(self.myAddress, "run")
        if message == "run":
            self.batch_load()

    def batch_load(self):
        if len(self.bulk_load_queue) == 0:
            return
        print(f"BULK LOADING: {len(self.bulk_load_queue)}")
        inputs = [data[0].input for data in self.bulk_load_queue]
        keys = [data[0].key for data in self.bulk_load_queue]
        senders = [data[1] for data in self.bulk_load_queue]
        with torch.no_grad():
            pis, values = self.model.predict(torch.cat(inputs, dim=0))
        for i in range(len(inputs)):
            output = (pis[i, :, :].unsqueeze(0), values[i].unsqueeze(0))
            self.send(senders[i], PredictionResultMessage(output=output, key=keys[i]))
        self.bulk_load_queue = []
