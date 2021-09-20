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
            print("INITIALIZED")
            self.model = message.model
        else:
            print("MESSAGE!!!", message)
        if isinstance(message, PredictMessage):
            print("PREDICTING")
            if self.model is None:
                raise Exception(
                    "Must first send this actor a InitModelPredictActorMessage to load the current model"
                )
            self.bulk_load_queue.append((message.input, sender))
            self.send(self.myAddress, "run")
        if message == "run":
            self.batch_load()
        print("MESSAGES PROCESSED")

    def batch_load(self):
        if len(self.bulk_load_queue) == 0:
            return
        print(f"BULK LOADING: {len(self.bulk_load_queue)}")
        inputs = [data[0] for data in self.bulk_load_queue]
        senders = [data[1] for data in self.bulk_load_queue]
        with torch.no_grad():
            pis, values = self.model.predict(torch.cat(inputs, dim=0))
        for i in range(len(inputs)):
            output = (pis[i, :, :].unsqueeze(0), values[i].unsqueeze(0))
            self.send(senders[i], output)
        self.bulk_load_queue = []
