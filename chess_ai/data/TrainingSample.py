from dataclasses import dataclass

from chess_ai.translation.InputState import InputState
from chess_ai.translation.Action import Action


@dataclass
class TrainingSample:
    action: Action
    input: InputState
