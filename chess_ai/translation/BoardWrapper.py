from typing import List, Tuple
import chess
import numpy as np
from functools import lru_cache
from cachetools import cached, LRUCache
from cachetools.keys import hashkey

from .InputState import InputState
from .Action import ACTION_PROBS_SHAPE, Action


class BoardWrapper:
    board: chess.Board

    def __init__(self, board: chess.Board):
        self.board = board
        self._hash = None

    @property
    def hash(self):
        if self._hash is None:
            self._hash = hash(tuple(self.board.move_stack))
        return self._hash

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        return self.hash == other.hash


@cached(
    cache=LRUCache(maxsize=10000),
    key=lambda board_wrapper, move: hashkey(board_wrapper.hash, move.uci()),
)
def get_next_board_wrapper(board_wrapper: BoardWrapper, move: chess.Move):
    next_board = board_wrapper.board.copy()
    next_board.push(move)
    return BoardWrapper(next_board)


@lru_cache(maxsize=10000)
def get_board_input_tensor(board_wrapper: BoardWrapper):
    return InputState(board_wrapper.board).to_tensor()


@lru_cache(maxsize=10000)
def generate_actions_mask_and_coords(
    board_wrapper: BoardWrapper,
) -> Tuple[np.ndarray, List[Action]]:
    """
    return a tuple containing an action mask, and a list of valid actions
    The mask is a matrix of the size of the action outputs from the model
    where 1 means the move is valid and 0 means it's invalid
    """
    board = board_wrapper.board
    mask = np.zeros(ACTION_PROBS_SHAPE)
    valid_actions: List[Action] = []

    for move in board.legal_moves:
        action = Action(move, board.turn)
        mask[action.coords] = 1
        valid_actions.append(action)
    return mask, valid_actions
