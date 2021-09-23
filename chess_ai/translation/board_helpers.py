from typing import List, Tuple
import chess
import numpy as np
from functools import lru_cache
from cachetools import cached, LRUCache

from .InputState import InputState
from .Action import ACTION_PROBS_SHAPE, Action


def get_next_board_hash(board_hash, move):
    return hash((board_hash, move))


def get_board_hash(board: chess.Board):
    board_hash = hash(None)
    for move in board.move_stack:
        board_hash = get_next_board_hash(board_hash, move)
    return board_hash


@cached(cache=LRUCache(maxsize=100000), key=lambda board_hash, _board: board_hash)
def get_legal_actions(_board_hash, board):
    return [Action(move, board.turn) for move in board.legal_moves]


@cached(cache=LRUCache(maxsize=100000), key=lambda board_hash, _board: board_hash)
def get_board_input_tensor(_board_hash, board):
    return InputState(board).to_tensor()


@cached(cache=LRUCache(maxsize=100000), key=lambda board_hash, _board: board_hash)
def generate_actions_mask_and_coords(
    board_hash,
    board,
) -> Tuple[np.ndarray, List[Action]]:
    """
    return a tuple containing an action mask, and a list of valid actions
    The mask is a matrix of the size of the action outputs from the model
    where 1 means the move is valid and 0 means it's invalid
    """
    mask = np.zeros(ACTION_PROBS_SHAPE)
    valid_actions: List[Action] = get_legal_actions(board_hash, board)

    for action in valid_actions:
        mask[action.coords] = 1
    return mask, valid_actions
