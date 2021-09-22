# modified from https://github.com/suragnair/alpha-zero-general

from dataclasses import dataclass

# import logging
import math
from typing import Callable, Tuple
import torch
import numpy as np
from promise import Promise
from .translation.Action import ACTION_PROBS_SHAPE, unravel_action_index
from .translation.BoardWrapper import (
    BoardWrapper,
    generate_actions_mask_and_coords,
    get_next_board_wrapper,
    get_board_input_tensor,
)
from .ModelPredictActor import ModelPredictActor, PredictMessage

EPS = 1e-8

# log = logging.getLogger(__name__)


TOTAL_ACTION_INDICES = np.prod(ACTION_PROBS_SHAPE)


class ChessMCTS:
    """
    Monte Carlo Tree Search for Chess
    """

    loader: ModelPredictActor

    num_simulations: int
    cpuct: float

    def __init__(
        self,
        loader: Callable[[torch.Tensor], Promise[Tuple[torch.Tensor, torch.Tensor]]],
        device: torch.device,
        num_simulations: int = 50,
        cpuct: float = 1.0,
    ):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.device = device
        self.loader = loader

    def get_action_probabilities(self, board_wrapper, temp=1) -> Promise:
        """
        This function performs num_simulations simulations of MCTS starting from board.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """

        def post_search_calculate_action_probabilities():
            state_hash = board_wrapper.hash
            counts = np.zeros(ACTION_PROBS_SHAPE)
            for action_index in range(TOTAL_ACTION_INDICES):
                action_coord = unravel_action_index(action_index)
                if (state_hash, action_coord) in self.Nsa:
                    counts[action_coord] = self.Nsa[(state_hash, action_coord)]

            if temp == 0:
                best_action_indices = np.argwhere(counts.flatten() == np.max(counts))
                chosen_action_index = np.random.choice(best_action_indices.flatten())
                chosen_action_coord = unravel_action_index(chosen_action_index)
                probs = np.zeros(ACTION_PROBS_SHAPE)
                probs[chosen_action_coord] = 1
                return probs

            counts **= 1.0 / temp
            counts_sum = np.sum(counts)
            probs = counts / counts_sum
            return probs

        return Promise.all(
            [self.search(board_wrapper) for _ in range(self.num_simulations)]
        ).then(post_search_calculate_action_probabilities)

    def search(self, board_wrapper: BoardWrapper) -> Promise:
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.
        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.
        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.
        Returns:
            v: the negative of the value of the current board
        """

        state_hash = board_wrapper.hash
        board = board_wrapper.board

        if state_hash not in self.Es:
            self.Es[state_hash] = None
            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner is None:
                    self.Es[state_hash] = 0
                else:
                    self.Es[state_hash] = 1 if outcome.winner == board.turn else -1
        if self.Es[state_hash] is not None:
            # terminal node
            return -self.Es[state_hash]

        if state_hash not in self.Ps:
            # leaf node
            input_tensor = (
                get_board_input_tensor(board_wrapper).unsqueeze(0).to(self.device)
            )

            def on_model_res(res):
                print("res received!")
                action_probs_tensor, value_tensor = res

                # TODO: does it make more sense to keep everything in pytorch tensors?
                action_probs = action_probs_tensor[0].detach().cpu().numpy()
                value = value_tensor[0].detach().cpu().numpy()
                valid_actions_mask, valid_actions = generate_actions_mask_and_coords(
                    board_wrapper
                )
                valid_action_probs = action_probs * valid_actions_mask
                self.Ps[state_hash] = valid_action_probs

                sum_Ps_s = np.sum(self.Ps[state_hash])
                if sum_Ps_s > 0:
                    self.Ps[state_hash] /= sum_Ps_s  # renormalize
                else:
                    # if all valid moves were masked make all valid moves equally probable

                    # NB! All valid moves may be masked if either your model architecture is insufficient or you've get overfitting or something else.
                    # If you have got dozens or hundreds of these messages you should pay attention to your model and/or training process.
                    print("All valid moves were masked, doing a workaround.")
                    self.Ps[state_hash] = valid_action_probs + valid_actions_mask
                    self.Ps[state_hash] /= np.sum(self.Ps[state_hash])

                self.Vs[state_hash] = valid_actions
                self.Ns[state_hash] = 0

                return -value

            print("asking for res")
            return self.loader(state_hash, input_tensor).then(on_model_res)

        valid_actions = self.Vs[state_hash]
        best_action_score = -float("inf")
        best_action = None

        # pick the action with the highest upper confidence bound
        for action in valid_actions:
            if (state_hash, action.coords) in self.Qsa:
                qval = self.Qsa[(state_hash, action.coords)]
                action_prob = self.Ps[state_hash][action.coords]
                score = qval + self.cpuct * action_prob * math.sqrt(
                    self.Ns[state_hash]
                ) / (1 + self.Nsa[(state_hash, action.coords)])
            else:
                score = (
                    self.cpuct
                    * self.Ps[state_hash][action.coords]
                    * math.sqrt(self.Ns[state_hash] + EPS)
                )  # Q = 0 ?

            if score > best_action_score:
                best_action_score = score
                best_action = action

        # TODO: this might be slow to copy the whole board, try seeing if we can get away with pushing / popping afterwards in the future
        next_board_wrapper = get_next_board_wrapper(board_wrapper, best_action.move)

        def on_finish_search(value):
            if (state_hash, best_action.coords) in self.Qsa:
                self.Qsa[(state_hash, best_action.coords)] = (
                    self.Nsa[(state_hash, best_action.coords)]
                    * self.Qsa[(state_hash, best_action.coords)]
                    + value
                ) / (self.Nsa[(state_hash, best_action.coords)] + 1)
                self.Nsa[(state_hash, best_action.coords)] += 1

            else:
                self.Qsa[(state_hash, best_action.coords)] = value
                self.Nsa[(state_hash, best_action.coords)] = 1

            self.Ns[state_hash] += 1
            return -value

        return self.search(next_board_wrapper).then(on_finish_search)
