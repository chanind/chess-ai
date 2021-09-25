# modified from https://github.com/suragnair/alpha-zero-general

import logging
import math
import torch
import numpy as np
from numpy.random import default_rng
from .translation.Action import ACTION_PROBS_SHAPE, unravel_action_index
from .translation.board_helpers import (
    generate_actions_mask_and_coords,
    get_legal_actions,
    get_next_board_hash,
    get_board_input_tensor,
)
from .AsyncPredictDataLoader import AsyncPredictDataLoader

EPS = 1e-8

log = logging.getLogger(__name__)


TOTAL_ACTION_INDICES = np.prod(ACTION_PROBS_SHAPE)

CHESS_DIRICHLET_ALPHA = 0.3  # from the alpha zero paper
DIRICHLET_NOISE_PORTION = 0.25

rng = default_rng()


class AsyncChessMCTS:
    """
    Monte Carlo Tree Search for Chess
    """

    loader: AsyncPredictDataLoader

    num_simulations: int
    cpuct: float

    def __init__(
        self,
        loader: AsyncPredictDataLoader,
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

        self.loadingCoros = {}  # Currently loading Ps values, can await this to block

        self.num_simulations = num_simulations
        self.cpuct = cpuct
        self.device = device
        self.loader = loader

    async def get_action_probabilities(self, board_hash, board, temp=1):
        """
        This function performs num_simulations simulations of MCTS starting from board.
        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for _ in range(self.num_simulations):
            await self.search(board_hash, board, True)

        counts = np.zeros(ACTION_PROBS_SHAPE)
        for action in get_legal_actions(board_hash, board):
            if (board_hash, action.coords) in self.Nsa:
                counts[action.coords] = self.Nsa[(board_hash, action.coords)]

        if temp == 0:
            best_action_indices = np.argwhere(counts.flatten() == np.max(counts))
            chosen_action_index = np.random.choice(best_action_indices.flatten())
            chosen_action_coord = unravel_action_index(chosen_action_index)
            probs = np.zeros(ACTION_PROBS_SHAPE)
            probs[chosen_action_coord] = 1
            return probs

        if temp != 1:
            counts **= 1.0 / temp
        counts_sum = np.sum(counts)
        probs = counts / counts_sum
        return probs

    async def search(self, board_hash, board, include_noise=False):
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

        if board_hash not in self.Es:
            self.Es[board_hash] = None
            if board.is_game_over():
                outcome = board.outcome()
                if outcome.winner is None:
                    self.Es[board_hash] = 0
                else:
                    self.Es[board_hash] = 1 if outcome.winner == board.turn else -1
        if self.Es[board_hash] is not None:
            # terminal node
            return -self.Es[board_hash]

        if board_hash in self.loadingCoros:
            await self.loadingCoros[board_hash]
        if board_hash not in self.Ps:
            # leaf node
            loadingCoro = self.loader.load(
                get_board_input_tensor(board_hash, board).unsqueeze(0).to(self.device)
            )
            self.loadingCoros[board_hash] = loadingCoro
            action_probs_tensor, value_tensor = await loadingCoro
            del self.loadingCoros[
                board_hash
            ]  # remove this from loading coros list so we don't block any other loads

            # TODO: does it make more sense to keep everything in pytorch tensors?
            action_probs = action_probs_tensor[0].detach().cpu().numpy()
            value = value_tensor[0].detach().cpu().numpy()
            valid_actions_mask, valid_actions = generate_actions_mask_and_coords(
                board_hash, board
            )
            valid_action_probs = action_probs * valid_actions_mask

            # add dirichlet noise to action probs
            if include_noise:
                legal_actions = get_legal_actions(board_hash, board)
                alphas = np.ones(len(legal_actions)) * CHESS_DIRICHLET_ALPHA
                raw_noise_values = rng.dirichlet(alphas)
                noise = np.zeros(ACTION_PROBS_SHAPE)
                for index, action in enumerate(legal_actions):
                    noise[action.coords] = raw_noise_values[index]

                valid_actions_sum = np.sum(valid_action_probs)
                if valid_actions_sum == 0:
                    valid_action_probs = noise
                else:
                    scaled_valid_action_probs = valid_action_probs / valid_actions_sum
                    valid_action_probs = (
                        1 - DIRICHLET_NOISE_PORTION
                    ) * scaled_valid_action_probs + DIRICHLET_NOISE_PORTION * noise

            self.Ps[board_hash] = valid_action_probs
            sum_Ps_s = np.sum(self.Ps[board_hash])
            if sum_Ps_s > 0:
                self.Ps[board_hash] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your model architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your model and/or training process.
                # log.error("All valid moves were masked, doing a workaround.")
                self.Ps[board_hash] = valid_action_probs + valid_actions_mask
                self.Ps[board_hash] /= np.sum(self.Ps[board_hash])

            self.Vs[board_hash] = valid_actions
            self.Ns[board_hash] = 0

            return -value

        valid_actions = self.Vs[board_hash]
        best_action_score = -float("inf")
        best_action = None

        # pick the action with the highest upper confidence bound
        for action in valid_actions:
            if (board_hash, action.coords) in self.Qsa:
                qval = self.Qsa[(board_hash, action.coords)]
                action_prob = self.Ps[board_hash][action.coords]
                score = qval + self.cpuct * action_prob * math.sqrt(
                    self.Ns[board_hash]
                ) / (1 + self.Nsa[(board_hash, action.coords)])
            else:
                score = (
                    self.cpuct
                    * self.Ps[board_hash][action.coords]
                    * math.sqrt(self.Ns[board_hash] + EPS)
                )  # Q = 0 ?

            if score > best_action_score:
                best_action_score = score
                best_action = action

        next_board_hash = get_next_board_hash(board_hash, best_action.move)
        board.push(best_action.move)
        value = await self.search(next_board_hash, board)
        board.pop()

        if (board_hash, best_action.coords) in self.Qsa:
            self.Qsa[(board_hash, best_action.coords)] = (
                self.Nsa[(board_hash, best_action.coords)]
                * self.Qsa[(board_hash, best_action.coords)]
                + value
            ) / (self.Nsa[(board_hash, best_action.coords)] + 1)
            self.Nsa[(board_hash, best_action.coords)] += 1

        else:
            self.Qsa[(board_hash, best_action.coords)] = value
            self.Nsa[(board_hash, best_action.coords)] = 1

        self.Ns[board_hash] += 1
        return -value
