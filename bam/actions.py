from abc import abstractmethod
from typing import Callable

import torch
from sbi.utils import BoxUniform
from torch import Tensor

from bam.utils.utils import atleast_2d_col


class Action:
    """action superclass"""

    def __init__(self, dist) -> None:
        self.dist = dist
        pass

    def get_action_distribution(self):
        """get distribution used to sample actions"""
        return self.dist

    def sample(self, n: int) -> Tensor:
        """sample n actions

        Args:
            n (int): number of samples

        Returns:
            torch.Tensor: columb vector of sampled actions
        """
        return atleast_2d_col(self.dist.sample((n,)))

    @abstractmethod
    def is_valid(self, a: Tensor) -> Tensor:
        """check if action is valid

        Args:
            a (torch.Tensor): action

        Returns:
            torch.Tensor: indicator whether action is integer between 0 and number of classes
        """
        raise (NotImplementedError)


class UniformAction(Action):
    """class for continuous action spaces"""

    def __init__(self, low: float, high: float) -> None:
        """create uniform, continuous actions space

        Args:
            low (float): lower boundary
            high (float): upper boundary
        """
        self.low = low
        self.high = high
        self.dist = BoxUniform([low], [high])

    def get_bounds(self):
        """return boundaries"""
        return self.low, self.high

    def is_valid(self, a: Tensor) -> Tensor:
        """check if action is valid

        Args:
            a (torch.Tensor): action

        Returns:
            torch.Tensor: indicator whether action lies within boundaries
        """
        return torch.logical_and(a >= self.low, a <= self.high).flatten()

    def preselect_actions_by_cost(
        self, cost_fn: Callable, theta: Tensor, n_sample: int = 10, n_select: int = 2
    ):
        # randomly select actions and evaluate their cost
        proposals = self.sample(n_sample * theta.shape[0])
        proposals = proposals.reshape(theta.shape[0], n_sample)
        costs = cost_fn(theta, proposals)
        # sort costs and select n_select lower-cost actions
        _, indices = costs.sort(dim=1)
        selected_idx = indices[:, :n_select]
        selected_actions = torch.stack(
            [proposals[i][selected_idx[i]] for i in range(selected_idx.shape[0])]
        )
        selected_costs = torch.stack(
            [costs[i][selected_idx[i]] for i in range(selected_idx.shape[0])]
        )
        return selected_actions, selected_costs


class CategoricalAction(Action):
    """class for discrete action spaces"""

    def __init__(self, num_actions: int, probs: list):
        """create action class

        Args:
            num_actions (int): number of classes
            probs (list): class probabilities
        """
        self.num_actions = num_actions
        if probs == None:
            self.probs = Tensor([1.0 / num_actions] * num_actions)
        else:
            assert (
                len(probs) == num_actions
            ), "Provide class probabilities for every class."
            self.probs = Tensor(probs)
        self.dist = torch.distributions.Categorical(probs=self.probs)

    def is_valid(self, a: Tensor) -> Tensor:
        return torch.logical_and(
            a == a.int(), torch.logical_and(a >= 0, a <= self.num_actions)
        ).flatten()
