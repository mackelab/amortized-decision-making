from os import path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import sbibm
import torch
from sbi.utils import BoxUniform
from sbi.utils.torchutils import atleast_2d

from loss_cal.actions import Action, CategoricalAction, UniformAction
from loss_cal.costs import RevGaussCost, StepCost_weighted
from loss_cal.tasks.task import BenchmarkTask

_task = sbibm.get_task("gaussian_linear")


class LinGauss(BenchmarkTask):
    def __init__(self, action_type: str, num_actions: int = None, probs: List = None) -> None:
        self.param_low = torch.Tensor([-0.3291] * 10)
        self.param_high = torch.Tensor([0.3291] * 10)  # based on where 99.9% of the mass of the prior are
        param_range = {"low": self.param_low, "high": self.param_high}
        parameter_aggegration = lambda params: torch.mean(params, dim=1).unsqueeze(1)

        if action_type == "discrete":
            self.num_actions = num_actions
            self.probs = probs
            assert num_actions is not None
            actions = CategoricalAction(num_actions=num_actions, probs=probs)
        else:
            self.action_low, self.action_high = 0.0, 100.0
            actions = UniformAction(low=self.action_low, high=self.action_high, dist=BoxUniform)

        super().__init__("gaussian_linear", action_type, actions, param_range, parameter_aggegration)

    ## Extra functions
    def plot_observations(self, rows: int = 2, cols: int = 5):
        n_observations = 10
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True)
        for idx in range(n_observations):
            obs = _task.get_observation(num_observation=idx + 1)
            params = _task.get_true_parameters(num_observation=idx + 1).squeeze()
            axes[idx // cols, idx % cols].plot(obs[0, :10])
            axes[idx // cols, idx % cols].set_title(
                rf"observation {idx}",
                size=10,
            )
        fig.suptitle("observations")
        return fig, axes

    # ## TODO
    # def bayes_optimal_action(
    #     self,
    #     x_o: torch.Tensor,
    #     a_grid: torch.Tensor,
    #     cost_fn: Callable,
    #     lower: float = 0.0,
    #     upper: float = 5.0,
    #     resolution: int = 500,
    # ) -> float:
    #     """Compute the Bayes optimal action under the ground truth posterior

    #     Args:
    #         x_o (torch.Tensor): observation, conditional of the posterior p(theta|x=x_o)
    #         a_grid (torch.Tensor): actions to compute the incurred costs for
    #         lower (float, optional): lower bound the parameter grid/integral. Defaults to 0.0.
    #         upper (float, optional): upper bound of the parameter grid/integral. Defaults to 5.0.
    #         resolution (int, optional): number of evaluation points. Defaults to 500.
    #         cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to RevGaussCost(factor=1).

    #     Returns:
    #         float: action with minimal incurred costs
    #     """
    #     raise NotImplementedError
    #     losses = torch.tensor(
    #         [
    #             self.expected_posterior_costs(
    #                 x=x_o, a=a, lower=lower, upper=upper, resolution=resolution, cost_fn=cost_fn
    #             )
    #             for a in a_grid
    #         ]
    #     )
    #     return a_grid[losses.argmin()]

    # def bayes_optimal_action_binary(
    #     self,
    #     n: int,
    #     param: int,
    #     cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
    #     verbose=False,
    # ) -> float:
    #     """Compute the Bayes optimal action under the ground truth posterior for binary action

    #     Args:
    #         x_o (torch.Tensor): observation, conditional of the posterior p(theta|x=x_o)
    #         a_grid (torch.Tensor): actions to compute the incurred costs for
    #         lower (float, optional): lower bound the parameter grid/integral. Defaults to 0.0.
    #         upper (float, optional): upper bound of the parameter grid/integral. Defaults to 5.0.
    #         resolution (int, optional): number of evaluation points. Defaults to 500.
    #         cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).

    #     Returns:
    #         float: action with minimal incurred costs
    #     """
    #     costs_action0, costs_action1 = self.expected_posterior_costs(
    #         n=n, a=torch.Tensor([[0.0], [1.0]]), param=param, cost_fn=cost_fn, verbose=verbose
    #     )
    #     return (costs_action0 > costs_action1).float()

    # def posterior_ratio_binary(
    #     self,
    #     n: int,
    #     param: int,
    #     cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
    #     verbose=False,
    # ) -> float:
    #     """Compute the posterior ratio: (exp. costs taking action 0)/(exp. costs taking action 0 + exp. costs taking action 1)

    #     Args:
    #         x_o (torch.Tensor): observation, conditional of posterior p(theta|x_o)
    #         lower (float, optional): lower bound of the parameter grid/integral. Defaults to 0.0.
    #         upper (float, optional): upper bound of the parameter grid/inetgral. Defaults to 5.0.
    #         resolution (int, optional): number of evaluation points. Defaults to 500.
    #         cost_fn (Callable, optional): cost function to compute incurred costs.Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).
    #     Returns:
    #         float: posterior ratio
    #     """
    #     int_0, int_1 = self.expected_posterior_costs(
    #         n=n, a=torch.Tensor([[0.0], [1.0]]), param=param, cost_fn=cost_fn, verbose=verbose
    #     )
    #     return int_0 / (int_0 + int_1)
