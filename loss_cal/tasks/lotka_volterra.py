from os import path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import sbibm
import torch
from sbi.utils import BoxUniform

from loss_cal.actions import Action, CategoricalAction, UniformAction
from loss_cal.tasks.task import BenchmarkTask


## TODO: go through all function and adapt accordingly
class LotkaVolterra(BenchmarkTask):
    def __init__(self, action_type: str, num_actions: int = None, probs: List = None) -> None:
        assert action_type in ["binary", "continuous"]

        self.param_low = torch.Tensor([0.0, 0.0, 0.0, 0.0])
        self.param_high = torch.Tensor(
            [2.8241, 0.1593, 2.8241, 0.1593]
        )  # based on where 99.9% of the mass of the prior are
        param_range = {"low": self.param_low, "high": self.param_high}
        parameter_aggegration = lambda params: params

        if action_type == "binary":
            self.num_actions = num_actions
            self.probs = probs
            assert num_actions is not None
            actions = CategoricalAction(num_actions=num_actions, probs=probs)

        else:
            self.action_low, self.action_high = 0.0, 100.0
            actions = UniformAction(
                low=self.action_low, high=self.action_high, dist=BoxUniform
            )  # percentage of rabbits to shoot

        super().__init__("lotka_volterra", action_type, actions, param_range, parameter_aggegration)

    def rescale(self, action: torch.Tensor, param: int):
        bounds = self.param_range
        if param == None:
            return action * (5.0 - 0.0) / (self.action_high - self.action_low)

            # return action * (bounds["high"][0] - bounds["low"][0]) / (self.action_high - self.action_low)
        else:
            return action * (bounds["high"][param] - bounds["low"][param]) / (self.action_high - self.action_low)

    ## Extra functions
    def plot_observations(self, rows: int = 2, cols: int = 5):
        # TODO: placement of legend
        n_observations = 10
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True)
        for idx in range(n_observations):
            obs = self.get_observation(n=idx + 1)
            axes[idx // cols, idx % cols].plot(obs[0, :10], label="rabbits")
            axes[idx // cols, idx % cols].plot(obs[0, 10:], label="foxes")
            axes[idx // cols, idx % cols].set_title(
                rf" ".join(
                    [
                        rf"{k}={v.item():.3f}"
                        for (k, v) in zip(self.parameter_names, self.get_true_parameters(n=idx + 1).T)
                    ]
                ),
                size=10,
            )
        axes[0, 0].legend()
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
