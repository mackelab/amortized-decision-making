from typing import List

import matplotlib.pyplot as plt
import torch
from sbi.utils import BoxUniform

from bam.actions import CategoricalAction, UniformAction
from bam.tasks.task import BenchmarkTask


class LotkaVolterra(BenchmarkTask):
    def __init__(
        self, action_type: str, num_actions: int = None, probs: List = None
    ) -> None:
        assert action_type in ["binary", "continuous"]

        self.param_low = torch.Tensor([0.0, 0.0, 0.0, 0.0])
        self.param_high = torch.Tensor(
            [4.1376, 0.2334, 4.1376, 0.2334]
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
                low=self.action_low, high=self.action_high
            )  # percentage of rabbits to shoot

        super().__init__(
            "lotka_volterra", action_type, actions, param_range, parameter_aggegration
        )

    def rescale(self, action: torch.Tensor, param: int):
        bounds = self.param_range
        if param == None:
            return action * (5.0 - 0.0) / (self.action_high - self.action_low)

            # return action * (bounds["high"][0] - bounds["low"][0]) / (self.action_high - self.action_low)
        else:
            return (
                action
                * (bounds["high"][param] - bounds["low"][param])
                / (self.action_high - self.action_low)
            )

    ## Extra functions
    def plot_observations(self, rows: int = 2, cols: int = 5):
        # TODO: placement of legend
        n_observations = 10
        fig, axes = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True
        )
        for idx in range(n_observations):
            obs = self.get_observation(n=idx + 1)
            axes[idx // cols, idx % cols].plot(obs[0, :10], label="rabbits")
            axes[idx // cols, idx % cols].plot(obs[0, 10:], label="foxes")
            axes[idx // cols, idx % cols].set_title(
                rf" ".join(
                    [
                        rf"{k}={v.item():.3f}"
                        for (k, v) in zip(
                            self.parameter_names, self.get_true_parameters(n=idx + 1).T
                        )
                    ]
                ),
                size=10,
            )
        axes[0, 0].legend()
        fig.suptitle("observations")
        return fig, axes
