from typing import List

import matplotlib.pyplot as plt
import torch

from loss_cal.actions import CategoricalAction, UniformAction
from loss_cal.tasks.task import BenchmarkTask


class SIR(BenchmarkTask):
    def __init__(
        self, action_type: str, num_actions: int = None, probs: List = None
    ) -> None:
        assert action_type in ["discrete", "continuous"]

        self.param_low = torch.Tensor([0.0, 0.0])
        self.param_high = torch.Tensor(
            [1.8754, 0.2319]
        )  # based on where 99.9% of the mass of the prior are (torch.distributions.LogNormal(...).icdf(torch.Tensor([0.999])))
        param_range = {"low": self.param_low, "high": self.param_high}
        parameter_aggegration = lambda params: (params[:, 0:1] / params[:, 1:])

        if action_type == "discrete":
            self.num_actions = num_actions
            self.probs = probs
            assert num_actions is not None
            actions = CategoricalAction(num_actions=num_actions, probs=probs)
        else:
            self.action_low, self.action_high = 0.0, 100.0
            actions = UniformAction(low=self.action_low, high=self.action_high)

        super().__init__(
            "sir", action_type, actions, param_range, parameter_aggegration
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
            axes[idx // cols, idx % cols].plot(obs[0, :10], label="infected")
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
