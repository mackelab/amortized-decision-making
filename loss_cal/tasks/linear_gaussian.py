from typing import List

import matplotlib.pyplot as plt
import torch
from sbi.utils import BoxUniform

from loss_cal.actions import CategoricalAction, UniformAction
from loss_cal.tasks.task import BenchmarkTask


class LinGauss(BenchmarkTask):
    def __init__(
        self, action_type: str, num_actions: int = None, probs: List = None
    ) -> None:
        self.param_low = torch.Tensor([-0.3291] * 10)
        self.param_high = torch.Tensor(
            [0.3291] * 10
        )  # based on where 99.9% of the mass of the prior are
        param_range = {"low": self.param_low, "high": self.param_high}
        parameter_aggegration = lambda params: torch.mean(params, dim=1).unsqueeze(1)

        if action_type == "discrete":
            self.num_actions = num_actions
            self.probs = probs
            assert num_actions is not None
            actions = CategoricalAction(num_actions=num_actions, probs=probs)
        else:
            self.action_low, self.action_high = 0.0, 100.0
            actions = UniformAction(low=self.action_low, high=self.action_high)

        super().__init__(
            "gaussian_linear", action_type, actions, param_range, parameter_aggegration
        )
        self.task_name = "linear_gaussian"  # rename at end of init

    ## Extra functions
    def plot_observations(self, rows: int = 2, cols: int = 5):
        n_observations = 10
        fig, axes = plt.subplots(
            rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True
        )
        for idx in range(n_observations):
            obs = self._task.get_observation(num_observation=idx + 1)
            params = self._task.get_true_parameters(num_observation=idx + 1).squeeze()
            axes[idx // cols, idx % cols].plot(obs[0, :10])
            axes[idx // cols, idx % cols].set_title(
                rf"observation {idx}",
                size=10,
            )
        fig.suptitle("observations")
        return fig, axes
