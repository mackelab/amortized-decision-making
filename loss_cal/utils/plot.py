from os import path
from typing import Callable, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

config_file = "loss_cal/utils/.matplotlibrc"


def plot_step_loss(
    cost_fn: Callable,
    low: float,
    high: float,
    threshold: float,
    figsize: Tuple = (4, 2),
    labels: list = [r"$L(\theta, 0)$", r"$L(\theta, 1)$"],
    labels_backgorund: list = [r"$D_1$", r"$D_2$"],
    xlabel: str = r"$\theta$",
    ylabel: str = r"$p(\theta)$",
    resolution: int = 100,
    save: bool = False,
    save_path: str = "results/classifier/concept/",
    plot_config_file=None,
) -> None:
    if plot_config_file is None:
        plot_config_file = config_file
    with mpl.rc_context(fname=plot_config_file):
        thetas = torch.arange(low, high, (high - low) / resolution)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.axvline(x=threshold, color="k", label="threshold")
        ax.axvspan(threshold, high * 1.05, facecolor="r", alpha=0.1, label=labels_backgorund[0])
        ax.axvspan(low * 1.05, threshold, facecolor="y", alpha=0.1, label=labels_backgorund[1])

        plt.plot(thetas, cost_fn(thetas, 0), color="#EFB913", label=labels[0])
        plt.plot(thetas, cost_fn(thetas, 1), color="#EC400E", label=labels[1])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xlim(low, high)
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            fancybox=True,
            shadow=True,
            ncol=6,
        )
        if save:
            try:
                plt.savefig(path.join(save_path, "costs.pdf"))
            except FileNotFoundError:
                print(f"Path '{save_path}' does not exist. Plot not saved. ")
        plt.show()
