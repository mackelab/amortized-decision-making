import torch
import sbibm
from sbibm.algorithms.sbi.utils import (
    wrap_prior_dist,
    wrap_simulator_fn,
)
import matplotlib.pyplot as plt
from os import path
import loss_calibration.utils as utils

_task = sbibm.get_task("sir")


def get_task():
    return _task


def get_prior():
    return _task.get_prior_dist()


def get_simulator():
    return _task.get_simulator()


def posterior_ratio_given_obs(
    n_obs: int,
    idx_parameter: int,
    threshold: float,
    costs: list,
):
    return utils.posterior_ratio_given_obs(
        _task, n_obs, idx_parameter, threshold, costs
    )


def plot_observations(rows=2, cols=5):
    # TODO: placement of legend
    n_observations = 10
    fig, axes = plt.subplots(
        rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True
    )
    for idx in range(n_observations):
        obs = _task.get_observation(num_observation=idx + 1)
        beta, gamma = _task.get_true_parameters(num_observation=idx + 1).squeeze()
        axes[idx // cols, idx % cols].plot(obs[0, :10], label="I")
        axes[idx // cols, idx % cols].set_title(
            rf"$\beta$={beta:.2f}, $\gamma$={gamma:.2f}",
            size=10,
        )
    axes[0, 0].legend()
    fig.suptitle("observations")
    plt.show()


def load_data(base_dir="./data"):
    return utils.load_data(_task.name, base_dir)


def generate_data(
    base_dir="./data",
    num_train_samples=100_000,
    num_test_samples=10_000,
    automatic_transforms_enabled: bool = False,
    save_data=True,
):
    dir = path.join(base_dir, _task.name)

    prior = _task.get_prior_dist()
    simulator = _task.get_simulator()
    transforms = _task._get_transforms(automatic_transforms_enabled)["parameters"]
    if automatic_transforms_enabled:
        prior = wrap_prior_dist(prior, transforms)
        simulator = wrap_simulator_fn(simulator, transforms)

    theta_train = prior.sample((num_train_samples,))
    x_train = simulator(theta_train)
    theta_val = prior.sample((num_test_samples,))
    x_val = simulator(theta_val)
    theta_test = prior.sample((num_test_samples,))
    x_test = simulator(theta_test)

    if save_data:
        torch.save(theta_train, path.join(dir, "theta_train.pt"))
        torch.save(x_train, path.join(dir, "x_train.pt"))
        torch.save(theta_val, path.join(dir, "theta_val.pt"))
        torch.save(x_val, path.join(dir, "x_val.pt"))
        torch.save(theta_test, path.join(dir, "theta_test.pt"))
        torch.save(x_test, path.join(dir, "x_test.pt"))
        print(f"Generated new training, test and vailadation data. Saved at: {dir}")

    return theta_train, x_train, theta_val, x_val, theta_test, x_test
