import logging
from typing import Optional
import sbibm
from sbi import inference as inference
from sbi.utils.get_nn_models import posterior_nn
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
    wrap_simulator_fn,
)
import torch
import matplotlib.pyplot as plt
from os import path

_task = sbibm.get_task("lotka_volterra")


def get_task():
    return _task


def get_prior():
    return _task.get_prior_dist()


def get_simulator():
    return _task.get_simulator()


def posterior_ratio_given_samples(
    posterior_samples: torch.Tensor, treshold: float, costs: list
):
    cost_fn = (posterior_samples > treshold).sum() * costs[0]
    cost_fp = (posterior_samples < treshold).sum() * costs[1]
    return cost_fn / (cost_fn + cost_fp)


def posterior_ratio_given_obs(
    n_obs: int,
    idx_parameter: int,
    treshold: float,
    costs: list,
):
    assert n_obs in range(1, 11)
    posterior_samples = _task.get_reference_posterior_samples(n_obs)[:, idx_parameter]
    cost_fn = (posterior_samples > treshold).sum() * costs[0]
    cost_fp = (posterior_samples < treshold).sum() * costs[1]
    return cost_fn / (cost_fn + cost_fp)


def plot_observations(rows=2, cols=5):
    # TODO: placement of legend
    n_observations = 10
    fig, axes = plt.subplots(
        rows, cols, figsize=(3 * cols, 3 * rows), constrained_layout=True
    )
    for idx in range(n_observations):
        obs = _task.get_observation(num_observation=idx + 1)
        alpha, beta, gamma, delta = _task.get_true_parameters(
            num_observation=idx + 1
        ).squeeze()
        axes[idx // cols, idx % cols].plot(obs[0, :10], label="rabbits")
        axes[idx // cols, idx % cols].plot(obs[0, 10:], label="foxes")
        axes[idx // cols, idx % cols].set_title(
            rf"$\alpha$={alpha:.2f}, $\beta$={beta:.2f}, $\gamma$={gamma:.2f}, $\delta$={delta:.2f}",
            size=10,
        )
    axes[0, 0].legend()
    fig.suptitle("observations")
    plt.show()


def load_data(base_dir="./data"):
    dir = path.join(base_dir, _task.name)
    try:
        theta_train = torch.load(path.join(dir, "theta_train.pt"))
        x_train = torch.load(path.join(dir, "x_train.pt"))
        theta_val = torch.load(path.join(dir, "theta_val.pt"))
        x_val = torch.load(path.join(dir, "x_val.pt"))
        theta_test = torch.load(path.join(dir, "theta_test.pt"))
        x_test = torch.load(path.join(dir, "x_test.pt"))
        print(f"Load data from '{dir}'.")
        return theta_train, x_train, theta_val, x_val, theta_test, x_test
    except FileNotFoundError:
        print("Data not found, check path or generate data first.")


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


def train_npe(
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    neural_net: str = "nsf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    num_atoms: int = 10,
    automatic_transforms_enabled: bool = False,
    z_score_x: bool = True,
    z_score_theta: bool = True,
    max_num_epochs: Optional[int] = None,
):
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    log = logging.getLogger(__name__)
    log.info(f"Running NPE")

    num_simulations = theta_train.shape[0]

    if simulation_batch_size > num_simulations:
        simulation_batch_size = num_simulations
        log.warn("Reduced simulation_batch_size to num_simulations")

    if training_batch_size > num_simulations:
        training_batch_size = num_simulations
        log.warn("Reduced training_batch_size to num_simulations")

    prior = _task.get_prior_dist()
    if observation is None:
        observation = _task.get_observation(num_observation)

    transforms = _task._get_transforms(automatic_transforms_enabled)["parameters"]

    density_estimator_fun = posterior_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )

    inference_method = inference.SNPE_C(prior, density_estimator=density_estimator_fun)
    posteriors = []
    proposal = prior

    # Train for one round
    density_estimator = inference_method.append_simulations(
        theta_train, x_train, proposal=proposal
    ).train(
        num_atoms=num_atoms,
        training_batch_size=training_batch_size,
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        use_combined_loss=False,
        show_train_summary=True,
        max_num_epochs=max_num_epochs,
    )
    posterior = inference_method.build_posterior(
        density_estimator, sample_with_mcmc=False
    )

    posterior = wrap_posterior(posteriors[-1], transforms)

    return posterior
