import sbibm
import torch
import matplotlib.pyplot as plt
from os import path

_task = sbibm.get_task("lotka_volterra")


def get_task():
    return _task


def get_prior():
    return _task.get_prior()


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
        return theta_train, x_train, theta_val, x_val, theta_test, x_test
    except FileNotFoundError:
        print("Data not found, check path or generate data first.")


def generate_data(base_dir="./data"):
    dir = path.join(base_dir, _task.name)
    prior = get_prior()
    simulator = get_simulator()
    theta_train = prior(num_samples=100_000)
    x_train = simulator(theta_train)
    theta_val = prior(num_samples=10_000)
    x_val = simulator(theta_val)
    theta_test = prior(num_samples=10_000)
    x_test = simulator(theta_test)
    torch.save(theta_train, path.join(dir, "theta_train.pt"))
    torch.save(x_train, path.join(dir, "x_train.pt"))
    torch.save(theta_val, path.join(dir, "theta_val.pt"))
    torch.save(x_val, path.join(dir, "x_val.pt"))
    torch.save(theta_test, path.join(dir, "theta_test.pt"))
    torch.save(x_test, path.join(dir, "x_test.pt"))
    print(f"Generated new training, test and vailadation data. Saved at: {dir}")
    return theta_train, x_train, theta_val, x_val, theta_test, x_test
