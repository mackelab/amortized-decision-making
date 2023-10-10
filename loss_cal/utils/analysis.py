from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from sbi.utils.sbiutils import gradient_ascent
from sbi.utils.torchutils import atleast_2d

from loss_cal.costs import expected_posterior_costs_given_posterior_samples


def compute_mean_distance(
    observations,
    task,
    cost_fn,
    npe,
    nn,
    parameter=None,
    num_samples=1000,
    lower=0,
    upper=5,
    show_progress_bars=False,
):
    a_grid = torch.arange(task.action_low, task.action_high, 2.0)

    diff_per_obs_nn = []  # obs x actions
    diff_per_obs_npe = []  # obs x actions
    for i, obs in enumerate(observations):
        if task.task_name == "toy_example":
            posterior_costs = task.expected_posterior_costs(
                x=obs, a=a_grid, lower=lower, upper=upper, resolution=1000, cost_fn=cost_fn
            ).squeeze()
            # TODO print("lower/upper fixed!!")
        else:
            posterior_costs = torch.tensor(
                [task.expected_posterior_costs(x=i + 1, a=a, param=parameter, cost_fn=cost_fn) for a in a_grid]
            )

        npe_samples = npe.sample((num_samples,), x=obs, show_progress_bars=show_progress_bars)
        npe_costs = expected_posterior_costs_given_posterior_samples(
            post_samples=task.param_aggregation(npe_samples),
            actions=task.actions,
            a=a_grid,
            param=0,
            cost_fn=cost_fn,
        ).squeeze()

        nn_costs = torch.tensor([nn(atleast_2d(obs), atleast_2d(a)) for a in a_grid])

        diff_per_obs_nn.append(posterior_costs - nn_costs)
        diff_per_obs_npe.append(posterior_costs - npe_costs)

    # average over observations
    mse_nn = torch.mean(torch.abs(torch.vstack(diff_per_obs_nn)), dim=0)
    mse_npe = torch.mean(torch.abs(torch.vstack(diff_per_obs_npe)), dim=0)

    return mse_nn, mse_npe


def expected_costs_wrapper(
    x: torch.Tensor,
    a: torch.Tensor,
    task,
    dist: str,
    cost_fn,
    param,
    verbose: bool = True,
    nn=None,
    npe=None,
    npe_samples=1000,
    idx=None,
    show_progress_bars=False,
):
    # make sure tensors are 2D
    if x is not None:
        x = atleast_2d(x)

        assert (
            x.shape[0] == 1 and x.shape[1] == task.dim_data
        ), f"Only one observation can be used to condition the posterior (got shape {x.shape}). Call function sequentially for multiple observations."
    else:
        assert (
            idx is not None
        ), "Provide either an observation or an index referencing the observations provided by sbibm."

    a = atleast_2d(a)
    if a.numel() == a.shape[0]:
        # turn into row vector (works as a is 1D)
        # necessary because gradient ascent assumes a to be a column vector
        # print("Turn a into row vector.")
        a = a.reshape(1, -1)

    assert dist in ["posterior", "nn", "npe"]

    # check if actions are valid
    inside_range = task.actions.is_valid(a)
    a_valid = a[:, inside_range]
    if verbose and not (inside_range).all():
        print("Some actions are invalid, expected costs with be inf for those actions. ")

    expected_costs = torch.empty_like(a)
    expected_costs[:, torch.logical_not(inside_range)] = torch.inf

    if dist == "posterior":
        if task.task_name == "toy_example":
            expected_costs[:, inside_range] = task.expected_posterior_costs(x=x, a=a_valid, cost_fn=cost_fn)
        else:
            expected_costs[:, inside_range] = task.expected_posterior_costs(
                x=idx, a=a_valid, param=param, cost_fn=cost_fn
            )

    elif dist == "nn":
        assert nn is not None, "Provide trained NN to evaluate costs."
        # turn a into column vector again, repeat x to match the number of actions
        expected_costs[:, inside_range] = nn(x.repeat(a_valid.shape[1], 1), a_valid.T).T
        # expected_costs[:, inside_range] = nn(x, a_valid)
    elif dist == "npe":
        assert npe is not None, "Provide trained NPE to evaluate costs."
        npe_samples = npe.sample((npe_samples,), x=x, show_progress_bars=show_progress_bars)
        expected_costs[:, inside_range] = expected_posterior_costs_given_posterior_samples(
            post_samples=task.param_aggregation(npe_samples),
            actions=task.actions,
            a=a_valid,
            param=param,
            cost_fn=cost_fn,
            verbose=verbose,
        )

    return expected_costs.flatten()


def reverse_costs(
    x: torch.Tensor,
    a: torch.Tensor,
    task=None,
    dist: str = None,
    cost_fn=None,
    param=None,
    verbose: bool = False,
    nn=None,
    npe=None,
    npe_samples=1000,
    idx=None,
    show_progress_bars=False,
) -> torch.Tensor:
    return 1 - expected_costs_wrapper(
        x=x,
        a=a,
        task=task,
        dist=dist,
        cost_fn=cost_fn,
        param=param,
        nn=nn,
        npe=npe,
        npe_samples=npe_samples,
        verbose=verbose,
        idx=idx,
        show_progress_bars=show_progress_bars,
    )


def find_optimal_action(
    x,
    task,
    dist: str,
    cost_fn,
    param,
    verbose: bool = True,
    nn=None,
    npe=None,
    npe_samples=1000,
    num_initial_actions=100,
    idx=None,
    show_progress_bars=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert dist in ["posterior", "nn", "npe"]
    if task.task_name != "toy_example":
        assert idx is not None, "Provide index of observation to compute incurred costs under true posterior."

    initial_actions = task.actions.sample(num_initial_actions)
    reverse_costs_given_x = partial(
        reverse_costs,
        x,
        task=task,
        dist=dist,
        cost_fn=cost_fn,
        param=param,
        nn=nn,
        npe=npe,
        npe_samples=npe_samples,
        verbose=verbose,
        idx=idx,
        show_progress_bars=show_progress_bars,
    )

    best_action, estimated_costs = gradient_ascent(
        potential_fn=reverse_costs_given_x, inits=initial_actions, theta_transform=None
    )

    # costs under true posterior
    if task.task_name == "toy_example":
        costs_under_posterior = task.expected_posterior_costs(x=x, a=best_action, cost_fn=cost_fn)
    else:
        costs_under_posterior = task.expected_posterior_costs(x=idx, a=best_action, param=param, cost_fn=cost_fn)

    return best_action, 1 - estimated_costs, costs_under_posterior
