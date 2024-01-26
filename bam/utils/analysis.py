from functools import partial
from typing import Callable, Tuple, Union

import torch
from sbi.inference.posteriors import DirectPosterior, MCMCPosterior
from sbi.inference import ABC
from sbi.utils.sbiutils import gradient_ascent
from sbi.utils.torchutils import atleast_2d

from bam.bam import FeedforwardNN
from bam.costs import expected_posterior_costs_given_posterior_samples
from bam.tasks.task import Task


def compute_mean_distance(
    observations: torch.Tensor,
    task: Task,
    cost_fn: Callable,
    npe: DirectPosterior,
    nn: FeedforwardNN,
    parameter=None,
    num_samples=1000,
    show_progress_bars=False,
):
    """compute average difference to ground truth costs

    Args:
        observations (torch.Tensor): observations to compute costs for
        task (Task): task object
        cost_fn (Callable): cost function
        npe (DirectPosterior): npe posterior
        nn (FeedforwardNN): neural network
        parameter (int, optional): parameter. Defaults to None.
        num_samples (int, optional): number of samples from NPE. Defaults to 1000.
        show_progress_bars (whether to show progress during sampling, optional): Show progress bar during sampling from NPE. Defaults to False.

    Returns:
        _type_: _description_
    """
    a_grid = torch.arange(task.action_low, task.action_high, 2.0)

    diff_per_obs_nn = []  # obs x actions
    diff_per_obs_npe = []  # obs x actions
    for i, obs in enumerate(observations):
        if task.task_name == "toy_example":
            posterior_costs = task.expected_posterior_costs(
                x=obs, a=a_grid, cost_fn=cost_fn, param=parameter
            ).squeeze()
        else:
            posterior_costs = torch.tensor(
                [
                    task.expected_posterior_costs(
                        x=i + 1, a=a, param=parameter, cost_fn=cost_fn
                    )
                    for a in a_grid
                ]
            )

        npe_samples = npe.sample(
            (num_samples,), x=obs, show_progress_bars=show_progress_bars
        )
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
    task: Task,
    method: str,
    cost_fn: Callable,
    param: int,
    verbose: bool = True,
    nn: FeedforwardNN = None,
    posterior_estimator: Union[DirectPosterior, MCMCPosterior] = None,
    estimator_samples: Union[int, torch.Tensor] = 1000,
    idx=None,
    show_progress_bars=False,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """Wrapper function for expected costs, returns inf if actions are invalid

    Args:
        x (torch.Tensor): observation
        a (torch.Tensor): action
        task (Task): task
        method (str): method used to estimate expected costs, one of ["posterior", "nn", "npe", "nle", "abc"]
        cost_fn (Callable): cost function
        param (int): parameter
        verbose (bool, optional): indicate invalid actions. Defaults to True.
        nn (FeedforwardNN, optional): neural network. Defaults to None.
        posterior_estimator (DirectPosterior| MCMPosterior, optional): estimated posterior. Defaults to None.
        estimator_samples (int, optional): number of samples from estimated posterior. Defaults to 1000.
        idx (int, optional): index for sbibm benchmark task. Defaults to None.
        show_progress_bars (bool, optional): show progress during sampling for the posterior. Defaults to False.
        device (torch.device, optional): device. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        torch.Tensor: expected costs
    """
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
        a = a.reshape(1, -1)

    assert method in ["posterior", "nn", "npe", "nle", "abc"]

    # check if actions are valid
    inside_range = task.actions.is_valid(a)
    a_valid = a[:, inside_range]
    if verbose and not (inside_range).all():
        print(
            "Some actions are invalid, expected costs with be inf for those actions. "
        )

    expected_costs = torch.empty_like(a)
    expected_costs[:, torch.logical_not(inside_range)] = torch.inf

    if method == "posterior":
        if task.task_name == "toy_example":
            expected_costs[:, inside_range] = task.expected_posterior_costs(
                x=x, a=a_valid, cost_fn=cost_fn
            )
        else:
            expected_costs[:, inside_range] = task.expected_posterior_costs(
                x=idx, a=a_valid, param=param, cost_fn=cost_fn
            )

    elif method == "nn":
        assert nn is not None, "Provide trained NN to evaluate costs."
        nn.to(device)
        # turn a into column vector again, repeat x to match the number of actions
        expected_costs[:, inside_range] = nn(x.repeat(a_valid.shape[1], 1), a_valid.T).T
        # expected_costs[:, inside_range] = nn(x, a_valid)
    elif method in ["npe", "nle"]:
        assert (posterior_estimator is not None and type(estimator_samples) == int) or (
            type(estimator_samples) == torch.Tensor
        ), f"Provide trained posterior estimator ({(posterior_estimator is not None and type(estimator_samples) == int)}) or samples ({type(estimator_samples) == torch.Tensor}) to evaluate costs."
        if type(estimator_samples) == torch.Tensor:
            pass  # print("Using provided samples")
        else:
            estimator_samples = posterior_estimator.sample(
                (estimator_samples,), x=x, show_progress_bars=show_progress_bars
            ).to(device)
        expected_costs[
            :, inside_range
        ] = expected_posterior_costs_given_posterior_samples(
            post_samples=task.param_aggregation(estimator_samples).to(device),
            actions=task.actions,
            a=a_valid,
            param=param,
            cost_fn=cost_fn,
            verbose=verbose,
        )
    elif method == "abc":
        # assert type(estimator_samples) == torch.Tensor, "Provide ABC samples."
        if type(estimator_samples) == torch.Tensor:
            print("Using provided ABC samples")
            pass
        else:
            print("ABC Sample from posterior")
            inference = ABC(task.get_simulator(), task.get_prior_dist())
            # num_simulations = ntrain used for other methods
            estimator_samples = inference(
                x, num_simulations=estimator_samples, quantile=100.0 / estimator_samples
            ).to(device)

        expected_costs[
            :, inside_range
        ] = expected_posterior_costs_given_posterior_samples(
            post_samples=task.param_aggregation(estimator_samples).to(device),
            actions=task.actions,
            a=a_valid,
            param=param,
            cost_fn=cost_fn,
            verbose=verbose,
        )

    else:
        raise NotImplementedError

    return expected_costs.flatten()


def reverse_costs(
    x: torch.Tensor,
    a: torch.Tensor,
    task: Task = None,
    method: str = None,
    cost_fn: Callable = None,
    param: int = None,
    verbose: bool = False,
    nn: FeedforwardNN = None,
    posterior_estimator: Union[DirectPosterior, MCMCPosterior] = None,
    estimator_samples: int = 1000,
    idx: int = None,
    show_progress_bars: bool = False,
    max_cost: float = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> torch.Tensor:
    """Compute reverse costs for gradient ascent

    Args:
        x (torch.Tensor): observation
        a (torch.Tensor): action
        task (Task, optional): task. Defaults to None.
        method (str, optional): method used to estimate the expected costs, one of ["posterior", "nn", "npe", "nle", "abc"]. Defaults to None.
        cost_fn (Callable, optional): cost function. Defaults to None.
        param (int, optional): paramter. Defaults to None.
        verbose (bool, optional): indicate invalid actions. Defaults to False.
        nn (FeedforwardNN, optional): neural network. Defaults to None.
        posterior_estimator (DirectPosterior | MCMCPosterior, optional): estimated posterior. Defaults to None.
        estimator_samples (int, optional): samples from estimated posterior. Defaults to 1000.
        idx (int, optional): index for sbibm benchmark tasks. Defaults to None.
        show_progress_bars (bool, optional): indicator whether to show progress. Defaults to False.
        max_cost (float, optional): maximal cost. Defaults to 1.
        device (torch.device, optional): device. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        torch.Tensor: 1 - expected costs
    """
    return max_cost - expected_costs_wrapper(
        x=x,
        a=a,
        task=task,
        method=method,
        cost_fn=cost_fn,
        param=param,
        nn=nn,
        posterior_estimator=posterior_estimator,
        estimator_samples=estimator_samples,
        verbose=verbose,
        idx=idx,
        show_progress_bars=show_progress_bars,
    )


def find_optimal_action(
    x: torch.Tensor,
    task: Task,
    method: str,
    cost_fn: Callable,
    param: int,
    verbose: bool = True,
    nn: FeedforwardNN = None,
    posterior_estimator: Union[DirectPosterior, MCMCPosterior] = None,
    estimator_samples: int = 1000,
    num_initial_actions=100,
    idx: int = None,
    show_progress_bars: bool = False,
    use_grid_search: bool = True,  ## TODO: assumes actions are 1D
    grid_resolution: int = 10_000,
    max_cost: float = 1,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find optimal action with gradient descent

    Args:
        x (torch.Tensor): observation
        task (Task): task
        method (str): method used to estimate the expected costs
        cost_fn (Callable): cost function
        param (int): paramter
        verbose (bool, optional): indicate invalid actions. Defaults to True.
        nn (FeedforwardNN, optional): neural network. Defaults to None.
        posterior_estimator (DirectPosterior| MCMPosterior, optional): estimated posterior. Defaults to None.
        estimator_samples (int, optional): number of samples from posterior estimator to estimate costs. Defaults to 1000.
        num_initial_actions (int, optional): number of initial actions. Defaults to 100.
        idx (int, optional): index in case of sbibm benchmark task to identify the reference observation/posterior. Defaults to None.
        show_progress_bars (bool, optional): indicator whether to show progress. Defaults to False.
        max_cost (float, optional): maximal cost. Defaults to 1.
        device (torch.device, optional): device. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: optimal action, estimated costs, ground truth costs of optimal action
    """
    assert method in ["posterior", "nn", "npe", "nle", "abc"]
    if task.task_name != "toy_example":
        assert (
            idx is not None
        ), "Provide index of observation to compute incurred costs under true posterior."

    initial_actions = task.actions.sample(num_initial_actions).to(device)
    reverse_costs_given_x = partial(
        reverse_costs,
        x.to(device),
        task=task,
        method=method,
        cost_fn=cost_fn,
        param=param,
        nn=nn,
        posterior_estimator=posterior_estimator,
        estimator_samples=estimator_samples,
        verbose=verbose,
        idx=idx,
        show_progress_bars=show_progress_bars,
        max_cost=max_cost,
    )

    # Find best action using grid search
    if use_grid_search:
        print("Using grid search.")
        if task.action_type == "continuous":
            actions_linspace = torch.linspace(
                task.action_low, task.action_high, grid_resolution
            )
            costs_linspace = expected_costs_wrapper(
                x.to(device),
                a=actions_linspace,
                task=task,
                method=method,
                cost_fn=cost_fn,
                param=param,
                nn=nn,
                posterior_estimator=posterior_estimator,
                estimator_samples=estimator_samples,
                verbose=verbose,
                idx=idx,
                show_progress_bars=show_progress_bars,
            )
            best_action = actions_linspace[costs_linspace.argmin()]
            estimated_costs = 1 - costs_linspace.min()
        else:
            raise NotImplementedError

    else:
        print("Using gradient descent.")
        # Find best action using gradient descent
        best_action, estimated_costs = gradient_ascent(
            potential_fn=reverse_costs_given_x,
            inits=initial_actions,
            theta_transform=None,
        )

    # costs under true posterior
    if task.task_name == "toy_example":
        costs_under_posterior = task.expected_posterior_costs(
            x=x, a=best_action, cost_fn=cost_fn
        ).squeeze(1)
    else:
        costs_under_posterior = task.expected_posterior_costs(
            x=idx, a=best_action, param=param, cost_fn=cost_fn
        ).squeeze(1)

    return best_action, 1 - estimated_costs, costs_under_posterior
