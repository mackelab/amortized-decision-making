from typing import Callable, List

import torch
from sbi.utils.torchutils import atleast_2d

from bam.actions import Action


def RevGaussCost(
    parameter_range: List[float],
    action_range: List[float] = None,
    factor: float = 1.0,
    exponential: int = 1,
    max_val: float = 10.0,
    offset: float = 0.0,
    aligned: bool = True,
):
    """Reverse Gaussian cost function

    Args:
        factor (float, optional): Factor to influence the std of the Gaussian. Defaults to 0.75.

    Returns:
        function : cost function cost(theta, action)
    """

    assert (
        isinstance(parameter_range, list) and len(parameter_range) == 2
    ), "provide list with minimum and maximum value for rescaling"
    assert (
        isinstance(action_range, list) and len(action_range) == 2
    ), "provide list with minimum and maximum value for rescaling"

    def cost_fn(
        true_theta: torch.Tensor or float,
        action: torch.Tensor or float,
    ):
        """cost function

        Args:
            true_theta (torch.Tensor or float): true parameter values
            action (torch.Tensor or float): actions

        Returns:
            torch.Tensor: incurred costs
        """
        assert isinstance(true_theta, torch.Tensor) or isinstance(
            action, torch.Tensor
        ), "One of the inputs has to be a torch.Tensor."

        # rescale to be within [0,10]
        rescaled_action = rescale(
            action, action_range[0], action_range[1], new_max=max_val
        )
        rescaled_theta = rescale(
            true_theta, parameter_range[0], parameter_range[1], new_max=max_val
        )
        rescaled_offset = rescale(
            parameter_range[0] + offset,
            parameter_range[0],
            parameter_range[1],
            new_max=max_val,
        )

        if not aligned:
            std = factor / (
                torch.abs(max_val - torch.abs(rescaled_theta - rescaled_offset)) + 0.1
            )
            return 1 - torch.exp(
                -((rescaled_theta - rescaled_action) ** 2) / std**exponential
            )
        else:
            std = factor / (torch.abs(rescaled_theta - rescaled_offset) + 0.1)
            return 1 - torch.exp(
                -((rescaled_theta - rescaled_action) ** 2) / std**exponential
            )

    return cost_fn


def rescale(
    val: torch.Tensor, range_min: torch.Tensor, range_max: torch.Tensor, new_max=10.0
):
    return new_max * (val - range_min) / (range_max - range_min)


def MultiClassStepCost(
    theta_crit: torch.Tensor,
    factors: torch.Tensor,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """0-1 loss for multiple classes"""
    # add an initial value with zero costs to avoid indexing issues (allow theta and a with multiple entries)
    # factors = torch.ones(theta_crit.numel() + 1, theta_crit.numel() + 1)
    theta_crit = torch.concat(
        [torch.Tensor([-(2**32)]), theta_crit, torch.Tensor([(2**32)])]
    ).to(
        device
    )  # extend theta_crit by large negative/positive, not inf

    assert (
        factors.numel() == (theta_crit.numel() - 1) ** 2
    ), f"Provide one cost factor for every combination of actions (true, predicted), got {factors.numel()} == {(theta_crit.numel() - 1) ** 2}"

    def cost_fn(
        true_theta: torch.Tensor or float,
        action: torch.Tensor or float,
    ):
        """cost function

        Args:
            true_theta (torch.Tensor or float): true parameter values
            action (torch.Tensor or float): actions

        Returns:
            torch.Tensor: incurred costs
        """
        assert isinstance(true_theta, torch.Tensor) or isinstance(
            action, torch.Tensor
        ), "One of the inputs has to be a torch.Tensor."
        assert (
            sum(action == a for a in torch.arange(0.0, len(theta_crit) - 1))
            .bool()
            .all()
        ), f"All actions have to be integers between 0 and {len(theta_crit)-1}."
        idx = torch.as_tensor(action, dtype=torch.int64).squeeze()

        costs = torch.zeros(max(true_theta.shape[0], action.shape[0]), 1).to(device)
        print("Costs", costs.shape, (idx != 0).shape)
        print("costs", costs[idx != 0].shape)
        for i in range(1, len(theta_crit)):
            costs[idx != i - 1] += (
                factors[idx[idx != i - 1], i - 1 : i].to(device)
                * torch.logical_and(
                    theta_crit[i - 1] <= true_theta[idx != i - 1],
                    true_theta[idx != i - 1] < theta_crit[i],
                )
            ).float()
        return costs

    return cost_fn


def MultiClass01Cost(
    theta_crit,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """0-1 loss for multiple classes"""
    # add an initial value with zero costs to avoid indexing issues (allow theta and a with multiple entries)
    theta_crit = torch.concat(
        [torch.Tensor([-(2**32)]), theta_crit, torch.Tensor([(2**32)])]
    ).to(
        device
    )  # extend theta_crit by large negative/positive, not inf
    print("theta_crit extended: ", theta_crit)

    def cost_fn(
        true_theta: torch.Tensor or float,
        action: torch.Tensor or float,
    ):
        """cost function

        Args:
            true_theta (torch.Tensor or float): true parameter values
            action (torch.Tensor or float): actions

        Returns:
            torch.Tensor: incurred costs
        """
        assert isinstance(true_theta, torch.Tensor) or isinstance(
            action, torch.Tensor
        ), "One of the inputs has to be a torch.Tensor."

        assert (
            sum(action == a for a in torch.arange(0.0, len(theta_crit))).bool().all()
        ), f"All actions have to be integers between 0 and {len(theta_crit)}."
        idx = torch.as_tensor(action, dtype=torch.int64)

        costs = (true_theta > theta_crit[idx + 1]).float() + (
            true_theta < theta_crit[idx]
        ).float()
        return costs

    return cost_fn


def StepCost_weighted(weights: list, threshold: float):
    """Step function for binary classification

    Args:
        weights (list): costs for each class (weights[0] = cost of FN, weights[1]=cost of FP)
        threshold (float): decision threshold

    Returns:
        function: cost function cost(theta, action)
    """

    assert (
        len(weights) == 2
    ), f"Binary classification, expected 2 weights, got {len(weights)}"

    def cost_fn(true_theta: torch.Tensor or float, action: torch.Tensor or float):
        """step function

        Args:
            true_theta (torch.Tensororfloat): true parameter value
            action (torch.Tensororfloat): action

        Returns:
            torch.Tensor: incurred costs
        """

        if type(action) == float or type(action) == int:
            assert action == 0.0 or action == 1.0, "Decision has to be either 0 or 1"
        else:
            assert torch.logical_or(
                action == 0.0, action == 1.0
            ).all(), (
                "All values have to be either 0 (below threshold) or  1(above treshold)"
            )

        return (
            action
            * (1 - torch.gt(true_theta, threshold).type(torch.float))
            * weights[1]
            + (1 - action)
            * torch.gt(true_theta, threshold).type(torch.float)
            * weights[0]
        )

    return cost_fn


def expected_posterior_costs_given_posterior_samples(
    post_samples: torch.Tensor,
    actions: Action,
    a: torch.Tensor,
    cost_fn: Callable,
    param: int or None,
    verbose: bool = True,
):
    """Compute expected costs under the posterior given x and a,  E_p(theta|x)[C(theta, a)]

    Args:
        x_o (torch.Tensor): observation, conditional of posterior p(theta|x_o)
        a_o (torch.Tensor): action, fixed in cost function C(theta, a_o)
        lower (float, optional): lower bound of the parameter grid/integral. Defaults to 0.0.
        upper (float, optional): upper bound of the parameter grid/inetgral. Defaults to 5.0.
        resolution (int, optional): number of evaluation points. Defaults to 500.
        cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to RevGaussCost(factor=1).

    Returns:
        torch.Tensor: expected costs
    """
    # make sure tensors are 2D
    a = atleast_2d(a)
    if verbose and not (actions.is_valid(a)).all():
        print(
            "Some actions are invalid, expected costs with be inf for those actions. "
        )

    expected_costs = torch.empty_like(a)
    mask = actions.is_valid(a)
    expected_costs[:, torch.logical_not(mask)] = torch.inf

    if param is not None:
        post_samples = post_samples[:, param : param + 1]

    a_valid = a[:, mask]
    incurred_costs = cost_fn(post_samples, a_valid)
    # expected posterior costs
    expected_costs[:, mask] = incurred_costs.mean(dim=0)
    return expected_costs
