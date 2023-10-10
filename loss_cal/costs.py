from typing import Callable

import torch
from sbi.utils.torchutils import atleast_2d

from loss_cal.actions import Action


def get_costs(name: str):
    if name == "reverse_gaussian":
        return RevGaussCost
    if name == "squared":
        return SquaredCost
    if name == "linear":
        return LinearCost_weighted
    if name == "step":
        return StepCost_weighted
    else:
        raise NotImplementedError()


def RevGaussCost(
    parameter_range,
    action_range=100,
    factor: float = 1,
    exponential=1,
    max_val=10.0,
    offset=0,
    aligned=True,
):
    """Reverse Gaussian cost function

    Args:
        factor (float, optional): Factor to influence the std of the Gaussian. Defaults to 0.75.

    Returns:
        function : cost function cost(theta, action)
    """

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

        rescaled_theta = true_theta * max_val / parameter_range
        rescaled_offset = offset * max_val / parameter_range
        rescaled_action = action * max_val / action_range - rescaled_offset

        if aligned:
            return 1 - torch.exp(
                -((rescaled_theta - rescaled_action) ** 2)
                / (factor / torch.abs(rescaled_theta + 0.5) ** exponential) ** 2
            )
        else:
            return 1 - torch.exp(
                -((rescaled_theta - rescaled_action) ** 2)
                / (factor / (max_val - rescaled_theta + 0.5) ** exponential) ** 2
            )

    return cost_fn


def SquaredCost(factor: float = 5):
    """Squared action-dependent cost function (high valued actions will incurr higher costs)

    Args:
        factor (float, optional): Factor to amplify steepness of the parabola. Defaults to 5.

    Returns:
        function : cost function cost(theta, action)
    """

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

        return factor * (action + 1) * (true_theta - action) ** 2

    return cost_fn


def StepCost_weighted(weights: list, threshold: float):
    """Step function for binary classification

    Args:
        weights (list): costs for each class (weights[0] = cost of FN, weights[1]=cost of FP)
        threshold (float): decision threshold

    Returns:
        function: cost function cost(theta, action)
    """

    assert len(weights) == 2, f"Binary classification, expected 2 weights, got {len(weights)}"

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
            ).all(), "All values have to be either 0 (below threshold) or  1(above treshold)"

        return (
            action * (1 - torch.gt(true_theta, threshold).type(torch.float)) * weights[1]
            + (1 - action) * torch.gt(true_theta, threshold).type(torch.float) * weights[0]
        )

    return cost_fn


def LinearCost_weighted(weights: list, threshold: float):
    """Linearly increasing costs for binary classification

    Args:
        weights (list): costs for each class (weights[0] = cost of FN, weights[1]=cost of FP)
        threshold (float): decision threshold

    Returns:
        function: cost function cost(theta, action)
    """

    assert len(weights) == 2, f"Binary classification, expected 2 weights, got {len(weights)}"

    def cost_fn(true_theta: torch.Tensor or float, action: torch.Tensor or float):
        """step function

        Args:
            true_theta (torch.Tensororfloat): true parameter value
            action (torch.Tensororfloat): action

        Returns:
            torch.Tensor: incurred costs
        """
        if type(action) == float or type(action) == int:
            assert action == 0.0 or action == 1.0, "Action has to be either 0 or 1"
        else:
            assert true_theta.shape == action.shape, f"Shapes must match, got {true_theta.shape} and {action.shape}."
            assert torch.logical_or(
                action == 0.0, action == 1.0
            ).all(), "All values have to be either 0 (below threshold) or  1(above treshold)"

        return action * (1 - torch.gt(true_theta, threshold).type(torch.float)) * weights[1] * (
            threshold - true_theta
        ) + (1 - action) * torch.gt(true_theta, threshold).type(torch.float) * weights[0] * (true_theta - threshold)

    return cost_fn


# TODO: check if used
class SigmoidLoss_weighted:
    def __init__(self, weights, threshold) -> None:
        assert len(weights) == 2, f"Binary classification, expected 2 values, got {len(weights)}"
        self.weights = weights
        self.threshold = threshold

    def __call__(self, true_theta, decision, dim=None, slope=100):
        """custom BCE with class weights

        Args:
            theta (torch.Tensor): observed/true parameter values
            decision (torch.Tensor or float): indicates decision: 0 (below threshold) or  1(above treshold)
            threshold (torch.Tensor): threshold for binarized decisons.

        Returns:
            float: incurred loss
        """

        if type(decision) == float or type(decision) == int:
            assert decision == 0.0 or decision == 1.0, "Decision has to be either 0 or 1"
        else:
            assert torch.logical_or(
                decision == 0.0, decision == 1.0
            ).all(), "All values have to be either 0 (below threshold) or  1 (above treshold)"

        return (
            decision * (1 - torch.sigmoid(slope * (true_theta - self.threshold))) * self.weights[1]
            + (1 - decision) * torch.sigmoid(slope * (true_theta - self.threshold)) * self.weights[0]
        )


# TODO: check if used
def BCELoss_weighted(weights: list, threshold: float, cost_fn: str = "step"):
    """weighted BCE

    Args:
        weights (list): costs for each class (weights[0] = cost of FN, weights[1]=cost of FP)
        threshold (float): decision threshold
        cost_fn (str, optional): which cost function to use, one of 'step', 'linear'. Defaults to 'step'.
    """

    def loss(prediction: torch.Tensor, target: torch.Tensor, theta: torch.Tensor):
        """computation of incurred costs

        Args:
            prediction (torch.Tensor): output of the classifier
            target (torch.Tensor): ground truth decision
            theta (torch.Tensor): associated value of theta to compute the cost/weights

        Returns:
            float: loss value
        """
        assert prediction.shape == target.shape == theta.shape, "All arguments should have the same shape."
        assert cost_fn in [
            "step",
            "linear",
        ], "The cost function has to be one of 'step' or 'linear'."

        cost_functions = {
            "step": StepCost_weighted(weights, threshold),
            "linear": LinearCost_weighted(weights, threshold),
        }
        costs = cost_functions[cost_fn]

        prediction = torch.clamp(prediction, min=1e-7, max=1 - 1e-7)
        bce = -target * torch.log(prediction) * costs(theta, 0.0) - (1 - target) * torch.log(1 - prediction) * costs(
            theta, 1.0
        )
        return bce

    return loss


def expected_posterior_costs_given_posterior_samples(
    post_samples: torch.Tensor,
    actions: Action,
    a: torch.Tensor,
    cost_fn: Callable,
    param: int or None,
    verbose=True,
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
        print("Some actions are invalid, expected costs with be inf for those actions. ")

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
