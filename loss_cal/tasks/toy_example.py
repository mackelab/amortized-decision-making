from typing import Any, Callable, List, Optional, Tuple

import torch
from sbi.utils import BoxUniform
from sbi.utils.torchutils import atleast_2d
from torch import Tensor
from torch.distributions import Normal

from loss_cal.actions import CategoricalAction, UniformAction
from loss_cal.costs import RevGaussCost, StepCost_weighted
from loss_cal.tasks.task import Task


class ToyExample(Task):
    def __init__(
        self,
        action_type: str,
        low: float = 0.0,
        high: float = 5.0,
        num_actions: int = None,
        probs: List = None,
    ) -> None:
        assert action_type in ["binary", "continuous"]

        prior_params = {
            "low": Tensor([low]),
            "high": Tensor([high]),
        }
        self.param_low, self.param_high = prior_params["low"], prior_params["high"]
        param_range = {"low": self.param_low, "high": self.param_high}
        parameter_aggregation = lambda params: params
        prior_dist = BoxUniform(**prior_params)

        self.simulator_mean = lambda theta: 50 + 0.5 * theta * (5 - theta) ** 4
        self.simulator_std = 10.0

        if action_type == "binary":
            self.num_actions = num_actions
            self.probs = probs
            assert num_actions is not None
            actions = CategoricalAction(num_actions=num_actions, probs=probs)

        else:
            self.action_low, self.action_high = 0.0, 100.0
            actions = UniformAction(low=self.action_low, high=self.action_high, dist=BoxUniform)

        super().__init__(
            "toy_example",
            action_type,
            actions,
            dim_data=1,
            dim_parameters=1,
            prior_params=prior_params,
            prior_dist=prior_dist,
            param_range=param_range,
            parameter_aggregation=parameter_aggregation,
            name_display="Toy Example",
        )

    def get_prior(self) -> Callable[..., Any]:
        return self.prior_dist

    def get_simulator(self) -> Callable[..., Any]:
        return self.sample_simulator

    def sample_prior(self, n: int) -> Tensor:
        return self.prior_dist.sample((n,))

    def sample_simulator(self, theta: Tensor) -> Tensor:
        return self.simulator_mean(theta) + self.simulator_std * torch.randn(theta.shape)

    def evaluate_prior(self, theta: Tensor) -> Tensor:
        return self.prior_dist.log_prob(theta)

    def evaluate_likelihood(self, theta: Tensor, x: Tensor) -> Tensor:
        """Evaluate the likelihood p(x|theta)

        Args:
            theta (Tensor): paramter value
            x (Tensor): observation

        Returns:
            Tensor: log prob of the likelihood
        """
        mean = self.simulator_mean(theta)
        noise_dist = Normal(mean, self.simulator_std)
        return noise_dist.log_prob(x)

    def evaluate_joint(self, theta: Tensor, x: Tensor) -> Tensor:
        """Evaluate the log probability of the joint p(theta, x)

        Args:
            theta (Tensor): parameter value
            x (Tensor): observation

        Returns:
            Tensor: log p(theta,x)
        """
        l = self.evaluate_likelihood(theta, x)
        p = self.evaluate_prior(theta).unsqueeze(1)
        return l + p

    def _normalize(self, values: Tensor) -> float:
        """Compute the normalization constant

        Args:
            values (Tensor): set of values to normalize
            lower (float): lower bound
            upper (float): upper bound
            resolution (int): resolution

        Returns:
            float: normalization constant
        """
        resolution = 1000
        sum_val = torch.sum(values)
        return sum_val * (self.param_high - self.param_low) / (resolution - 1)

    def gt_posterior(self, x: Tensor) -> Tensor:
        """Compute the ground truth posterior distribution p(theta|x)

        Args:
            x (Tensor): observation
            lower (float, optional): lower bound for theta. Defaults to 0.0.
            upper (float, optional): upper bound for theta. Defaults to 5.0.
            resolution (int, optional): resolution of the grid. Defaults to 500.

        Returns:
            Tensor: approximation to the ground truth posterior
        """
        assert (
            x.numel() == 1
        ), "Only one observation can be used to condition the posterior. Call function sequentially for multiple observations."

        resolution = 1000
        theta_grid = torch.linspace(self.param_low.item(), self.param_high.item(), resolution).unsqueeze(1)
        joint = self.evaluate_joint(theta_grid, x)  # log prob
        joint_ = torch.exp(joint)
        norm_constant = self._normalize(joint_)
        norm_joint = joint_ / norm_constant
        return norm_joint

    def expected_posterior_costs(
        self, x: Tensor, a: Tensor, cost_fn: Callable[..., Any], param: int = None, verbose=True
    ) -> Callable[..., Any]:
        assert type(x) == Tensor, "Provide a observation as tensor."
        # make sure tensors are 2D
        x = atleast_2d(x)
        a = atleast_2d(a)
        assert (
            x.numel() == 1
        ), "Only one observation can be used to condition the posterior. Call function sequentially for multiple observations."
        # check if actions are valid
        if verbose and not (self.actions.is_valid(a)).all():
            print("Some actions are invalid, expected costs with be inf for those actions. ")

        expected_costs = torch.empty_like(a)
        mask = self.actions.is_valid(a)
        expected_costs[:, torch.logical_not(mask)] = torch.inf

        a_valid = a[:, mask]
        resolution = 1000
        theta_grid = torch.linspace(self.param_low.item(), self.param_high.item(), resolution).unsqueeze(1)
        post = self.gt_posterior(x)

        incurred_costs = cost_fn(theta_grid, a_valid)

        # expected posterior costs
        expected_costs[:, mask] = (post * incurred_costs * (self.param_high - self.param_low) / (resolution - 1)).sum(
            dim=0
        )

        return expected_costs

    def bayes_optimal_action(
        self,
        x_o: Tensor,
        a_grid: Tensor,
        cost_fn: Callable,
        lower: float = 0.0,
        upper: float = 5.0,
        resolution: int = 500,
    ) -> float:
        """Compute the Bayes optimal action under the ground truth posterior

        Args:
            x_o (Tensor): observation, conditional of the posterior p(theta|x=x_o)
            a_grid (Tensor): actions to compute the incurred costs for
            lower (float, optional): lower bound the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/integral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to RevGaussCost(factor=1).

        Returns:
            float: action with minimal incurred costs
        """
        losses = torch.tensor([self.expected_posterior_costs(x=x_o, a=a, lower=lower, cost_fn=cost_fn) for a in a_grid])
        return a_grid[losses.argmin()]

    ## Functions relevant for binary actions only
    def expected_posterior_costs_binary(
        self,
        x_o: Tensor,
        cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
    ) -> Tuple[Tensor, Tensor]:
        """Compute expected costs under the posterior

        Args:
            x_o (Tensor): observation, conditional of posterior p(theta|x_o)
            lower (float, optional): lower bound of the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/inetgral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).

        Returns:
            Tuple[Tensor, Tensor]: costs incurred when taking action 0, costs incurred when taking action 1
        """
        resolution = 1000
        theta_grid = torch.linspace(self.param_low.item(), self.param_high.item(), resolution).unsqueeze(1)
        post = self.gt_posterior(x_o)
        incurred_costs_action0 = cost_fn(theta_grid, 0)  # predicting 0 (FN)
        incurred_costs_action1 = cost_fn(theta_grid, 1)  # predicting 1 (FP)
        # expected posterior costs
        expected_costs_action0 = (
            post * incurred_costs_action0 * (self.param_high.item() - self.param_low.item()) / (resolution - 1)
        ).sum()
        expected_costs_action1 = (
            post * incurred_costs_action1 * (self.param_high.item() - self.param_low.item()) / (resolution - 1)
        ).sum()
        return expected_costs_action0, expected_costs_action1

    def bayes_optimal_action_binary(
        self,
        x_o: Tensor,
        cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
    ) -> float:
        """Compute the Bayes optimal action under the ground truth posterior for binary action

        Args:
            x_o (Tensor): observation, conditional of the posterior p(theta|x=x_o)
            a_grid (Tensor): actions to compute the incurred costs for
            lower (float, optional): lower bound the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/integral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs. Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).

        Returns:
            float: action with minimal incurred costs
        """
        costs_action0, costs_action1 = self.expected_posterior_costs_binary(x_o=x_o, cost_fn=cost_fn)
        return (costs_action0 > costs_action1).float()

    def posterior_ratio_binary(
        self,
        x_o: Tensor,
        cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
    ) -> float:
        """Compute the posterior ratio: (exp. costs taking action 0)/(exp. costs taking action 0 + exp. costs taking action 1)

        Args:
            x_o (Tensor): observation, conditional of posterior p(theta|x_o)
            lower (float, optional): lower bound of the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/inetgral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs.Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).
        Returns:
            float: posterior ratio
        """
        int_0, int_1 = self.expected_posterior_costs_binary(x_o=x_o, cost_fn=cost_fn)
        return int_0 / (int_0 + int_1)

    def posterior_ratio_binary_given_posterior(
        self,
        posterior: Callable,
        x_o: Tensor,
        cost_fn: Callable = StepCost_weighted(weights=[5.0, 1.0], threshold=2.0),
    ):
        """Compute the posterior ratio for a given posterior: (exp. costs taking action 0)/(exp. costs taking action 0 + exp. costs taking action 1)

        Args:
            x_o (Tensor): observation, conditional of posterior p(theta|x_o)
            lower (float, optional): lower bound of the parameter grid/integral. Defaults to 0.0.
            upper (float, optional): upper bound of the parameter grid/inetgral. Defaults to 5.0.
            resolution (int, optional): number of evaluation points. Defaults to 500.
            cost_fn (Callable, optional): cost function to compute incurred costs.Defaults to StepCost_weighted(weights=[5.0, 1.0], threshold=2.0).
        Returns:
            float: posterior ratio
        """
        resolution = 1000
        # evaluate posterior on linspace
        theta_linspace = torch.linspace(self.param_low.item(), self.param_high.item(), resolution).unsqueeze(dim=1)
        posterior_evals = torch.exp(posterior.log_prob(theta_linspace, x=x_o)).unsqueeze(dim=1)
        assert theta_linspace.shape == posterior_evals.shape
        # compute integrals
        int_0 = (
            posterior_evals * cost_fn(theta_linspace, 0) * (self.param_high.item() - self.param_low.item()) / resolution
        ).sum()
        int_1 = (
            posterior_evals * cost_fn(theta_linspace, 1) * (self.param_high.item() - self.param_low.item()) / resolution
        ).sum()
        ratio = int_0 / (int_0 + int_1)
        return ratio
