from typing import Any, Callable

import torch
from sbi.inference.potentials.base_potential import BasePotential
from sbi.utils import BoxUniform
from sbi.utils.sbiutils import gradient_ascent
from sbi.utils.torchutils import atleast_2d

from loss_calibration.sampling import inverse_transform_sampling


class Actions:
    def __init__(self, low: float = 0.0, high: float = 5.0) -> None:
        self.low = low
        self.high = high
        self.dist = BoxUniform([low], [high])

    def get_action_distribution(self):
        return self.dist

    def get_bounds(self):
        return self.low, self.high

    def is_valid(self, a: torch.Tensor):
        return torch.logical_and(a >= self.low, a <= self.high).flatten()

    def sample(self, n: int):
        return self.dist.sample((n,))


def get_action_distribution(low: float = 0.0, high: float = 5.0):
    return BoxUniform([low], [high])


def sample_actions(n: int, low: float = 0.0, high: float = 5.0):
    distribution = get_action_distribution(low, high)
    return distribution.sample((n,))


def preselect_actions(cost_fn: Callable, theta: torch.Tensor, n_sample: int = 10, n_select: int = 2):
    # randomly select actions and evaluate their cost
    proposals = sample_actions(n_sample * theta.shape[0])
    proposals = proposals.reshape(theta.shape[0], n_sample)
    costs = cost_fn(theta, proposals)
    # sort costs and select n_select lower-cost actions
    _, indices = costs.sort(dim=1)
    selected_idx = indices[:, :n_select]
    selected_proposals = torch.stack([proposals[i][selected_idx[i]] for i in range(selected_idx.shape[0])])
    selected_costs = torch.stack([costs[i][selected_idx[i]] for i in range(selected_idx.shape[0])])
    return selected_proposals, selected_costs


def sequential_action_selection(
    n_rounds: int,
    samples_per_round: int,
    proposal: Callable,
):
    # need potential function or acquisition fct?
    pass


class AcquisitionPotential(BasePotential):
    allow_iid_x = False

    def __init__(self, prior, x_o, acquisition_fn, device="cpu"):
        super().__init__(prior, x_o, device=device)
        self.acquisition_fn = acquisition_fn

    def __call__(self, a, track_gradients=True):
        with torch.set_grad_enabled(track_gradients):
            return self.acquisition_fn(self.x_o, a)


class Acquisition:
    def __init__(self, cost_estimator):
        self.cost_estimator = cost_estimator

    def update_estimator(self, cost_estimator):
        self.cost_estimator = cost_estimator

    def check_estimator(self):
        assert self.cost_estimator is not None, "Set estimator first."

    def __call__(self, x: torch.Tensor, a: torch.Tensor):
        ## TODO: here we need the reverse expected costs
        self.check_estimator()

        if x.ndim < 2:
            x = x.reshape(-1, 1)  # NxD
        if a.ndim < 2:
            a = a.reshape(-1, 1)

        # check that a is valid
        # TODO: actions as class ? very example-specific
        a_dist = get_action_distribution()
        lower, upper = a_dist.base_dist.low, a_dist.base_dist.high

        inside_range = torch.logical_and(a >= lower, a <= upper).flatten()
        a_valid = a[inside_range]

        predicted_cost = torch.zeros_like(a)
        predicted_cost[torch.logical_not(inside_range)] = torch.inf
        predicted_cost[inside_range] = self.cost_estimator(x.repeat(a_valid.shape[0], 1), a_valid)

        complementary_cost = 1 - predicted_cost

        return complementary_cost


def choose_action(
    potential_fn: Callable,
):
    # how to choose the action given the posterior?
    # fine-grained grid only works in low-dimensional settings
    # instead: gradient descent on initially sampled actions
    argmax, max = gradient_ascent(
        potential_fn=potential_fn,
        inits=None,
        theta_transform=None,
        num_iter=1000,
        num_to_optimize=100,
        learning_rate=0.01,
    )


class InverseTransformSampler:
    def __init__(
        self,
        potential_fn: Callable,
        # proposal: Any,  # needed ?
        # theta_transform: Optional[TorchTransform] = None,  # needed ?
    ):
        self.potential = potential_fn
        # self.proposal = proposal
        self.lower_bounder, self.upper_bound = 0, 5  # TODO

    def sample(self, num_samples):
        # potential = partial(self.potential, x)  # TODO: more generic!
        num_samples = num_samples[0]
        samples = inverse_transform_sampling(self.potential, num_samples, self.lower_bounder, self.upper_bound)
        return samples
