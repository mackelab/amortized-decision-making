import torch
from sbi.utils import BoxUniform
from torch.distributions import Normal
from loss_calibration.loss import StepLoss_weighted


def get_prior(low=0.0, high=5.0):
    return BoxUniform([low], [high])


def get_simulator():
    return sample_simulator


def sample_prior(n, low=0.0, high=5.0):
    prior = BoxUniform([low], [high])
    return prior.sample_n(n)


def sample_simulator(theta):
    return 0.5 * theta + 2 + torch.randn(theta.shape) * 0.2


def evaluate_prior(theta, low=0.0, high=5.0):
    prior = BoxUniform([low], [high])
    return prior.log_prob(theta)


def evaluate_likelihood(theta, x):  # p(x|theta)
    mean = 0.5 * theta + 2
    noise_dist = Normal(mean, 0.2)
    return noise_dist.log_prob(x)


def evaluate_joint(theta, x):
    l = evaluate_likelihood(theta, x)
    p = evaluate_prior(theta).unsqueeze(1)
    return l + p


def normalize(values, lower, upper, resolution):
    sum_val = torch.sum(values)
    return sum_val * (upper - lower) / (resolution - 1)


def gt_posterior(x, lower=0.0, upper=5.0, resolution=500):
    theta_grid = torch.linspace(lower, upper, resolution).unsqueeze(1)
    joint = evaluate_joint(theta_grid, x)
    joint_ = torch.exp(joint)
    norm_constant = normalize(joint_, lower, upper, resolution)
    norm_joint = joint_ / norm_constant
    return norm_joint


def evaluate_cost(thetas, costs, threshold):
    cost_fn = StepLoss_weighted(costs, threshold=threshold)
    loss_fn = cost_fn(thetas, 0)
    loss_fp = cost_fn(thetas, 1)
    return loss_fn, loss_fp


def expected_posterior_loss(
    x_o, costs, threshold, lower=0.0, upper=5.0, resolution=500
):
    """Compute expected posterior loss

    Args:
        x_o (torch.Tensor): Conditional of posterior p(theta|x=x_o).
        lower (float, optional): Lower bound of parameter grid. Defaults to 0.0.
        upper (float, optional): Upper bound of parameter grid. Defaults to 5.0.
        resolution (int, optional): Resolution of parameter grid. Defaults to 500.
        costs (list, optional): Cost of misclassification. Defaults to [5.0, 1.0].
        threshold (float, optional): Threshold for binarized decisions. Defaults to 2.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: loss incurred for predicting 0, loss incurred for predicting 1
    """
    theta_grid = torch.linspace(lower, upper, resolution).unsqueeze(1)
    post = gt_posterior(x_o, lower, upper, resolution)
    cost_pred0, cost_pred1 = evaluate_cost(theta_grid, costs=costs, threshold=threshold)
    # expected posterior loss
    loss_pred0 = (post * cost_pred0 * (upper - lower) / (resolution - 1)).sum()
    loss_pred1 = (post * cost_pred1 * (upper - lower) / (resolution - 1)).sum()
    return loss_pred0, loss_pred1


def posterior_ratio(
    x_o,
    costs,
    threshold,
    lower=0.0,
    upper=5.0,
    resolution=500,
):
    int_fn, int_fp = expected_posterior_loss(
        x_o,
        costs=costs,
        threshold=threshold,
        lower=lower,
        upper=upper,
        resolution=resolution,
    )
    return int_fn / (int_fp + int_fn)
