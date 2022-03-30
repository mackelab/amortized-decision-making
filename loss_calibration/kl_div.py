import torch
from torch.distributions import Normal
from typing import List, Tuple

# Note: torch.nn.function implements KL divergence as well.


def kl(p, log_q):
    """Compute reverse KL divergence

    Args:
        p (torch.Tensor): evaluations of p
        log_q (torch.Tensor): evaluations of log q

    Returns:
        torch.Tensor: KL divergence
    """
    assert p.shape == log_q.shape
    return torch.sum(torch.where(p != 0, p * (p.log() - log_q), torch.zeros(1)))


def reverse_kl(log_q, p):
    """Compute reverse KL divergence

    Args:
        log_q (torch.Tensor): evaluations of log q
        p (torch.Tensor): evaluations of p

    Returns:
        torch.Tensor: KL divergence
    """
    assert p.shape == log_q.shape
    return torch.sum(
        torch.where(
            torch.logical_not(log_q.isneginf()),
            log_q.exp() * (log_q - p.log()),
            torch.zeros(1),
        )
    )


def minimize_kl(
    mu: float,
    sigma: float,
    thetas: torch.Tensor,
    px,
    epochs: int = 100,
    kl_mode: str = "forward",
) -> Tuple[float, float, List, List]:
    """Loop to minimize the KL divergence

    Args:
        mu (float): initial mean of Gaussian approximation
        sigma (float): initial standard deviation of Gaussian approximation
        thetas (torch.Tensor): linspace, where to evaluate q and p
        px (function): true distribution, defined as function of theta
        epochs (int): number of epochs to train. Defaults to 100.
        kl_mode (str, optional): Whether to minimize forward or reverse KL. Defaults to "forward".

    Returns:
        Tuple[float, float, List, List]: mean, std. dev. of Gaussian approximation, list of accumulated evaluations, list of accumulated loss values
    """

    # initial approximation
    q_mu = torch.tensor(
        [mu], requires_grad=True
    )  # torch.randint(0, 120, (1,)).float();q_mu.requires_grad=True
    q_sigma = torch.tensor(
        [sigma], requires_grad=True
    )  # torch.randint(0, 60, (1,)).float();q_sigma.requires_grad=True

    opt = torch.optim.Adam([q_mu, q_sigma])

    q_logevals_all = []
    loss_vals = []

    for e in range(epochs):
        opt.zero_grad()
        q = Normal(q_mu, q_sigma)
        q_logevals = q.log_prob(thetas)
        if kl_mode == "reverse":
            loss = reverse_kl(q_logevals, px(thetas))
        else:
            loss = kl(px(thetas), q_logevals)
        loss.backward()
        opt.step()

        q_logevals_all.append(q_logevals.detach().numpy())
        loss_vals.append(loss.detach().numpy())

    return q_mu.item(), q_sigma.item(), q_logevals_all, loss_vals
