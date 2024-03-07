import glob
import logging
from os import path
from typing import Optional

import sbibm
import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import posterior_nn, likelihood_nn
from sbi.utils.sbiutils import seed_all_backends
from sbibm.algorithms.sbi.utils import wrap_posterior, wrap_prior_dist

from bam.tasks import get_task


# [x] NPE
from sbi.inference import SNPE_C, DirectPosterior

# [ ] NLE
from sbi.inference import SNLE_A, MCMCPosterior, likelihood_estimator_based_potential

# [ ] ABC
# from sbi.inference import ABC

# [ ] NRE
from sbi.inference import SNRE


def train_neural_estimator(
    task_name: str,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    method: str = "npe",
    flow: str = "nsf",
    hidden_features: int = 50,
    automatic_transforms_enabled: bool = False,
    z_score_x: Optional[str] = "independent",
    z_score_theta: Optional[str] = "independent",
    max_num_epochs: Optional[int] = 2**31 - 1,
    num_mcmc_chains=100,  # default is 1
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=0,
    **task_kwargs,
):
    assert method.lower() in ["npe", "nle"]
    seed_all_backends(seed)
    log = logging.getLogger(__name__)
    log.info(f"Running {method.upper()}")

    assert (
        task_name
        in ["toy_example", "linear_gaussian", "bvep"] + sbibm.get_available_tasks()
    ), "Task has to be available in sbibm or 'toy_example' or 'bvep'."
    task = get_task(task_name, **task_kwargs)

    prior = task.get_prior_dist()
    if task_name not in ["linear_gaussian", "toy_example", "bvep"]:
        transforms = task._task._get_transforms(automatic_transforms_enabled)[
            "parameters"
        ]

        if automatic_transforms_enabled:
            prior = wrap_prior_dist(prior, transforms)

    if method.lower() == "npe":
        density_estimator_fun = posterior_nn(
            model=flow.lower(),
            hidden_features=hidden_features,
            z_score_x=z_score_x,
            z_score_theta=z_score_theta,
        )

        inference_method = SNPE_C(
            prior=prior,
            density_estimator=density_estimator_fun,
            device=device.type,
        )
        proposal = prior

        # Train for one round
        posterior_estimator = inference_method.append_simulations(
            theta_train, x_train, proposal=proposal
        ).train(max_num_epochs=max_num_epochs)
        posterior = inference_method.build_posterior(posterior_estimator)

    elif method.lower() == "nle":
        density_estimator_fun = likelihood_nn(
            model=flow.lower(),
            hidden_features=hidden_features,
            z_score_theta=z_score_theta,
            z_score_x=z_score_x,
        )
        inference_method = SNLE_A(
            prior=prior,
            density_estimator=density_estimator_fun,
            device=device.type,
        )
        likelihood_estimator = inference_method.append_simulations(
            theta_train, x_train
        ).train(max_num_epochs=max_num_epochs)

        potential_fn, parameter_transform = likelihood_estimator_based_potential(
            likelihood_estimator, prior, x_o=None
        )

        sampler = "slice_np_vectorized"  # default is 'slice_np'
        posterior = MCMCPosterior(
            potential_fn,
            proposal=prior,
            theta_transform=parameter_transform,
            method=sampler,
            num_chains=num_mcmc_chains,
        )
    else:
        raise NotImplementedError

    return posterior


def load_estimator(dir: str, nsim: int, flow: str = "nsf"):
    """load a single npe posterior

    Args:
        dir (str): path to npe files
        nsim (int): number of simulations
        flow (str, optional): neural density estimator. Defaults to "nsf".

    Returns:
        _type_: _description_
    """
    return torch.load(path.join(dir, f"{flow}_n{nsim}.pt"))


def load_neural_estimators(dir: str, nsim: int = None, flow: str = "nsf"):
    """Load all trained estimators in directory

    Args:
        dir (str): path to trained estimators
        nsim (int, optional): number of simulations. Defaults to None.
        flow (str, optional): neural density estimator. Defaults to "nsf".

    Returns:
        List[DirectPosterior, List[int]: list of estimators and of dataset sizes
    """
    if nsim == None:
        model_files = glob.glob(path.join(dir, f"{flow}_n*.pt"))
        get_nsim = lambda file: int(file.split(f"{flow}_n")[1].split(".")[0])

        model_files.sort(key=get_nsim)
        print(f"Loading estimators trained on ... simulations:")
        estimators = []
        nsim = []

        for file in model_files:
            npe = torch.load(file)
            print("- ", get_nsim(file))
            estimators.append(npe)
            nsim.append(get_nsim(file))
    else:
        print(f"Loading estimator trained on {nsim} simulations:")
        estimators = torch.load(path.join(dir, f"{flow}_n{nsim}.pt"))

    return estimators, nsim
