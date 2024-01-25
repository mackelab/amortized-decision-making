import glob
import logging
from os import path
from typing import Optional

import sbibm
import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import posterior_nn
from sbi.utils.sbiutils import seed_all_backends
from sbibm.algorithms.sbi.utils import wrap_posterior, wrap_prior_dist

from bam.tasks import get_task


def train_npe(
    task_name: str,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    neural_net: str = "nsf",
    hidden_features: int = 50,
    automatic_transforms_enabled: bool = False,
    z_score_x: Optional[str] = "independent",
    z_score_theta: Optional[str] = "independent",
    max_num_epochs: Optional[int] = 2**31 - 1,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=0,
    **task_kwargs,
):
    assert (
        task_name
        in ["toy_example", "linear_gaussian", "bvep"] + sbibm.get_available_tasks()
    ), "Task has to be available in sbibm or 'toy_example' or 'bvep'."
    seed_all_backends(seed)
    log = logging.getLogger(__name__)
    log.info(f"Running NPE")

    task = get_task(task_name, **task_kwargs)
    prior = task.get_prior_dist()
    if task_name not in ["linear_gaussian", "toy_example", "bvep"]:
        transforms = task._task._get_transforms(automatic_transforms_enabled)[
            "parameters"
        ]

        if automatic_transforms_enabled:
            prior = wrap_prior_dist(prior, transforms)

    density_estimator_fun = posterior_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )

    inference_method = inference.SNPE_C(
        prior, density_estimator=density_estimator_fun, device=device.type
    )
    proposal = prior

    # Train for one round
    density_estimator = inference_method.append_simulations(
        theta_train, x_train, proposal=proposal
    ).train(max_num_epochs=max_num_epochs)
    posterior = inference_method.build_posterior(density_estimator)

    if task_name not in ["linear_gaussian", "toy_example", "bvep"]:
        posterior_wrapped = wrap_posterior(posterior, transforms)

    return posterior  # , posterior_wrapped


def load_npe(dir: str, nsim: int, flow: str = "nsf"):
    """load a single npe posterior

    Args:
        dir (str): path to npe files
        nsim (int): number of simulations
        flow (str, optional): neural density estimator. Defaults to "nsf".

    Returns:
        _type_: _description_
    """
    return torch.load(path.join(dir, f"{flow}_n{nsim}.pt"))


def load_npes(dir: str, nsim: int = None, flow: str = "nsf"):
    """Load all trained NPEs

    Args:
        dir (str): path to npe files
        nsim (int, optional): number of simulations. Defaults to None.
        flow (str, optional): neural density estimator. Defaults to "nsf".

    Returns:
        List[DirectPosterior, List[int]: list of npes and of dataset sizes
    """
    if nsim == None:
        model_files = glob.glob(path.join(dir, f"{flow}_n*.pt"))
        get_nsim = lambda file: int(file.split(f"{flow}_n")[1].split(".")[0])

        model_files.sort(key=get_nsim)
        print(f"Loading npe trained on ... simulations:")
        npes = []
        nsim = []

        for file in model_files:
            npe = torch.load(file)
            print("- ", get_nsim(file))
            npes.append(npe)
            nsim.append(get_nsim(file))
    else:
        npes = torch.load(path.join(dir, f"{flow}_n{nsim}.pt"))

    return npes, nsim
