import argparse
import logging
from os import path
from typing import Optional

import sbibm
import torch
from sbi import inference as inference
from sbi.utils.get_nn_models import posterior_nn
from sbibm.algorithms.sbi.utils import wrap_posterior, wrap_prior_dist

import loss_calibration.toy_example as toy
from loss_calibration.utils import check_base_dir_exists


def train_npe(
    task_name: str,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    neural_net: str = "nsf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    num_atoms: int = 10,
    automatic_transforms_enabled: bool = False,
    z_score_x: Optional[str] = "independent",
    z_score_theta: Optional[str] = "independent",
    max_num_epochs: Optional[int] = 2**31 - 1,
):

    assert (
        task_name in ["toy_example"] + sbibm.get_available_tasks()
    ), "Task has to be available in sbibm or 'toy_example'."
    log = logging.getLogger(__name__)
    log.info(f"Running NPE")

    num_simulations = theta_train.shape[0]

    if simulation_batch_size > num_simulations:
        simulation_batch_size = num_simulations
        log.warn("Reduced simulation_batch_size to num_simulations")

    if training_batch_size > num_simulations:
        training_batch_size = num_simulations
        log.warn("Reduced training_batch_size to num_simulations")

    if task_name == "toy_example":
        prior = toy.get_prior()
    else:
        task = sbibm.get_task(task_name)
        prior = task.get_prior_dist()

        transforms = task._get_transforms(automatic_transforms_enabled)["parameters"]

        if automatic_transforms_enabled:
            prior = wrap_prior_dist(prior, transforms)

    density_estimator_fun = posterior_nn(
        model=neural_net.lower(),
        hidden_features=hidden_features,
        z_score_x=z_score_x,
        z_score_theta=z_score_theta,
    )

    inference_method = inference.SNPE_C(prior, density_estimator=density_estimator_fun)
    proposal = prior

    # Train for one round
    density_estimator = inference_method.append_simulations(
        theta_train, x_train, proposal=proposal
    ).train(
        num_atoms=num_atoms,
        training_batch_size=training_batch_size,
        retrain_from_scratch=False,
        discard_prior_samples=False,
        use_combined_loss=False,
        show_train_summary=True,
        max_num_epochs=max_num_epochs,
    )
    posterior = inference_method.build_posterior(density_estimator)

    if task_name != "toy_example":
        posterior_wrapped = wrap_posterior(posterior, transforms)

    return posterior  # , posterior_wrapped


def main(args):
    estimator = args.estimator
    assert estimator in [
        "nsf",
        "maf",
    ], "Density estimator has to be either 'nsf' or 'maf'."

    task_name = args.task
    assert task_name in [
        "toy_example",
        "sir",
        "lotka_volterra",
    ], "Choose one of 'toy_example', 'sir' or 'lotka_volterra'."

    ntrain = args.ntrain

    data_dir = path.join("../data/", task_name)
    theta_train = torch.load(path.join(data_dir, "theta_train.pt"))
    x_train = torch.load(path.join(data_dir, "x_train.pt"))
    if ntrain > theta_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif ntrain < theta_train.shape[0]:
        theta_train = theta_train[:ntrain]
        x_train = x_train[:ntrain]

    base_dir = f"../results/{task_name}/npe/"
    check_base_dir_exists(base_dir)

    print(
        f"Training posterior with {args.ntrain} simulations: \ndensity estimator: {estimator}\ndata_dir: {data_dir}\nsave_dir: ./results/{task_name}\n"
    )

    npe_posterior = train_npe(task_name, theta_train, x_train, max_num_epochs=2)
    torch.save(npe_posterior, path.join(base_dir, f"{estimator}_n{ntrain}.pt"))
    print(f"Saved NPE at {base_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

    parser.add_argument(
        "--task",
        type=str,
        help="sbibm task name",
    )

    parser.add_argument(
        "--estimator",
        type=str,
        default="nsf",
        help="Type of density estimator, one of 'nsf', 'maf'. Defaults to 'nsf'.",
    )

    parser.add_argument(
        "--ntrain",
        type=int,
        default=50000,
        help="Number of training samples",
    )

    args = parser.parse_args()

    main(args)
