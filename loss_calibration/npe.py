import logging
import torch
from typing import Optional
from sbibm.algorithms.sbi.utils import (
    wrap_posterior,
    wrap_prior_dist,
)
from sbi import inference as inference
from sbi.utils.get_nn_models import posterior_nn


def train_npe(
    task,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    neural_net: str = "nsf",
    hidden_features: int = 50,
    simulation_batch_size: int = 1000,
    training_batch_size: int = 10000,
    num_atoms: int = 10,
    automatic_transforms_enabled: bool = False,
    z_score_x: bool = True,
    z_score_theta: bool = True,
    max_num_epochs: Optional[int] = None,
):
    assert not (num_observation is None and observation is None)
    assert not (num_observation is not None and observation is not None)
    log = logging.getLogger(__name__)
    log.info(f"Running NPE")

    num_simulations = theta_train.shape[0]

    if simulation_batch_size > num_simulations:
        simulation_batch_size = num_simulations
        log.warn("Reduced simulation_batch_size to num_simulations")

    if training_batch_size > num_simulations:
        training_batch_size = num_simulations
        log.warn("Reduced training_batch_size to num_simulations")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(num_observation)

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
        retrain_from_scratch_each_round=False,
        discard_prior_samples=False,
        use_combined_loss=False,
        show_train_summary=True,
        max_num_epochs=max_num_epochs,
    )
    posterior = inference_method.build_posterior(density_estimator)

    posterior_wrapped = wrap_posterior(posterior, transforms)

    return posterior  # , posterior_wrapped
