# usage: python train_posterior_estimator.py task.name=toy_example action=continuous model=npe seed=0
# usage: python train_posterior_estimator.py task.name=toy_example action=continuous model=nle seed=0


import os
from os import path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sbi.utils.sbiutils import seed_all_backends

from bam.utils.utils import load_data, create_filestructure
from bam.posterior_estimator import train_neural_estimator


@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    seed = cfg.seed
    seed_all_backends(seed)

    assert path.isdir(cfg.data_dir), "data_dir is no existing directory"
    assert path.isdir(cfg.res_dir), "res_dir is no existing directory"

    task_name = cfg.task.name
    action_type = cfg.action.type
    if action_type == "discrete":
        num_actions = (
            None if cfg.action.num_actions == "None" else int(cfg.action.num_actions)
        )
        probs = None if cfg.action.probs == "None" else list(cfg.action.probs)
        task_specifications = {
            "action_type": action_type,
            "num_actions": num_actions,
            "probs": probs,
        }
    else:
        task_specifications = {"action_type": action_type}
    assert task_name in [
        "toy_example",
        "sir",
        "lotka_volterra",
        "linear_gaussian",
        "bvep",
    ], "Choose one of 'linear_gaussian', 'toy_example', 'sir' or 'lotka_volterra', 'bvep'."

    assert cfg.model.type in [
        "npe",
        "nle",
    ], "Specified method (model type) must be one of 'npe' or 'nle'."
    method = cfg.model.type

    estimator = cfg.model.density_estimator
    assert estimator in [
        "nsf",
        "maf",
    ], "Density estimator has to be either 'nsf' or 'maf'."

    ntrain = cfg.ntrain
    epochs = cfg.model.epochs
    device = torch.device(
        "cpu"
    )  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Current path:", os.getcwd())
    print("Data dir:", cfg.data_dir)

    theta_train, x_train, _, _ = load_data(
        task_name=task_name, base_dir=cfg.data_dir, device=device
    )
    if ntrain > theta_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    if ntrain < theta_train.shape[0]:
        theta_train = theta_train[:ntrain]
        x_train = x_train[:ntrain]

    save_dir_seeded = path.join(cfg.res_dir, f"{task_name}/{method}/{seed}")
    create_filestructure(save_dir_seeded)

    print(
        f"Training posterior with {cfg.ntrain} simulations: \ndensity estimator: {estimator}\ndata at: {path.join(cfg.data_dir, task_name)}\nsave at: {save_dir_seeded}\n"
    )

    print("Data shapes", x_train.shape, theta_train.shape)
    estimated_posterior = train_neural_estimator(
        method=method,
        task_name=task_name,
        theta_train=theta_train,
        x_train=x_train,
        flow=estimator,
        max_num_epochs=epochs,
        device=device,
        seed=seed,
        **task_specifications,
    )

    # test whether sampling works
    # estimated_posterior.sample((1,), x=x_train[0], show_progress_bars=False)
    # print("Sampling test successful.")
    # issue: nle posterior cannot be pickled after sampling

    torch.save(
        estimated_posterior,
        path.join(save_dir_seeded, f"{estimator}_n{ntrain}.pt"),
    )
    print(f"Saved {method.upper()} at {save_dir_seeded}.")


if __name__ == "__main__":
    main()
