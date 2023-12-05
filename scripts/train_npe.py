# usage: python train_npe.py task.name=toy_example action=continuous model=npe seed=0

import os
from os import path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sbi.utils.sbiutils import seed_all_backends

from loss_cal.npe import train_npe
from loss_cal.utils.utils import check_base_dir_exists, create_seed_dir, load_data


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

    assert cfg.model.type == "npe"

    estimator = cfg.model.estimator
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

    save_dir = path.join(cfg.res_dir, f"{task_name}/npe/")
    check_base_dir_exists(save_dir)
    save_dir_seeded = create_seed_dir(save_dir, seed=seed)
    # prepare_filestructure(path.join(save_dir, "seeds", str(cfg.seed)))

    print(
        f"Training posterior with {cfg.ntrain} simulations: \ndensity estimator: {estimator}\ndata at: {path.join(cfg.data_dir, task_name)}\nsave at: {save_dir}\n"
    )

    print("Data shapes", x_train.shape, theta_train.shape)
    npe_posterior = train_npe(
        task_name=task_name,
        theta_train=theta_train,
        x_train=x_train,
        neural_net=estimator,
        max_num_epochs=epochs,
        device=device,
        seed=seed,
        **task_specifications,
    )

    # test whether sampling works
    npe_posterior.sample((1,), x=x_train[0], show_progress_bars=False)

    torch.save(
        npe_posterior,
        path.join(save_dir_seeded, f"{estimator}_n{ntrain}.pt"),
        # path.join(save_dir, "seeds", str(cfg.seed), f"{estimator}_n{ntrain}.pt"),
    )
    print(f"Saved NPE at {save_dir_seeded}.")


if __name__ == "__main__":
    main()
