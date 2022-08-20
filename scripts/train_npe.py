from os import path

import torch
from loss_calibration.npe import train_npe
from loss_calibration.utils import (
    check_base_dir_exists,
    load_data,
    prepare_filestructure,
)

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    _ = torch.manual_seed(cfg.seed)

    assert path.isdir(cfg.data_dir), "data_dir is no existing directory"
    assert path.isdir(cfg.res_dir), "res_dir is no existing directory"

    task_name = cfg.task.name
    assert task_name in [
        "toy_example",
        "sir",
        "lotka_volterra",
        "linear_gaussian",
    ], "Choose one of 'linear_gaussian', 'toy_example', 'sir' or 'lotka_volterra'."

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

    theta_train, x_train, _, _, _, _ = load_data(task_name, cfg.data_dir, device)
    if ntrain > theta_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif ntrain < theta_train.shape[0]:
        theta_train = theta_train[:ntrain]
        x_train = x_train[:ntrain]

    save_dir = path.join(cfg.res_dir, f"{task_name}/npe/")
    check_base_dir_exists(save_dir)
    prepare_filestructure(path.join(save_dir, "seeds", str(cfg.seed)))

    print(
        f"Training posterior with {cfg.ntrain} simulations: \ndensity estimator: {estimator}\ndata at: {path.join(cfg.data_dir, task_name)}\nsave at: {save_dir}\n"
    )

    npe_posterior = train_npe(
        task_name,
        theta_train,
        x_train,
        neural_net=estimator,
        max_num_epochs=epochs,
        device=device,
    )
    torch.save(
        npe_posterior,
        path.join(save_dir, "seeds", str(cfg.seed), f"{estimator}_n{ntrain}.pt"),
    )
    print(f"Saved NPE at {save_dir}.")


if __name__ == "__main__":
    main()
