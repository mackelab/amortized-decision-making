from os import path

import torch
from loss_calibration.npe import train_npe
from loss_calibration.utils import (
    check_base_dir_exists,
    load_data,
    prepare_filestructure,
)
import loss_calibration.toy_example as toy
import loss_calibration.sir as sir
import loss_calibration.lotka_volterra as lv
from loss_calibration.active_learning import ActiveLearning

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    _ = torch.manual_seed(cfg.seed)

    assert path.isdir(cfg.data_dir), "data_dir is no existing directory"
    assert path.isdir(cfg.res_dir), "res_dir is no existing directory"

    task_name = cfg.task.name
    if task_name == "toy_example":
        prior = toy.get_prior()
        simulator = toy.get_simulator()
    elif task_name == "sir":
        prior = sir.get_prior()
        simulator = sir.get_simulator()
    elif task_name == "lotka_volterra":
        prior = lv.get_prior()
        simulator = lv.get_simulator()
    else:
        raise ValueError(
            "task has to be one of ['toy_example', 'sir', 'lotka_volterra']."
        )

    threshold = round(cfg.task.T, ndigits=4)
    parameter = cfg.task.parameter
    costs = list(cfg.task.costs)

    save_dir = path.join(cfg.res_dir, task_name, "active_learning")
    print(f"Save results at {save_dir}")

    al = ActiveLearning(
        prior=prior,
        simulator=simulator,
        threshold=threshold,
        threshold_dim=parameter,
        costs=costs,
        save_dir=save_dir,
    )

    al.run(n_rounds=5)


if __name__ == "__main__":
    main()
