import torch
import glob
import sbibm
from os import path
from loss_calibration.utils import load_data, posterior_ratio_given_samples
from loss_calibration.toy_example import ratio_given_posterior

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    assert path.isdir(cfg.data_dir), "data_dir is no existing directory"
    assert path.isdir(cfg.res_dir), "res_dir is no existing directory"

    estimator = cfg.model.estimator
    assert estimator in [
        "nsf",
        "maf",
    ], "Density estimator has to be either 'nsf' or 'maf'."

    task_name = cfg.task.name
    assert task_name in [
        "toy_example",
        "sir",
        "lotka_volterra",
    ], f"Choose one of 'toy_example', 'sir' or 'lotka_volterra'., got {task_name}."

    device = torch.device(
        "cpu"
    )  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    threshold = round(cfg.task.T, ndigits=4)
    parameter = cfg.task.parameter
    costs = list(cfg.task.costs)
    nsim = cfg.ntrain

    file = path.join(cfg.res_dir, task_name, f"npe/{estimator}_n{nsim}.pt")
    npe_posterior = torch.load(file)

    if task_name in ["lotka_volterra", "sir"]:
        # use 10 reference observations + sample from posterior
        task = sbibm.get_task(task_name)
        num_samples_from_posterior = 10_000

        npe_samples = []
        npe_ratios = []
        for n in range(10):
            samples = npe_posterior.sample(
                (num_samples_from_posterior,), x=task.get_observation(n + 1)
            )
            npe_samples.append(samples)
            npe_ratio = posterior_ratio_given_samples(
                samples[:, parameter], threshold, costs
            )
            npe_ratios.append(npe_ratio)

        torch.save(
            torch.stack(npe_ratios),
            path.join(
                cfg.res_dir,
                task_name,
                f"npe/{cfg.experiment}/{estimator}_n{nsim}_predictions_t{parameter}_{threshold}_c{int(costs[0])}_{int(costs[1])}.pt",
            ),
        )
    elif task_name == "toy_example":
        # use test data + evaluate posterior on linspace (1D)
        _, _, theta_test, x_test, _, _ = load_data(task_name, cfg.data_dir, device)
        theta_test = theta_test[: cfg.ntest]
        x_test = x_test[: cfg.ntest]
        N_test = theta_test.shape[0]
        print("N_test = ", N_test)

        npe_ratio = torch.as_tensor(
            [
                ratio_given_posterior(
                    npe_posterior, x_o, threshold=threshold, costs=costs
                )
                for x_o in x_test
            ]
        ).unsqueeze(1)
        torch.save(
            npe_ratio,
            path.join(
                cfg.res_dir,
                task_name,
                f"npe/{cfg.experiment}/{estimator}_n{nsim}_predictions_t{parameter}_{str(threshold).replace('.', '_')}_c{int(costs[0])}_{int(costs[1])}.pt",
            ),
        )


if __name__ == "__main__":
    main()
