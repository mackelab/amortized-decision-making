import torch
from os import path
from loss_calibration.npe import train_npe
from loss_calibration.utils import prepare_for_npe_training, save_npe_metadata
import sbibm
import argparse


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

    # task = sbibm.get_task(task_name)

    base_dir = f"../results/{task_name}/npe/"
    model_dir = prepare_for_npe_training(base_dir, ntrain)
    save_npe_metadata(model_dir, estimator, ntrain)

    print(
        f"Training posterior with {args.ntrain} simulations: \ndensity estimator: {estimator}\ndata_dir: {data_dir}\nsave_dir: ./results/{task_name}\n"
    )

    npe_posterior = train_npe(task_name, theta_train, x_train, num_observation=1)
    torch.save(
        npe_posterior,
        f"../results/{task_name}/npe/npe_posterior_sim{theta_train.shape[0]}_unwraped.pt",
    )
    print(f"Saved NPE at './results/{task_name}/'.")


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
