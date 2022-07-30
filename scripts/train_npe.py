import argparse
from os import path

import torch
from loss_calibration.npe import train_npe
from loss_calibration.utils import check_base_dir_exists, load_data


def main(args):
    assert path.isdir(args.data_dir), "data_dir is no existing directory"
    assert path.isdir(args.res_dir), "res_dir is no existing directory"

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
    epochs = args.epochs
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    theta_train, x_train, _, _, _, _ = load_data(task_name, args.data_dir, device)
    if ntrain > theta_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif ntrain < theta_train.shape[0]:
        theta_train = theta_train[:ntrain]
        x_train = x_train[:ntrain]

    save_dir = path.join(args.res_dir, f"{task_name}/npe/")
    check_base_dir_exists(save_dir)

    print(
        f"Training posterior with {args.ntrain} simulations: \ndensity estimator: {estimator}\ndata at: {path.join(args.data_dir, task_name)}\nsave at: {save_dir}\n"
    )

    npe_posterior = train_npe(
        task_name, theta_train, x_train, max_num_epochs=epochs, device=device
    )
    torch.save(npe_posterior, path.join(save_dir, f"{estimator}_n{ntrain}.pt"))
    print(f"Saved NPE at {save_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

    parser.add_argument(
        "--task",
        type=str,
        help="sbibm task name or 'toy_example'",
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

    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")

    parser.add_argument(
        "--data_dir",
        default="../data/",
        help="Base directory of training data",
    )
    parser.add_argument(
        "--res_dir",
        default="../results/",
        help="Base directory for saving",
    )

    args = parser.parse_args()

    main(args)
