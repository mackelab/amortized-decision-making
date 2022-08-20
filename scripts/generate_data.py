# script to train clasifier, pass arguments by argparse

import argparse
from os import path
import torch
import loss_calibration.toy_example as toy
import loss_calibration.lotka_volterra as lv
import loss_calibration.linear_gaussian as lin_gauss
import loss_calibration.sir as sir

torch.manual_seed(758)


def main(args):
    task = args.task
    n_train = args.ntrain
    n_val = args.nval
    n_test = args.ntest

    assert task in ["toy_example", "lotka_volterra", "sir", "linear_gaussian"]

    if task == "toy_example":
        prior = toy.get_prior()
        simulator = toy.get_simulator()
    elif task == "lotka_volterra":
        prior = lv.get_prior()
        simulator = lv.get_simulator()
    elif task == "sir":
        prior = sir.get_prior()
        simulator = sir.get_simulator()
    elif task == "linear_gaussian":
        prior = lin_gauss.get_prior()
        simulator = lin_gauss.get_simulator()

    thetas = prior.sample((n_train + n_test + n_val,))
    observations = simulator(thetas)

    # save data
    torch.save(thetas[:n_train], path.join(args.data_dir, task, "theta_train.pt"))
    torch.save(observations[:n_train], path.join(args.data_dir, task, "x_train.pt"))
    torch.save(
        thetas[n_train : n_train + n_val],
        path.join(args.data_dir, task, "theta_val.pt"),
    )
    torch.save(
        observations[n_train : n_train + n_val],
        path.join(args.data_dir, task, "x_val.pt"),
    )
    torch.save(
        thetas[n_train + n_val :], path.join(args.data_dir, task, "theta_test.pt")
    )
    torch.save(
        observations[n_train + n_val :], path.join(args.data_dir, task, "x_test.pt")
    )

    print(f"Saved data to '{path.join(args.data_dir, task)}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a new data set")

    parser.add_argument(
        "--task",
        type=str,
        help="Task to generate data for. One of ['toy_example', 'lotka_volterra']",
    )

    parser.add_argument(
        "--ntrain", type=int, default=500000, help="Number of training samples"
    )
    parser.add_argument(
        "--nval", type=int, default=100000, help="Number of validation samples"
    )
    parser.add_argument(
        "--ntest", type=int, default=100000, help="Number of test samples"
    )
    parser.add_argument(
        "--data_dir",
        default="../data/",
        help="Directory to save the data set",
    )

    args = parser.parse_args()
    main(args)
