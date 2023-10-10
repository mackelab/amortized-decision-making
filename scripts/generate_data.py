# script to train clasifier, pass arguments by argparse

import argparse
from os import getcwd, path

import torch

from loss_cal.tasks.linear_gaussian import LinGauss
from loss_cal.tasks.lotka_volterra import LotkaVolterra
from loss_cal.tasks.sir import SIR
from loss_cal.tasks.toy_example import ToyExample

torch.manual_seed(758)


def main(args):
    task_name = args.task
    action_type = args.type

    n_train = args.ntrain
    n_val = args.nval
    n_test = args.ntest

    # print(f"task {task}, n_train {n_train}")
    # print(f"Current directory: {getcwd()}")

    assert task_name in ["toy_example", "lotka_volterra", "sir", "linear_gaussian"]
    assert action_type in ["binary", "continuous"]

    if task_name == "toy_example":
        task = ToyExample()
        prior = task.get_prior()
        simulator = task.get_simulator()
        factor, exponential = 2, 3
    elif task_name == "lotka_volterra":
        task = LotkaVolterra()
        prior = task.get_prior()
        simulator = task.get_simulator()
        factor, exponential = 2, 2
    elif task_name == "sir":
        task = SIR()
        prior = task.get_prior()
        simulator = task.get_simulator()
        factor, exponential = 1, 2

    elif task_name == "linear_gaussian":
        task = LinGauss()
        prior = task.get_prior()
        simulator = task.get_simulator()
        factor, exponential = 0.5, 2

    actions = task.actions

    print("Sample parameter values.")
    thetas = task.sample_prior(n_train + n_test + n_val)
    print(thetas.shape)
    print("Run simulator.")
    observations = []
    for i, th in enumerate(thetas):
        print(f"{i}/{thetas.shape[0]}", end="\r")
        observations.append(simulator(th))
    observations = torch.vstack(observations)
    print(observations.shape)

    if action_type != "binary":
        print("Sample actions.")
        actions = actions.sample(thetas.shape[0])
        torch.save(actions[:n_train], path.join(args.data_dir, task_name, "actions_train.pt"))
        torch.save(actions[n_train : n_train + n_val], path.join(args.data_dir, task_name, "actions_val.pt"))
        torch.save(actions[n_train + n_val :], path.join(args.data_dir, task_name, "actions_test.pt"))

    # save data
    print("Save data.")
    torch.save(thetas[:n_train], path.join(args.data_dir, task_name, "theta_train.pt"))
    torch.save(observations[:n_train], path.join(args.data_dir, task_name, "x_train.pt"))
    torch.save(
        thetas[n_train : n_train + n_val],
        path.join(args.data_dir, task_name, "theta_val.pt"),
    )
    torch.save(
        observations[n_train : n_train + n_val],
        path.join(args.data_dir, task_name, "x_val.pt"),
    )
    torch.save(thetas[n_train + n_val :], path.join(args.data_dir, task_name, "theta_test.pt"))
    torch.save(observations[n_train + n_val :], path.join(args.data_dir, task_name, "x_test.pt"))

    print(f"Saved data to '{path.join(args.data_dir, task_name)}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a new data set")

    parser.add_argument(
        "--task",
        type=str,
        help="Task to generate data for. One of ['toy_example', 'lotka_volterra', 'sir', 'linear_gaussian']",
    )

    parser.add_argument(
        "--type",
        type=str,
        default="continuous",
        help="Type of actions. One of ['binary', 'continuous']",
    )

    parser.add_argument("--ntrain", type=int, default=500000, help="Number of training samples")
    parser.add_argument("--nval", type=int, default=100000, help="Number of validation samples")
    parser.add_argument("--ntest", type=int, default=100000, help="Number of test samples")
    parser.add_argument(
        "--data_dir",
        default="../data/",
        help="Directory to save the data set. Make sure it exists before as well as a subfolder with the task name.",
    )

    args = parser.parse_args()
    main(args)
