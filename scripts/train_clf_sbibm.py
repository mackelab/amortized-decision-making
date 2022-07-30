import argparse
import torch
from os import path
import os


def main(args):
    task_name = args.task
    assert task_name in [
        "sir",
        "lotka_volterra",
    ], "Choose one of 'sir' or 'lotka_volterra'."
    print("Task: ", task_name)

    # tresholds
    if task_name == "lotka_volterra":
        labels = ["alpha", "beta", "gamma", "delta"]
        thresholds_alpha = torch.linspace(0.2, 1.7, 16)
        thresholds_beta = torch.linspace(0.01, 0.25, 25)
        thresholds_gamma = torch.linspace(0.2, 3.5, 34)
        thresholds_delta = torch.linspace(0.01, 0.14, 14)
        indices = (
            [0] * thresholds_alpha.shape[0]
            + [1] * thresholds_beta.shape[0]
            + [2] * thresholds_gamma.shape[0]
            + [3] * thresholds_delta.shape[0]
        )
        thresholds = torch.cat(
            [thresholds_alpha, thresholds_beta, thresholds_gamma, thresholds_delta]
        )

        indexed_thresholds = list(
            zip(indices, list(torch.round(thresholds, decimals=2).numpy()))
        )
        print("Train with tresholds: ", indexed_thresholds)

        costs_list = [
            [1.0, 20.0],
            [1.0, 10.0],
            [1.0, 5.0],
            [1.0, 1.0],
            [5.0, 1.0],
            [10.0, 1.0],
            [20.0, 1.0],
        ]
        costs = [20.0, 1.0]
    elif task_name == "sir":
        labels = ["beta", "gamma"]
        thresholds_beta = torch.linspace(0.2, 0.8, 25)
        thresholds_gamma = torch.linspace(0.08, 0.23, 31)
        indices = [0] * thresholds_beta.shape[0] + [1] * thresholds_gamma.shape[0]
        thresholds = [thresholds_beta, thresholds_gamma]

        indexed_thresholds = list(
            zip(indices, list(torch.round(torch.cat(thresholds), decimals=2).numpy()))
        )


        costs_list = [
            [1.0, 20.0],
            [1.0, 10.0],
            [1.0, 5.0],
            [1.0, 1.0],
            [5.0, 1.0],
            [10.0, 1.0],
            [20.0, 1.0],
        ]
        costs = [20.0, 1.0]
    else:
        print("Task not yet defined!")

    # for costs in costs_list:
    for (idx, T) in indexed_thresholds:
        print(idx, T)
        os.system(
            f"python train_classifier.py  --task {task_name} --costs {','.join(str(c) for c in costs)} --T {T} --parameter {idx} --ntrain 100_000 --res_dir ../res/"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="sbibm task name",
    )

    args = parser.parse_args()

    main(args)
