import argparse
import os
from os import path

import torch


def main(args):
    task_name = args.task
    action_type = args.type
    assert task_name in [
        "toy_example",
        "sir",
        "lotka_volterra",
    ], "Choose one of 'sir' or 'lotka_volterra'."
    assert action_type in ["binary", "continuous"], "Specifiy the type of actions, one of 'binary' or 'continuous'."
    print("Task: ", task_name)

    # tresholds
    # if  task_name == "toy_example":

    if action_type == "binary":
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
            thresholds = [thresholds_alpha, thresholds_beta, thresholds_gamma, thresholds_delta]

            indexed_thresholds = list(zip(indices, list(torch.round(torch.cat(thresholds), decimals=2).numpy())))
            costs = [20.0, 1.0]
        elif task_name == "sir":
            labels = ["beta", "gamma"]
            thresholds_beta = torch.linspace(0.2, 0.8, 25)
            thresholds_gamma = torch.linspace(0.08, 0.23, 31)
            indices = [0] * thresholds_beta.shape[0] + [1] * thresholds_gamma.shape[0]
            thresholds = [thresholds_beta, thresholds_gamma]

            indexed_thresholds = list(zip(indices, list(torch.round(torch.cat(thresholds), decimals=2).numpy())))

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
        for idx, T in indexed_thresholds[:1]:
            print(idx, T)
            os.system(
                f"python train_nn.py experiment=vary_T task.name={task_name} task.parameter={idx} action={action_type} action.costs={'['+','.join(str(c) for c in costs)+']'} action.T={round(T, ndigits=2)}  ntrain=50_000 res_dir=../results/ data_dir=../data/"
            )

    elif action_type == "continuous":
        if task_name == "lotka_volterra":
            labels = ["alpha", "beta", "gamma", "delta"]
            params = torch.arange(0, len(labels))
            factor = 2
            exponential = 2
            # action_lower, action_upper, interval = 0, 100, 2.0
            # a_grid = torch.arange(action_lower, action_upper, interval)

        if task_name == "sir":
            labels = ["beta", "gamma"]
            params = torch.arange(0, len(labels))
            factor = 1
            exponential = 2

        # n = 50_000
        for n in [500, 1_000, 2_500, 5_000, 7_500, 10_000, 20_000, 30_000, 40_000, 50_000, 100_000]:
            for idx in params.tolist():
                print(idx)
                os.system(
                    # vary_params
                    f"python train_nn.py experiment=vary_sim task.name={task_name} task.parameter={idx} action={action_type} action.factor={factor} action.exponential={exponential} ntrain={n} model.hidden='[100,100,100]' model.epochs=1000 res_dir=../results/ data_dir=../data/"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="sbibm task name",
    )

    parser.add_argument(
        "--type",
        type=str,
        required=True,
        help="type of actions",
    )

    args = parser.parse_args()

    main(args)
