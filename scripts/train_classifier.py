# script to train clasifier, pass arguments by argparse

import argparse
from os import path

import matplotlib.pyplot as plt
import torch
from loss_calibration.classifier import build_classifier, train
from loss_calibration.utils import load_data, prepare_for_training, save_metadata


def main(args):
    assert path.isdir(args.data_dir), "data_dir is no existing directory"
    assert path.isdir(args.res_dir), "res_dir is no existing directory"

    task_name = args.task
    assert task_name in [
        "toy_example",
        "sir",
        "lotka_volterra",
    ], "Choose one of 'toy_example', 'sir' or 'lotka_volterra'."

    model = args.model
    assert model in [
        "fc",
        "resnet",
    ], "Model type should be one of 'fc' or 'resnet'."

    seed = args.seed
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data_dir = path.join(args.data_dir, task_name)
    theta_train, x_train, theta_val, x_val, _, _ = load_data(
        task_name, args.data_dir, device
    )
    ntrain = args.ntrain
    if ntrain > theta_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif ntrain < theta_train.shape[0]:
        theta_train = theta_train[:ntrain]
        x_train = x_train[:ntrain]

    if args.parameter >= 0 and args.parameter <= theta_train.shape[1] - 1:
        dim = args.parameter
        print(f"Restrict parameters to parameter {dim}.")
        theta_train = theta_train[:, dim : dim + 1]
        theta_val = theta_val[:, dim : dim + 1]

    threshold = args.T
    costs = args.costs
    hidden_layers = args.hidden
    learning_rate = args.lr
    epochs = args.epochs

    # create directory & save metadata
    save_dir = path.join(args.res_dir, task_name, "classifier")
    model_dir = prepare_for_training(save_dir, round(threshold, ndigits=4), costs)
    save_metadata(
        model_dir,
        model=model,
        input=x_train.shape[1],
        hidden_layers=hidden_layers,
        costs=costs,
        T=(args.parameter, threshold),
        seed=seed,
        lr=learning_rate,
        ntrain=ntrain,
    )

    # training
    print(
        f"Training specification:\ntask: {task_name}\nmodel: {model}\nseed: {seed}\nmax epochs: {epochs}\nlearning rate: {learning_rate}\ncosts: {costs}\nthreshold: {threshold}\
            \nntrain: {ntrain}\ndata at: {data_dir}\nsave at: {save_dir}\ndevice: {device}"
    )

    clf = build_classifier(model, x_train, hidden_layers, 1)

    clf, loss_values_train, loss_values_val = train(
        clf,
        x_train,
        theta_train,
        x_val,
        theta_val,
        costs,
        threshold,
        learning_rate=learning_rate,
        max_num_epochs=epochs,
        model_dir=model_dir,
        seed=seed,
    )

    # plot loss curve
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        torch.arange(loss_values_train.shape[0]).detach().numpy(),
        loss_values_train,
        label="train",
    )
    ax.plot(
        torch.arange(loss_values_val.shape[0]).detach().numpy(),
        loss_values_val,
        label="val",
    )
    ax.set_title("Loss curve")
    ax.set_ylabel("loss")
    ax.set_xlabel("epochs")
    ax.legend()
    fig.savefig(path.join(model_dir, f"loss_curve.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

    parser.add_argument(
        "--task",
        type=str,
        help="sbibm task name or 'toy_example'",
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Set seed for reproducibility."
    )

    parser.add_argument(
        "--model",
        default="fc",
        help="Model type, one of 'fc', 'resnet'.",
    )

    parser.add_argument(
        "--hidden",
        type=lambda s: [int(item) for item in s.split(",")],
        default=[100, 100, 100],
        help="List specifying the architecture of the network",
    )
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument(
        "--costs",
        type=lambda s: [float(item) for item in s.split(",")],
        required=True,
        help="List specifying the cost of misclassification",
    )
    parser.add_argument(
        "--T", type=float, required=True, help="Threshold for decision making"
    )
    parser.add_argument(
        "--parameter",
        type=int,
        default=-1,
        help="index of parameter used for decision-making",
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=500000,
        help="Number of training samples",
    )

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
