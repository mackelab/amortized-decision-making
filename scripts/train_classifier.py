# script to train clasifier, pass arguments by argparse

import argparse
from loss_calibration.classifier import FeedforwardNN, train
import torch
from datetime import datetime
from os import device_encoding, path, mkdir
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import json


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    th_train = torch.load(path.join(args.data_dir, "th_train.pt"), map_location=device)
    x_train = torch.load(path.join(args.data_dir, "x_train.pt"), map_location=device)
    if args.ntrain > th_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif args.ntrain < th_train.shape[0]:
        th_train = th_train[: args.ntrain]
        x_train = x_train[: args.ntrain]
    th_val = torch.load(path.join(args.data_dir, "th_val.pt"), map_location=device)
    x_val = torch.load(path.join(args.data_dir, "x_val.pt"), map_location=device)

    threshold = args.T
    costs = args.costs

    # create directory & save metadata
    timestamp = datetime.now().isoformat().split(".")[0].replace(":", "_")
    model_dir = path.join(
        args.save_dir, f"{timestamp}_t{int(threshold)}_c{int(costs[0])}_{int(costs[1])}"
    )
    try:
        mkdir(model_dir)
        mkdir(path.join(model_dir, "checkpoints/"))
        print(f"Directory {model_dir} created.")
    except FileExistsError:
        print(f"Directory {model_dir} already exists.")
    metadata = {
        "seed": args.seed,
        "architecture": "1-16-1",  # TODO: save correct model architecture
        "optimizer": "Adam",
        "Ntrain": th_train.shape[0],
        "threshold": threshold,
        "costs": costs,
    }
    json.dump(metadata, open(f"{model_dir}/metadata.json", "w"))

    # training
    print(
        f"Training specification:\nseed: {args.seed}\nepochs: {args.epochs}\ncosts: {args.costs}\nthreshold: {args.T}\
            \nntrain: {args.ntrain}\ndata_dir: {args.data_dir}\nsave_dir: {args.save_dir}\ndevice: {device}"
    )
    clf = FeedforwardNN(1, [16], 1)

    epochs = args.epochs

    clf, loss_values = train(
        clf,
        x_train,
        th_train,
        x_val,
        th_val,
        costs,
        threshold,
        max_num_epochs=epochs,
        model_dir=model_dir,
    )

    # save trained classifier and metadata
    plt.plot(torch.arange(loss_values.shape[0]).detach().numpy(), loss_values)
    plt.title("Loss curve")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig(path.join(model_dir, f"{timestamp}_loss.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

    parser.add_argument(
        "--seed", type=int, default=9834, help="Set seed for reproducibility."
    )
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument(
        "--costs",
        type=lambda s: [float(item) for item in s.split(",")],
        default=[1.0, 1.0],
        help="List specifying the cost of misclassification",
    )
    parser.add_argument(
        "--T", type=float, default=2.0, help="Threshold for decision-making"
    )
    parser.add_argument(
        "--ntrain",
        type=int,
        default=50000,
        help="Number of training samples",
    )
    parser.add_argument(
        "--data_dir",
        default="../data/1d_classifier/",
        help="Path to training data",
    )
    parser.add_argument(
        "--save_dir",
        default="../results/1d_classifier/",
        help="Output directory for trained model",
    )

    args = parser.parse_args()

    # TODO: train classifier
    main(args)
