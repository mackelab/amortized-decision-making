# script to train clasifier, pass arguments by argparse

import argparse
from loss_calibration.classifier import FeedforwardNN, train
from loss_calibration.loss import BCELoss_weighted
import torch
from datetime import datetime
from os import path
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv


def main(args):
    print(
        f"Training specification:\nepochs: {args.epochs}\nweights: {args.weights}\nthreshold: {args.T}\
            \nntrain: {args.ntrain}\ndata_dir: {args.data_dir}\nsave_dir: {args.save_dir}\n"
    )
    # load data
    th_train = torch.load(path.join(args.data_dir, "th_train.pt"))
    x_train = torch.load(path.join(args.data_dir, "x_train.pt"))
    if args.ntrain > th_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif args.ntrain < th_train.shape[0]:
        th_train = th_train[: args.ntrain]
        x_train = x_train[: args.ntrain]

    threshold = args.T
    weights = args.weights
    # training
    clf = FeedforwardNN(1, [16], 1)

    epochs = args.epochs
    optimizer = torch.optim.Adam(clf.parameters())
    criterion = BCELoss_weighted(weights, threshold)

    clf, loss_values = train(
        clf, x_train, th_train, threshold, epochs, criterion, optimizer
    )
    clf._summary["weights"] = weights  # work-around to save weights

    # save trained classifier and metadata
    timestamp = datetime.now().isoformat().split(".")[0].replace(":", "_")
    torch.save(clf, path.join(args.save_dir, f"{timestamp}_1d_classifier.pt"))
    torch.save(clf, path.join(args.save_dir, f"{timestamp}_loss.pt"))
    with open(
        path.join(args.save_dir, "classifiers.csv"), "a", encoding="UTF8", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            [timestamp, str(weights), threshold, epochs, optimizer, th_train.shape[0]]
        )

    plt.plot(torch.arange(epochs).detach().numpy(), loss_values)
    plt.title("Loss curve")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.savefig(path.join(args.save_dir, f"{timestamp}_loss.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

    parser.add_argument("--epochs", type=int, default=5000, help="Number of epochs")
    parser.add_argument(
        "--weights",
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
