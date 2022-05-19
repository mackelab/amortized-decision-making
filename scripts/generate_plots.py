# script to train clasifier, pass arguments by argparse

import argparse
import torch
from os import path
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import glob


def main(args):

    # load classifiers
    files = sorted(glob.glob(path.join(args.model_dir, "2022-04-25T17_2*.pt")))
    print(f"Generating plots for files {files}\n")
    clfs = []

    for file in files:
        clfs.append(torch.load(file))

    # # load test data
    # th_train = torch.load(path.join(args.data_dir, "th_test.pt"))
    # x_train = torch.load(path.join(args.data_dir, "x_test.pt"))

    # threshold = args.T

    # # plots

    # plt.plot(torch.arange(epochs).detach().numpy(), loss_values)
    # plt.title("Loss curve")
    # plt.ylabel("loss")
    # plt.xlabel("epochs")
    # plt.savefig(path.join(args.save_dir, f"{timestamp}_loss.pdf"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

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
        "--model_dir",
        default="../results/1d_classifier/",
        help="Path to trained models",
    )
    parser.add_argument(
        "--save_dir",
        default="../results/1d_classifier/",
        help="Output directory for figures",
    )

    args = parser.parse_args()

    # TODO: train classifier
    main(args)
