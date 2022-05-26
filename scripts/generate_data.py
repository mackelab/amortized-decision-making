# script to train clasifier, pass arguments by argparse

import argparse
from os import path
from sbi.utils import BoxUniform
import torch

torch.manual_seed(758)


def main(args):
    # define prior and simulator
    prior = BoxUniform([0.0], [5.0])

    def simulator(theta):
        # return 5*theta + torch.randn(theta.shape) * 3
        return (
            50 + 0.5 * theta * (5 - theta) ** 4 + torch.randn(theta.shape) * 10
        )  # * (theta+3)

    n_train = args.ntrain
    n_val = args.nval
    n_test = args.ntest

    thetas = prior.sample((n_train + n_test + n_val,))
    observations = simulator(thetas)

    # threshold = args.T
    # decisions = (thetas > threshold).float()  # labels 0,1

    # save data
    torch.save(thetas[:n_train], path.join(args.data_dir, "th_train.pt"))
    torch.save(observations[:n_train], path.join(args.data_dir, "x_train.pt"))
    torch.save(thetas[n_train : n_train + n_val], path.join(args.data_dir, "th_val.pt"))
    torch.save(
        observations[n_train : n_train + n_val], path.join(args.data_dir, "x_val.pt")
    )
    torch.save(thetas[n_train + n_val :], path.join(args.data_dir, "th_test.pt"))
    torch.save(observations[n_train + n_val :], path.join(args.data_dir, "x_test.pt"))

    print(f"Saved data to '{args.data_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a new data set")

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
        default="../data/1d_classifier",
        help="Directory to save the data set",
    )

    args = parser.parse_args()
    main(args)
