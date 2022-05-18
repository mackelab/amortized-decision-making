# script to train clasifier, pass arguments by argparse

import argparse
import torch
from datetime import datetime
from os import path
from sbi.utils import BoxUniform
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


def main(args):
    estimator = args.estimator
    assert estimator in [
        "nsf",
        "maf",
    ], "Density estimator has to be either 'nsf' or 'maf'."
    print(
        f"Training posterior with {args.ntrain} simulations: \ndensity estimator: {estimator}\ndata_dir: {args.data_dir}\nsave_dir: {args.save_dir}\n"
    )

    # load data
    th_train = torch.load(path.join(args.data_dir, "th_train.pt"))
    x_train = torch.load(path.join(args.data_dir, "x_train.pt"))
    if args.ntrain > th_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif args.ntrain < th_train.shape[0]:
        th_train = th_train[: args.ntrain]
        x_train = x_train[: args.ntrain]

    prior = BoxUniform(
        [0.0],
        [
            5.0,
        ],
    )
    inference = SNPE(prior=prior, density_estimator=estimator)
    inference = inference.append_simulations(th_train, x_train)
    density_estimator = inference.train()
    posterior_sbi = inference.build_posterior(density_estimator)

    # save trained classifier and metadata
    timestamp = datetime.now().isoformat().split(".")[0].replace(":", "_")
    torch.save(
        posterior_sbi,
        path.join(
            args.save_dir, f"{timestamp}_{args.estimator}_{th_train.shape[0]}.pt"
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the classifier")

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
    parser.add_argument(
        "--data_dir",
        default="../data/1d_classifier/",
        help="Path to training data",
    )
    parser.add_argument(
        "--save_dir",
        default="../results/sbi/",
        help="Output directory for trained model",
    )

    args = parser.parse_args()

    main(args)
