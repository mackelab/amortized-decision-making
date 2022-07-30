import torch
import os

# tresholds
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
    zip(indices, list(torch.round(thresholds_alpha, decimals=2).numpy()))
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


# for costs in costs_list:
for (idx, T) in indexed_thresholds:
    print(idx, T)
    os.system(
        f"python train_classifier.py  --task lotka_volterra --costs {','.join(str(c) for c in costs)} --T {T} --parameter {idx} --ntrain 100_000 --res_dir ../res/"
    )
