import torch
from loss_calibration.classifier import FeedforwardNN, train
from loss_calibration.utils import prepare_for_training, save_metadata
from loss_calibration.lotka_volterra import load_data

# load data
theta_train, x_train, theta_val, x_val, theta_test, x_test = load_data(
    base_dir="../data/"
)


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
    base_dir = "../results/lotka_volterra/classifier_varying_T"
    model_dir = prepare_for_training(base_dir, T, costs)

    seed = 0
    lr = 0.001
    input_dim = 20
    hidden_dims = [100, 100, 100]
    save_metadata(
        model_dir,
        input_dim,
        hidden_dims,
        costs,
        float(T),
        seed,
        lr,
        x_train.shape[0],
    )

    print("start training")
    clf = FeedforwardNN(input_dim, hidden_dims, 1)

    clf, loss_values_train, loss_values_val = train(
        clf,
        x_train,
        theta_train[:, idx : idx + 1],
        x_val,
        theta_val[:, idx : idx + 1],
        costs,
        T,
        learning_rate=lr,
        max_num_epochs=5000,
        stop_after_epochs=100,
        model_dir=model_dir,
        seed=seed,
    )
