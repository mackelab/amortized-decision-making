# script to train clasifier
from os import path
import os

import matplotlib.pyplot as plt
import torch
from loss_calibration.classifier import build_classifier, train
from loss_calibration.utils import load_data, prepare_for_training, save_metadata

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./configs/", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    assert path.isdir(cfg.data_dir), "data_dir is no existing directory"
    assert path.isdir(cfg.res_dir), "res_dir is no existing directory"

    task_name = cfg.task.name
    assert task_name in [
        "toy_example",
        "sir",
        "lotka_volterra",
    ], "Choose one of 'toy_example', 'sir' or 'lotka_volterra'."

    model = cfg.model.type
    assert model in [
        "fc",
        "resnet",
    ], "Model type should be one of 'fc' or 'resnet'."

    seed = cfg.seed
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    data_dir = path.join(cfg.data_dir, task_name)
    theta_train, x_train, theta_val, x_val, _, _ = load_data(
        task_name, cfg.data_dir, device
    )
    ntrain = cfg.ntrain
    if ntrain > theta_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif ntrain < theta_train.shape[0]:
        theta_train = theta_train[:ntrain]
        x_train = x_train[:ntrain]

    if cfg.task.parameter >= 0 and cfg.task.parameter <= theta_train.shape[1] - 1:
        dim = cfg.task.parameter
        print(f"Restrict parameters to parameter {dim}.")
        theta_train = theta_train[:, dim : dim + 1]
        theta_val = theta_val[:, dim : dim + 1]

    threshold = round(cfg.task.T, ndigits=4)
    parameter = cfg.task.parameter
    costs = list(cfg.task.costs)
    hidden_layers = list(cfg.model.hidden)
    learning_rate = cfg.model.lr
    z_scoring = cfg.model.zscore
    epochs = cfg.model.epochs

    # create directory & save metadata
    save_dir = path.join(cfg.res_dir, task_name, "classifier")
    model_dir = prepare_for_training(
        save_dir, parameter, round(threshold, ndigits=4), costs
    )
    save_metadata(
        model_dir,
        model=model,
        input=x_train.shape[1],
        hidden_layers=hidden_layers,
        z_scoring=z_scoring,
        costs=costs,
        T=(parameter, threshold),
        seed=seed,
        lr=learning_rate,
        ntrain=ntrain,
        epochs=epochs,
        data_dir=data_dir,
    )

    # training
    print(f"data at: {data_dir}\nsave at: {save_dir}\ndevice: {device}")

    clf = build_classifier(model, x_train, hidden_layers, 1, z_scoring=z_scoring)
    print(clf)

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
    main()
