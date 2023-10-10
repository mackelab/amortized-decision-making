# script to train clasifier
# usage: python train_nn.py  experiment=vary_T task.name=lotka_volterra task.costs=[20,1] task.T=0.675 task.parameter=0 ntrain=50_000 res_dir=../results/

import os
from os import path
from typing import Callable

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from sbi.utils.sbiutils import seed_all_backends

from loss_cal.costs import RevGaussCost, StepCost_weighted
from loss_cal.predictor import build_nn, train
from loss_cal.tasks import get_task
from loss_cal.utils.utils import load_data, prepare_for_training, save_metadata


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
        "linear_gaussian",
    ], "Choose one of 'toy_example', 'sir', 'lotka_volterra' or  'linear_gaussian."
    task = get_task(task_name=task_name)

    model = cfg.model.type
    assert model in ["fc"], "Model type should be 'fc'."

    seed = cfg.seed
    seed_all_backends(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load configs
    parameter = cfg.task.parameter
    aligned = True
    offset = 0
    if parameter == "None":
        parameter = None
        if task_name == "sir":
            print("Fix parameter range to R0 specific range (4) and set aligned=False.")
            parameter_range = 4.0
            aligned = False
            offset = 1
        elif task_name == "linear_gaussian":
            parameter_range = (task.param_high[0] - task.param_low[0]).squeeze().item()
            aligned = True
            offset = torch.abs(task.param_high[0])
        else:
            print("Task not defined without specified parameter!")
            raise (NotImplementedError)
    else:
        parameter_range = (task.param_high[parameter] - task.param_low[parameter]).squeeze().item()
    print("parameter_range", parameter_range)
    action_range = task.action_high - task.action_low
    action_type = cfg.action.type
    stop_after_epochs = cfg.model.stop_after_epochs

    if action_type == "binary":
        threshold = round(cfg.action.T, ndigits=4)
        costs = list(cfg.action.costs)
        cost_fn = StepCost_weighted(costs, threshold=threshold)
    elif action_type == "continuous":
        factor = cfg.action.factor
        exponential = float(cfg.action.exponential)
        cost_fn = RevGaussCost(
            parameter_range=parameter_range,
            action_range=action_range,
            factor=factor,
            exponential=exponential,
            aligned=aligned,
            offset=offset,
        )
    else:
        raise NotImplementedError(f"Provided type of action '{action_type}' not defined.")

    str_to_tranformation = {"ReLU": torch.nn.ReLU(), "Sigmoid": torch.nn.Sigmoid(), "Identity": torch.nn.Identity()}
    assert (
        cfg.model.output_transform in str_to_tranformation.keys()
    ), "Output transformation not implemented, one if ['ReLU', 'Sigmoid', 'Identity']."
    assert (
        cfg.model.activation in str_to_tranformation.keys()
    ), "Output transformation not implemented, one if ['ReLU', 'Sigmoid', 'Identity']."

    hidden_layers = list(cfg.model.hidden)
    output_transform = str_to_tranformation[cfg.model.output_transform]
    activation_fn = str_to_tranformation[cfg.model.activation]
    learning_rate = cfg.model.lr
    z_scoring = cfg.model.zscore
    epochs = None if cfg.model.epochs == "None" else int(cfg.model.epochs)

    # load data
    ntrain = cfg.ntrain
    x_train, theta_train, x_val, theta_val = load_data_for_training(
        data_dir=cfg.data_dir,
        task_name=task_name,
        ntrain=ntrain,
        parameter=parameter,
        device=device,
    )

    # create directory & save metadata
    save_dir = path.join(
        cfg.res_dir,
        task_name,
        action_type,
        "nn",
        model,
        cfg.experiment,
        seed
        #  " ".join([str(cfg.model.hidden).replace(", ", "-")[1:-1], cfg.model.output_transform]),
    )
    model_dir = prepare_for_training(
        base_dir=save_dir,
        parameter=parameter,
        action_type=action_type,
        action_parameters=(round(threshold, ndigits=4), costs) if action_type == "binary" else (factor, exponential),
    )
    save_metadata(
        model_dir,
        model=model,
        input=x_train.shape[1],
        hidden_layers=hidden_layers,
        output_transform=output_transform,
        activation=activation_fn,
        z_scoring=z_scoring,
        parameter=parameter,
        action_type=action_type,
        action_parameters=(threshold, costs) if action_type == "binary" else (factor, exponential),
        seed=seed,
        lr=learning_rate,
        ntrain=ntrain,
        stop_after_epochs=stop_after_epochs,
        epochs=epochs,
        data_dir=path.join(cfg.data_dir, task_name),
    )

    # training
    print(f"save model at: {save_dir}\ndevice: {device}")

    nn = build_nn(
        model="fc",
        x_train=x_train,
        action_train=task.actions.sample(x_train.shape[0]),  # sample actions to initialize Standardize layer
        hidden_dims=hidden_layers,
        output_dim=1,
        activation=activation_fn,
        output_transform=output_transform,
        z_scoring=z_scoring,
    )
    print(nn, end="\n-----\n")

    nn, loss_values_train, loss_values_val = train(
        model=nn,
        x_train=x_train,
        theta_train=theta_train,
        cost_fn=cost_fn,
        x_val=x_val,
        theta_val=theta_val,
        actions=task.actions,
        learning_rate=learning_rate,
        stop_after_epochs=stop_after_epochs,
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


def load_data_for_training(
    data_dir: str,
    task_name: str,
    ntrain: int,
    parameter: int,
    device: str,
):
    task = get_task(task_name=task_name)

    (
        theta_train,
        x_train,
        theta_val,
        x_val,
        _,
        _,
    ) = load_data(task_name=task_name, base_dir=data_dir, device=device)

    theta_train = task.param_aggregation(theta_train)
    theta_val = task.param_aggregation(theta_val)

    if parameter is not None and parameter >= 0 and parameter <= theta_train.shape[1] - 1:
        print(f"Restrict parameters to parameter {parameter}.")
        theta_train = theta_train[:, parameter : parameter + 1]
        theta_val = theta_val[:, parameter : parameter + 1]

    if ntrain > theta_train.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")
    elif ntrain < theta_train.shape[0]:
        theta_train = theta_train[:ntrain]
        x_train = x_train[:ntrain]

    # DOUBLE DATA TO ACCOMODATE BINARY ACTIONS
    # if action_type == "binary":
    #     costs_train = torch.concat(
    #         [
    #             cost_fn(theta_train, 0),
    #             cost_fn(theta_train, 1),
    #         ]
    #     )

    #     actions_train = torch.concat([torch.zeros((theta_train.shape[0], 1)), torch.ones((theta_train.shape[0], 1))])
    #     theta_train = theta_train.repeat(2, 1)
    #     x_train = x_train.repeat(2, 1)
    #     costs_val = torch.concat(
    #         [
    #             cost_fn(theta_val, 0),
    #             cost_fn(theta_val, 1),
    #         ]
    #     )
    #     actions_val = torch.concat([torch.zeros((theta_val.shape[0], 1)), torch.ones((theta_val.shape[0], 1))])
    #     theta_val = theta_val.repeat(2, 1)
    #     x_val = x_val.repeat(2, 1)

    print("Data shapes train (x, theta):", x_train.shape, theta_train.shape)
    print("Data shapes val (x,theta):", x_val.shape, theta_val.shape)

    return x_train, theta_train, x_val, theta_val


if __name__ == "__main__":
    main()
