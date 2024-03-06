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

from bam.costs import MultiClass01Cost, MultiClassStepCost, RevGaussCost
from bam.bam import build_nn, train
from bam.tasks import get_task
from bam.utils.utils import load_data, prepare_for_training, save_metadata


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
        "bvep",
    ], "Choose one of 'toy_example', 'sir', 'lotka_volterra', 'linear_gaussian' or 'bvep'."
    action_type = cfg.action.type
    if action_type == "discrete":
        num_actions = (
            None if cfg.action.num_actions == "None" else int(cfg.action.num_actions)
        )
        probs = None if cfg.action.probs == "None" else list(cfg.action.probs)
        task_specifications = {
            "action_type": action_type,
            "num_actions": num_actions,
            "probs": probs,
        }
    else:
        task_specifications = {"action_type": action_type}

    task = get_task(task_name=task_name, **task_specifications)

    model = cfg.model.type
    assert model in ["fc"], "Model type should be 'fc'."

    seed = int(cfg.seed)
    seed_all_backends(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load configs
    parameter = cfg.task.parameter if cfg.task.parameter != "None" else None
    if parameter == None:
        if task_name == "sir":
            print(
                "Fix parameter range to R0 specific range (23.8148, based on 99.99% quantile of training data), aligned=False, offset=1.0."
            )
            parameter_range = [0.0, 23.8148]
        elif task_name == "linear_gaussian":
            parameter_range = [
                task.param_low[0].squeeze().item(),
                task.param_high[0].squeeze().item(),
            ]
        else:
            print("Task not defined without specified parameter!")
            raise (NotImplementedError)
    else:
        parameter_range = [
            task.param_low[parameter].squeeze().item(),
            task.param_high[parameter].squeeze().item(),
        ]
    print("parameter_range", parameter_range)
    stop_after_epochs = cfg.model.stop_after_epochs

    if action_type == "discrete":
        theta_crit = torch.Tensor(list(cfg.action.T))
        costs = [1.0] * theta_crit.shape[0]
        cost_fn = MultiClass01Cost(theta_crit=theta_crit)
        # factors = torch.ones(num_actions, num_actions)
        # factors[0, 2] = 10.0
        # print("factors (BVEP specific!)", factors)
        # cost_fn = MultiClassStepCost(theta_crit=theta_crit, factors=factors)
    elif action_type == "continuous":
        factor = cfg.action.factor
        exponential = float(cfg.action.exponential)
        aligned = cfg.action.aligned
        offset = cfg.action.offset
        action_range = [task.action_low, task.action_high]
        cost_fn = RevGaussCost(
            parameter_range=parameter_range,
            action_range=action_range,
            factor=factor,
            exponential=exponential,
            aligned=aligned,
            offset=offset,
        )
    else:
        raise NotImplementedError(
            f"Provided type of action '{action_type}' not defined."
        )

    str_to_tranformation = {
        "ReLU": torch.nn.ReLU(),
        "Sigmoid": torch.nn.Sigmoid(),
        "Identity": torch.nn.Identity(),
        "Softplus": torch.nn.Softplus(),
    }
    assert (
        cfg.model.output_transform in str_to_tranformation.keys()
    ), "Output transformation not implemented, one if ['ReLU', 'Sigmoid', 'Identity', 'Softplus']."
    assert (
        cfg.model.activation in str_to_tranformation.keys()
    ), "Output transformation not implemented, one if ['ReLU', 'Sigmoid', 'Identity', 'Softplus']."

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
        task=task,
        ntrain=ntrain,
        parameter=parameter,
        device=device,
    )
    num_action_samples_train = cfg.num_action_samples_train
    num_action_samples_val = cfg.num_action_samples_val
    sample_actions_in_loop = cfg.sample_in_loop

    print("data shapes (train): ", x_train.shape, theta_train.shape)
    print("data shapes (val): ", x_val.shape, theta_val.shape)

    # create directory & save metadata
    save_dir = path.join(
        cfg.res_dir,
        task_name,
        action_type,
        "nn",
        model,
        cfg.experiment,
        str(seed),
        f"{num_action_samples_train}actions" if not sample_actions_in_loop else "inloop"
        #  " ".join([str(cfg.model.hidden).replace(", ", "-")[1:-1], cfg.model.output_transform]),
    )
    print("Saving model at:", save_dir)

    model_dir = prepare_for_training(
        base_dir=save_dir,
        ntrain=ntrain,
        parameter=parameter,
        action_type=action_type,
        action_parameters=([round(t, ndigits=4) for t in theta_crit.tolist()], costs)
        if action_type == "discrete"
        else (factor, exponential),
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
        action_parameters=([round(t, ndigits=4) for t in theta_crit.tolist()], costs)
        if action_type == "discrete"
        else (factor, exponential),
        num_action_samples_train=num_action_samples_train,
        num_action_samples_val=num_action_samples_val,
        sample_in_loop=sample_actions_in_loop,
        seed=seed,
        lr=learning_rate,
        ntrain=ntrain,
        stop_after_epochs=stop_after_epochs,
        epochs=epochs,
        data_dir=path.join(cfg.data_dir, task_name),
    )

    # training
    nn = build_nn(
        model="fc",
        x_train=x_train.to(device),
        action_train=task.actions.sample(x_train.shape[0]).to(
            device
        ),  # sample actions to initialize Standardize layer
        hidden_dims=hidden_layers,
        output_dim=1,
        activation=activation_fn,
        output_transform=output_transform,
        z_scoring=z_scoring,
        seed=seed,
    )
    print(nn, end="\n-----\n")

    nn, loss_values_train, loss_values_val = train(
        model=nn,
        x_train=x_train.to(device),
        theta_train=theta_train.to(device),
        cost_fn=cost_fn,
        x_val=x_val.to(device),
        theta_val=theta_val.to(device),
        actions=task.actions,
        num_action_samples_train=num_action_samples_train,
        num_action_samples_val=num_action_samples_val,
        sample_actions_in_loop=sample_actions_in_loop,
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
    task: str,
    ntrain: int,
    parameter: int,
    validation_fraction: float = 0.1,
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    (
        theta,
        x,
        _,
        _,
    ) = load_data(task_name=task.task_name, base_dir=data_dir, device=device)

    # TRAIN-VAL SPLIT (90:10)

    # Get total number of training examples.
    num_examples = ntrain
    # Select random train and validation splits from (theta, x) pairs.
    num_training_examples = int((1 - validation_fraction) * num_examples)
    num_validation_examples = num_examples - num_training_examples

    if ntrain > theta.shape[0]:
        raise ValueError("Not enough samples available, create a new dataset first.")

    permuted_indices = torch.randperm(num_examples)
    train_indices, val_indices = (
        permuted_indices[:num_training_examples],
        permuted_indices[num_training_examples:],
    )

    theta_train = theta[train_indices]
    x_train = x[train_indices]
    theta_val = theta[val_indices]
    x_val = x[val_indices]

    theta_train = task.param_aggregation(theta_train)
    theta_val = task.param_aggregation(theta_val)

    if (
        parameter is not None
        and parameter >= 0
        and parameter <= theta_train.shape[1] - 1
    ):
        print(f"Restrict parameters to parameter {parameter}.")
        theta_train = theta_train[:, parameter : parameter + 1]
        theta_val = theta_val[:, parameter : parameter + 1]

    print("Data shapes train (x, theta):", x_train.shape, theta_train.shape)
    print("Data shapes val (x,theta):", x_val.shape, theta_val.shape)

    return x_train, theta_train, x_val, theta_val


if __name__ == "__main__":
    main()
