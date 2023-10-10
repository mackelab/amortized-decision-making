import json
from datetime import datetime
from os import getcwd, mkdir, path
from typing import Callable, Optional, Tuple

import torch


def load_data(
    task_name: str,
    base_dir: str = "./data",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load training, validation and test data from folder.

    Args:
        task_name (str): Name of the task, used as folder name.
        base_dir (str, optional): Base directory where data is stored in folder 'task_name'. Defaults to "./data".

    Returns:
        Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]: training, validationa and test data (theta and x)
    """
    dir = path.join(base_dir, task_name)
    print(f"Load data from '{dir}', device = {device}.")
    try:
        theta_train = torch.load(path.join(dir, "theta_train.pt"), map_location=device)
        x_train = torch.load(path.join(dir, "x_train.pt"), map_location=device)
        theta_val = torch.load(path.join(dir, "theta_val.pt"), map_location=device)
        x_val = torch.load(path.join(dir, "x_val.pt"), map_location=device)
        theta_test = torch.load(path.join(dir, "theta_test.pt"), map_location=device)
        x_test = torch.load(path.join(dir, "x_test.pt"), map_location=device)
    except FileNotFoundError:
        print("Data not found, check path or generate data first.")
        print(f"Current path: {getcwd()}, provided path to data: {dir}")
    return theta_train, x_train, theta_val, x_val, theta_test, x_test


def save_data(
    task_name: str,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    theta_val: torch.Tensor,
    x_val: torch.Tensor,
    theta_test: torch.Tensor,
    x_test: torch.Tensor,
    base_dir: str = "./data",
):
    """Save data at provided directory

    Args:
        task_name (str): Name of task, used as name of the folder
        theta_train (torch.Tensor): training data - parameters
        x_train (torch.Tensor): training data - observations
        theta_val (torch.Tensor): validation data - parameters
        x_val (torch.Tensor): validation data - observations
        theta_test (torch.Tensor): test data - parameters
        x_test (torch.Tensor): test data - observations
        base_dir (str, optional): Base directory where to save folder with data. Defaults to "./data".
    """
    dir = path.join(base_dir, task_name)
    torch.save(theta_train, path.join(dir, "theta_train.pt"))
    torch.save(x_train, path.join(dir, "x_train.pt"))
    torch.save(theta_val, path.join(dir, "theta_val.pt"))
    torch.save(x_val, path.join(dir, "x_val.pt"))
    torch.save(theta_test, path.join(dir, "theta_test.pt"))
    torch.save(x_test, path.join(dir, "x_test.pt"))
    print(f"Saved training, test and vailadation data at: {dir}")


def prepare_for_training(
    base_dir: str,
    parameter: int,
    action_type: str,
    action_parameters: Tuple,
) -> str:
    """Create directory with timestamp to save model

    Args:
        base_dir (str): Base directory to save model
        threshold (float): treshold
        costs (list): list of costs of classification

    Returns:
        str: _description_
    """
    timestamp = datetime.now().isoformat().split(".")[0].replace(":", "_")
    assert action_type in {"binary", "continuous"}, "Type of actions has to be one of 'binary' and 'continuous'."
    to_str = (
        lambda n: str(int(n)) if type(n) == int or (type(n) == float and n.is_integer()) else str(n).replace(".", "_")
    )
    if action_type == "binary":
        threshold, costs = action_parameters
        model_dir = path.join(
            base_dir,
            f"{timestamp}_param_{parameter}_threshold_{to_str(threshold)}_costs_{to_str(costs[0])}_{to_str(costs[1])}",
        )
    else:
        factor, exponential = action_parameters
        model_dir = path.join(
            base_dir,
            f"{timestamp}_param_{parameter}_factor_{to_str(factor)}_exp_{to_str(exponential)}",
        )
    create_filestructure(model_dir=model_dir)
    return model_dir


def create_filestructure(model_dir: str):
    """Create directory for model and checkpoints

    Args:
        model_dir (str): Path to save the model
    """
    print(f"Current directory: '{getcwd()}'")
    try:
        mkdir(model_dir)
        mkdir(path.join(model_dir, "checkpoints/"))
        print(f"Created directory {model_dir}.")
    except FileExistsError:
        print(f"Directory {model_dir} already exists.")


def create_checkpoint_dir(model_dir: str):
    check_base_dir_exists(model_dir)
    try:
        mkdir(path.join(model_dir, "checkpoints/"))
        print(f"Created directory '{path.join(model_dir, 'checkpoints/')}'.")
    except FileExistsError:
        print(f"Subdirectory 'checkpoints' already exists. Delete first if wanted.")


def create_seed_dir(model_dir: str, seed: int):
    check_base_dir_exists(model_dir)
    new_dir = path.join(model_dir, str(seed))
    try:
        mkdir(new_dir)
        print(f"Created directory '{new_dir}'.")
    except FileExistsError:
        print(f"Subdirectory for seed {seed} already exists. Delete first if wanted.")
    return new_dir


def save_metadata(
    model_dir: str,
    model: str,
    input: int,
    hidden_layers: list,
    output_transform: Callable,
    activation: Callable,
    z_scoring: str,
    parameter: int,
    action_type: str,
    action_parameters: Tuple,
    seed: int,
    lr: float,
    ntrain: int,
    stop_after_epochs: int,
    epochs: int,
    data_dir=str,
):
    assert action_type in ["binary", "continuous"], "Specifiy the type of actions, one of 'binary' or 'continuous'."

    metadata = {
        "seed": seed,
        "model": model,
        "architecture": f"{input}-{'-'.join(map(str, hidden_layers))}-1",
        "output_transform": str(output_transform)[:-2],  # remove brackets
        "activation": str(activation)[:-2],  # remove brackets
        "z_scoring": z_scoring,
        "optimizer": "Adam",
        "learning_rate": lr,
        "ntrain": ntrain,
        "stop_after_epochs": stop_after_epochs,
        "parameter": parameter,
        "max_num_epochs": epochs,
        "data_dir": data_dir,
    }

    if action_type == "binary":
        T, costs = action_parameters
        metadata.update({"threshold": T, "costs": costs})
    if action_type == "continuous":
        factor, exponential = action_parameters
        metadata.update({"factor": factor, "exponential": exponential})

    json.dump(metadata, open(f"{model_dir}/metadata.json", "w"))


def check_base_dir_exists(base_dir: str):
    """Check if directory exists

    Args:
        base_dir (str): Base directory to save model
        threshold (float): treshold
        costs (list): list of costs of classification

    Returns:
        str: _description_
    """
    assert path.isdir(base_dir), "base_dir is no existing directory"


def atleast_2d_col(x: torch.Tensor):
    if x.ndim < 2:
        x = x.reshape(-1, 1)
    return x


def posterior_ratio_given_samples(posterior_samples: torch.Tensor, treshold: float, costs: list) -> float:
    """Compute posterior ratio based on samples.

    Args:
        posterior_samples (torch.Tensor): samples from the posterior
        treshold (float): treshold used for decision making
        costs (list): costs of classification

    Returns:
        float: posterior ratio
    """
    cost_fn = (posterior_samples > treshold).sum() * costs[0]
    cost_fp = (posterior_samples < treshold).sum() * costs[1]
    return cost_fn / (cost_fn + cost_fp)
