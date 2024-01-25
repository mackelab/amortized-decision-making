import json
from datetime import datetime
from os import getcwd, mkdir, path
from typing import Callable, Tuple
from pathlib import Path

import torch


def load_data(
    task_name: str,
    base_dir: str = "./data",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Load training and test data from folder.

    Args:
        task_name (str): Name of the task, used as folder name.
        base_dir (str, optional): Base directory where data is stored in folder 'task_name'. Defaults to "./data".

    Returns:
        Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]: training and test data (theta and x)
    """
    dir = path.join(base_dir, task_name)
    print(f"Load data from '{dir}', device = {device}.")
    try:
        theta_train = torch.load(path.join(dir, "theta_train.pt"), map_location=device)
        x_train = torch.load(path.join(dir, "x_train.pt"), map_location=device)
        theta_test = torch.load(path.join(dir, "theta_test.pt"), map_location=device)
        x_test = torch.load(path.join(dir, "x_test.pt"), map_location=device)
    except FileNotFoundError:
        print("Data not found, check path or generate data first.")
        print(f"Current path: {getcwd()}, provided path to data: {dir}")

    return (
        theta_train,
        x_train,
        theta_test,
        x_test,
    )


def save_data(
    task_name: str,
    theta_train: torch.Tensor,
    x_train: torch.Tensor,
    theta_test: torch.Tensor,
    x_test: torch.Tensor,
    base_dir: str = "./data",
):
    """Save data at provided directory

    Args:
        task_name (str): Name of task, used as name of the folder
        theta_train (torch.Tensor): training data - parameters
        x_train (torch.Tensor): training data - observations
        theta_test (torch.Tensor): test data - parameters
        x_test (torch.Tensor): test data - observations
        base_dir (str, optional): Base directory where to save folder with data. Defaults to "./data".
    """
    dir = path.join(base_dir, task_name)
    torch.save(theta_train, path.join(dir, "theta_train.pt"))
    torch.save(x_train, path.join(dir, "x_train.pt"))
    torch.save(theta_test, path.join(dir, "theta_test.pt"))
    torch.save(x_test, path.join(dir, "x_test.pt"))
    print(f"Saved training, test and vailadation data at: {dir}")


def prepare_for_training(
    base_dir: str,
    ntrain: int,
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
    assert action_type in {
        "discrete",
        "continuous",
    }, "Type of actions has to be one of 'discrete' and 'continuous'."
    to_str = (
        lambda n: str(int(n))
        if type(n) == int or (type(n) == float and n.is_integer())
        else str(n).replace(".", "_")
    )
    if action_type == "discrete":
        threshold, costs = action_parameters
        model_dir = path.join(
            base_dir,
            f"{timestamp}_n{ntrain}_param_{parameter}_threshold_{'_'.join([to_str(t) for t in threshold])}_costs_{'_'.join([to_str(c) for c in costs])}",
        )
    else:
        factor, exponential = action_parameters
        model_dir = path.join(
            base_dir,
            f"{timestamp}_n{ntrain}_param_{parameter}_factor_{to_str(factor)}_exp_{to_str(exponential)}",
        )
    create_filestructure(model_dir=model_dir)
    return model_dir


def create_filestructure(model_dir: str, print_cwd=True):
    """Create directory for model and checkpoints

    Args:
        model_dir (str): Path to save the model
    """
    if print_cwd:
        print(f"Current directory: '{getcwd()}'")
    if Path(model_dir).is_dir():
        print(
            f"Warning: Directory {model_dir} already exists, files might get overwritten."
        )
    else:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        print(f"Created directory '{model_dir}'.")


def create_checkpoint_dir(model_dir: str):
    """Create a checkpoint folder at provided directory

    Args:
        model_dir (str): root path of model
    """
    check_base_dir_exists(model_dir)
    checkpoint_dir = path.join(model_dir, "checkpoints/")
    create_filestructure(checkpoint_dir, print_cwd=False)


def create_seed_dir(model_dir: str, seed: int):
    """Create a folder for the seed at provided directory

    Args:
        model_dir (str): base directory
        seed (int): seed

    Returns:
        str: directory of newly created seed folder
    """
    check_base_dir_exists(model_dir)
    seed_dir = path.join(model_dir, str(seed))
    create_filestructure(seed_dir, print_cwd=False)


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
    num_action_samples_train: int,
    num_action_samples_val: int,
    sample_in_loop: bool,
    seed: int,
    lr: float,
    ntrain: int,
    stop_after_epochs: int,
    epochs: int,
    data_dir=str,
):
    """save metadata of model"""
    assert action_type in [
        "discrete",
        "continuous",
    ], "Specifiy the type of actions, one of 'discrete' or 'continuous'."

    metadata = {
        "seed": seed,
        "model": model,
        "architecture": f"{input}-{'-'.join(map(str, hidden_layers))}-1",
        "output_transform": str(output_transform).split("(")[0],  # remove brackets
        "activation": str(activation).split("(")[0],  # remove brackets
        "z_scoring": z_scoring,
        "optimizer": "Adam",
        "learning_rate": lr,
        "ntrain": ntrain,
        "stop_after_epochs": stop_after_epochs,
        "num_action_samples_train": num_action_samples_train,
        "num_action_samples_val": num_action_samples_val,
        "sample_in_loop": sample_in_loop,
        "parameter": parameter,
        "max_num_epochs": epochs,
        "data_dir": data_dir,
    }

    if action_type == "discrete":
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
    """turn tensor into 2d column vector if not already 2d"""
    if x.ndim < 2:
        x = x.reshape(-1, 1)
    return x


def posterior_ratio_given_samples(
    posterior_samples: torch.Tensor, treshold: float, costs: list
) -> float:
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
