from email.mime import base
from os import mkdir, path
from datetime import datetime
import torch
import json
from typing import Tuple


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


def posterior_ratio_given_obs(
    task,
    n_obs: int,
    idx_parameter: int,
    threshold: float,
    costs: list,
) -> float:
    """Computer posterior ratio given index of the reference observation from sbibm

    Args:
        task (): sbibm task object
        n_obs (int): number of observation
        idx_parameter (int): index of the parameter used for decision making
        threshold (float): threshold used for decision making
        costs (list): costs of classification

    Returns:
        float: _description_
    """
    assert n_obs in range(1, 11)
    posterior_samples = task.get_reference_posterior_samples(n_obs)[:, idx_parameter]
    cost_fn = (posterior_samples > threshold).sum() * costs[0]
    cost_fp = (posterior_samples < threshold).sum() * costs[1]
    return cost_fn / (cost_fn + cost_fp)


def prepare_for_training(
    base_dir: str,
    parameter: int,
    threshold: float,
    costs: list,
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
    model_dir = path.join(
        base_dir,
        f"{timestamp}_t{parameter}_{str(threshold).replace('.', '_')}_c{int(costs[0])}_{int(costs[1])}",
    )
    prepare_filestructure(model_dir)
    return model_dir


def prepare_filestructure(model_dir: str):
    """Create directory for model and checkpoints

    Args:
        model_dir (str): Path to save the model
    """
    try:
        mkdir(model_dir)
        mkdir(path.join(model_dir, "checkpoints/"))
        print(f"Created directory {model_dir}.")
    except FileExistsError:
        print(f"Directory {model_dir} already exists.")


def save_metadata(
    model_dir: str,
    model: str,
    input: int,
    hidden_layers: list,
    z_scoring: str,
    costs: list,
    T: float,
    seed: int,
    lr: float,
    ntrain: int,
    epochs: int,
    data_dir=str,
):
    metadata = {
        "seed": seed,
        "model": model,
        "architecture": f"{input}-{'-'.join(map(str, hidden_layers))}-1",
        "z_scoring": z_scoring,
        "optimizer": "Adam",
        "learning_rate": lr,
        "Ntrain": ntrain,
        "threshold": T,
        "costs": costs,
        "max_num_epochs": epochs,
        "data_dir": data_dir,
    }

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


def load_data(
    task_name: str,
    base_dir: str = "./data",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Load training, validation and test data from folder.

    Args:
        task_name (str): Name of the task, used as folder name.
        base_dir (str, optional): Base directory where data is stored in folder 'task_name'. Defaults to "./data".

    Returns:
        Tuple[ torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor ]: training, validationa and test data (theta and x)
    """
    dir = path.join(base_dir, task_name)
    try:
        theta_train = torch.load(path.join(dir, "theta_train.pt"), map_location=device)
        x_train = torch.load(path.join(dir, "x_train.pt"), map_location=device)
        theta_val = torch.load(path.join(dir, "theta_val.pt"), map_location=device)
        x_val = torch.load(path.join(dir, "x_val.pt"), map_location=device)
        theta_test = torch.load(path.join(dir, "theta_test.pt"), map_location=device)
        x_test = torch.load(path.join(dir, "x_test.pt"), map_location=device)
        print(f"Load data from '{dir}', device = {device}.")
        return theta_train, x_train, theta_val, x_val, theta_test, x_test
    except FileNotFoundError:
        print("Data not found, check path or generate data first.")


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


def format_axis(ax, xhide=True, yhide=True, xlabel="", ylabel="", tickformatter=None):
    for loc in ["right", "top", "left", "bottom"]:
        ax.spines[loc].set_visible(False)
    if xhide:
        ax.set_xlabel("")
        ax.xaxis.set_ticks_position("none")
        ax.xaxis.set_tick_params(labelbottom=False)
    if yhide:
        ax.set_ylabel("")
        ax.yaxis.set_ticks_position("none")
        ax.yaxis.set_tick_params(labelleft=False)
    if not xhide:
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position("bottom")
        ax.xaxis.set_tick_params(labelbottom=True)
        if tickformatter is not None:
            ax.xaxis.set_major_formatter(tickformatter)
        ax.spines["bottom"].set_visible(True)
    if not yhide:
        ax.set_ylabel(ylabel)
        ax.yaxis.set_ticks_position("left")
        ax.yaxis.set_tick_params(labelleft=True)
        if tickformatter is not None:
            ax.yaxis.set_major_formatter(tickformatter)
        ax.spines["left"].set_visible(True)
    return ax


def raw_stats_given_predictions(predictions, theta_test, threshold):
    d_test = (theta_test > threshold).float()
    d_predicted = (predictions > 0.5).float()
    tn = torch.logical_and(d_predicted == 0, d_test == 0)
    tp = torch.logical_and(d_predicted == 1, d_test == 1)
    fn = torch.logical_and(d_predicted == 0, d_test == 1)
    fp = torch.logical_and(d_predicted == 1, d_test == 0)
    return tp, fn, fp, tn


def stats_given_predictions(predictions, theta_test, threshold):
    assert predictions.shape == theta_test.shape
    tp, fn, fp, tn = raw_stats_given_predictions(predictions, theta_test, threshold)
    acc = (tp.sum() + tn.sum()) / (tp.sum() + fn.sum() + fp.sum() + tn.sum())
    return tp.sum(), fn.sum(), fp.sum(), tn.sum(), acc
