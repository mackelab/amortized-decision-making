from os import mkdir, path
from datetime import datetime
import torch
import json


def prepare_for_training(
    base_dir,
    threshold,
    costs,
):
    timestamp = datetime.now().isoformat().split(".")[0].replace(":", "_")
    model_dir = path.join(
        base_dir,
        f"{timestamp}_t{str(threshold).replace('.', '_')}_c{int(costs[0])}_{int(costs[1])}",
    )
    prepare_filestructure(model_dir)
    return model_dir


def prepare_filestructure(model_dir):
    try:
        mkdir(model_dir)
        mkdir(path.join(model_dir, "checkpoints/"))
        print(f"Created directory {model_dir}.")
    except FileExistsError:
        print(f"Directory {model_dir} already exists.")


def save_metadata(model_dir, input, hidden_layers, costs, T, seed, lr, ntrain):
    metadata = {
        "seed": seed,
        "architecture": f"{input}-{'-'.join(map(str, hidden_layers))}-1",
        "optimizer": "Adam",
        "learning_rate": lr,
        "Ntrain": ntrain,
        "threshold": T,
        "costs": costs,
    }

    json.dump(metadata, open(f"{model_dir}/metadata.json", "w"))


def load_data(task_name, base_dir="./data"):
    dir = path.join(base_dir, task_name)
    try:
        theta_train = torch.load(path.join(dir, "theta_train.pt"))
        x_train = torch.load(path.join(dir, "x_train.pt"))
        theta_val = torch.load(path.join(dir, "theta_val.pt"))
        x_val = torch.load(path.join(dir, "x_val.pt"))
        theta_test = torch.load(path.join(dir, "theta_test.pt"))
        x_test = torch.load(path.join(dir, "x_test.pt"))
        print(f"Load data from '{dir}'.")
        return theta_train, x_train, theta_val, x_val, theta_test, x_test
    except FileNotFoundError:
        print("Data not found, check path or generate data first.")


def save_data(
    task_name,
    theta_train,
    x_train,
    theta_val,
    x_val,
    theta_test,
    x_test,
    base_dir="./data",
):
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
