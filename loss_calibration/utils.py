from os import mkdir, path
from datetime import datetime
import torch


def prepare_for_training(base_dir, threshold, costs):
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


def load_data(task_name, base_dir="./data"):
    dir = path.join(base_dir, task_name)
    try:
        theta_train = torch.load(path.join(dir, "theta_train.pt"))
        x_train = torch.load(path.join(dir, "x_train.pt"))
        theta_val = torch.load(path.join(dir, "theta_val.pt"))
        x_val = torch.load(path.join(dir, "x_val.pt"))
        theta_test = torch.load(path.join(dir, "theta_test.pt"))
        x_test = torch.load(path.join(dir, "x_test.pt"))
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
