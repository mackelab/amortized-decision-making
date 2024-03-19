from os import path
import glob
import json
from copy import deepcopy

from typing import Callable, Dict, Iterable, Optional, Tuple, Union

import torch

from sbi.utils.sbiutils import Standardize, seed_all_backends
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bam.actions import Action
from bam.tasks import get_task
from bam.utils.utils import atleast_2d_col, create_checkpoint_dir, load_data

# from loss_cal.costs import BCELoss_weighted


class FeedforwardNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable,
        output_dim: int,
        activation: Callable = nn.Sigmoid(),
        output_transform: Callable = nn.Identity(),
        mean: Union[torch.Tensor, float] = None,
        std: Union[torch.Tensor, float] = None,
        z_scoring: Optional[str] = "Independent",
    ):
        """Initializa a feedforward neural network.

        Args:
            input_dim (int): dimoension of the input
            hidden_dims (Iterable): list of hidden dimensions
            output_dim (int): dimension of the output
            activation (Callable, optional): Activation function. Defaults to nn.Sigmoid().
            output_transform (Callable, optional): Tranformation of the output of the final layer. Defaults to nn.Identity().
            mean (Union[torch.Tensor, float], optional): Provide mean for z-scoring. Defaults to None.
            std (Union[torch.Tensor, float], optional): Provide standard deviation for z-scoring. Defaults to None.
            z_scoring (Optional[str], optional): Type of z-scoring, one of ['None', 'Independent', 'Structured']. Defaults to "Independent".

        Raises:
            ValueError: In case no hidden dimension is given.
            ValueError: Invalid x-scoring option, must be one of 'None', 'Independent', 'Structured'
        """
        if len(hidden_dims) == 0:
            raise ValueError(
                "Specify at least one hidden layer, list of hidden dims can't be empty."
            )

        if z_scoring.capitalize() in ["Independent", "Structured"]:
            assert (
                mean is not None and std is not None
            ), "Provide mean and standard deviation for z-scoring."
            z_scoring_bool = True
        elif z_scoring.capitalize() == "None":
            z_scoring_bool = False
        else:
            raise ValueError(
                "Invalid z-scoring opion, use 'None', 'Independent', 'Structured'."
            )

        super(FeedforwardNN, self).__init__()

        self.in_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = output_dim
        self.z_scored = z_scoring_bool

        # Layer
        if z_scoring_bool:
            self.standardize_layer = Standardize(mean, std)
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(in_dim, out_dim)
                for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:])
            ]
        )
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Activation function
        self.activation = activation
        self.output_transform = output_transform

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        """Forward pass of the neural network

        Args:
            x (torch.Tensor): observation
            a (torch.Tensor): action

        Raises:
            ValueError: input dimension has to match dimension of concat(x,a)

        Returns:
            torch.Tensor: prediction of size self.out_dim
        """
        x = atleast_2d_col(x)
        a = atleast_2d_col(a)
        assert (
            x.shape[0] == a.shape[0]
        ), "Observations and actions should share the first dimension, i.e. same number of samples."
        if x.shape[1] != self.in_dim - a.shape[1]:
            raise ValueError(
                f"Expected inputs of dim {self.in_dim - a.shape[1]}, got {x.shape[1]}."
            )

        # concatenate x and a to form input
        input = torch.concat([x, a], dim=1)

        if self.z_scored:
            input = self.standardize_layer(input)
        out = self.input_layer(input)
        out = self.activation(out)

        for i, layer in enumerate(self.hidden_layers):
            out = layer(out)
            # out = nn.BatchNorm1d(out)
            out = self.activation(out)

        out = self.final_layer(out)
        out = self.output_transform(out)

        return out


def get_mean_and_std(z_scoring: str, data_train: torch.Tensor, eps: float = 1e-14):
    """Compute mean and standard deviatiation frome the training data/

    Args:
        z_scoring (str): Type of z-scoring, use one of ["None", "Independent", "Structured"]
        x_train (torch.Tensor): training data
        min_std (float, optional): Lower cap of the standard deviation. Defaults to 1e-14.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: mean and standard deviation
    """
    assert z_scoring.capitalize() in ["None", "Independent", "Structured"]
    if z_scoring == "Independent":
        std = data_train.std(dim=0)
        # set small std to 1., not eps (https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70)
        std[std < eps] = 1.0
        return data_train.mean(dim=0), std
    elif z_scoring == "Structured":
        std = data_train.std()
        std[std < eps] = eps
        return data_train.mean(), std
    else:
        return torch.zeros(1), torch.ones(1)


def build_nn(
    model: str,
    x_train: Union[torch.Tensor, int],
    action_train: Union[torch.Tensor, int],
    hidden_dims: Iterable,
    output_dim: int,
    activation: Callable = nn.Sigmoid(),
    output_transform: Callable = nn.Identity(),
    z_scoring: Optional[str] = "Independent",
    mean=Optional[float],
    std=Optional[float],
    seed=0,
):
    """Wrapper function to prepare the z-scoring and adapt the input size to the training data

    Args:
        model (str): Type of model, currently only "fc"=fully connected supported.
        x_train (Union[torch.Tensor, int]): observations
        action_train (Union[torch.Tensor, int]): actions
        hidden_dims (Iterable): list of hidden layer dimensions
        output_dim (int): output size
        activation (Callable, optional): activation function. Defaults to nn.Sigmoid().
        output_transform (Callable, optional): transformation applied to the output of the final layer. Defaults to nn.Identity().
        z_scoring (Optional[str], optional): Type of z-scoring to apply, one of 'None', 'Independent', 'Structured'. Defaults to "Independent".
        mean (_type_, optional): Mean value used for z-scoring. Defaults to Optional[float].
        std (_type_, optional): Standard deviation for z-scoring. Defaults to Optional[float].

    Raises:
        ValueError: invalid z-scroing option
        NotImplementedError: only fully connected NN supported currently

    Returns:
        nn.Module: Instance of a neural network
    """
    # check z_scoring
    assert type(x_train) == type(action_train)
    if z_scoring.capitalize() in ["None", "Independent", "Structured"]:
        if isinstance(x_train, int):
            input_dim = x_train + action_train
            assert (
                mean is not None and std is not None
            ), "Provide training data or mean and std. Needed for z-scoring."
        else:
            input = torch.concat(
                [atleast_2d_col(x_train), atleast_2d_col(action_train)], dim=1
            )
            mean, std = get_mean_and_std(z_scoring, input)
            input_dim = input.shape[1]
        # input_dim += action_dim
    else:
        raise ValueError(
            "Invalid z-scoring option, use 'None', 'Independent', 'Structured'."
        )

    if model == "fc":
        seed_all_backends(seed)
        neural_net = FeedforwardNN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            output_transform=output_transform,
            mean=mean,
            std=std,
            z_scoring=z_scoring.capitalize(),
        )
    else:
        raise NotImplementedError
    return neural_net


def train(
    model: nn.Module,
    x_train: torch.Tensor,
    theta_train: torch.Tensor,
    cost_fn: Callable,
    x_val: torch.Tensor,
    theta_val: torch.Tensor,
    actions: Action,
    save_sampled_actions=False,
    num_action_samples_train=10,
    num_action_samples_val=10,
    sample_actions_in_loop=False,
    stop_after_epochs: int = 10,  # reduce from 50 to 10, because actions are sampled before training loop
    max_num_epochs: int = None,
    learning_rate: float = 5e-4,
    batch_size: int = 500,
    resume_training: bool = False,
    ckp_path: str = None,
    ckp_interval: int = 20,
    model_dir: str = None,
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=0,
    shuffle_data=True,
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """Training loop to train a model until convergence with L2 loss.

    Args:
        model (nn.Module): neural network
        x_train (torch.Tensor): training observations
        a_train (torch.Tensor): training actions
        costs_train (torch.Tensor): incurred costs
        x_val (torch.Tensor): validation observations
        a_val (torch.Tensor): validation actions
        costs_val (torch.Tensor): incurred costs
        stop_after_epochs (int, optional): Number of epochs without improvement. Defaults to 20.
        max_num_epochs (int, optional): Maximum number of epochs. Defaults to None.
        learning_rate (float, optional): learning rate. Defaults to 5e-4.
        batch_size (int, optional): batch size. Defaults to 5000.
        resume_training (bool, optional): indicator whether to resume training from a checkpoint. Defaults to False.
        ckp_path (str, optional): path to checkpoint. Defaults to None.
        ckp_interval (int, optional): interval for saving checkpoints. Defaults to 20.
        model_dir (str, optional): directory where to save the trained model. Defaults to None.
        device (str, optional): torch device. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").
        seed (int, optional): seed for training. Defaults to 0.
        shuffle_data (bool, optional): whether to shuffle the data in the dataloader. Defaults to True.

    Returns:
        Tuple[nn.Module, torch.Tensor, torch.Tensor]: best model, training losses, validation losses
    """

    seed_all_backends(seed)

    create_checkpoint_dir(model_dir)

    max_num_epochs = 2**31 - 1 if max_num_epochs is None else max_num_epochs

    _summary = {
        "validation_losses": [],
        "training_losses": [],
    }

    if sample_actions_in_loop:
        train_data = TensorDataset(
            x_train.repeat(num_action_samples_train, 1).to(device),
            theta_train.repeat(num_action_samples_train, 1).to(device),
        )
    else:
        # sample fixed number of actions in beginning
        actions_train = actions.sample(num_action_samples_train * x_train.shape[0])
        train_data = TensorDataset(
            x_train.repeat(num_action_samples_train, 1).to(device),
            theta_train.repeat(num_action_samples_train, 1).to(device),
            actions_train.to(device),
        )

    # fixed vaildation data
    actions_val = actions.sample(num_action_samples_val * x_val.shape[0])
    val_data = TensorDataset(
        x_val.repeat(num_action_samples_val, 1).to(device),
        theta_val.repeat(num_action_samples_val, 1).to(device),
        actions_val.to(device),
    )

    train_loader = DataLoader(
        train_data, batch_size=min(batch_size, x_train.shape[0]), shuffle=shuffle_data
    )
    val_loader = DataLoader(
        val_data, batch_size=min(batch_size, x_val.shape[0]), shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if resume_training:
        (
            model,
            optimizer,
            epoch,
            _best_val_loss,
            _best_model_state_dict,
            _epochs_since_last_improvement,
        ) = load_checkpoint(ckp_path, model, optimizer)
    else:
        epoch = 0
        _best_val_loss = float("inf")
        _best_model_state_dict = None
        _epochs_since_last_improvement = 0

    model.to(device)

    while epoch <= max_num_epochs:
        train_loss_sum = 0.0
        model.train()

        if sample_actions_in_loop:
            action_loader = DataLoader(
                TensorDataset(actions.sample(x_train.shape[0]).to(device)),
                batch_size=batch_size,
                shuffle=shuffle_data,
            )

        for b, data_batch in enumerate(
            train_loader
            if not sample_actions_in_loop
            else zip(train_loader, action_loader)
        ):
            if sample_actions_in_loop:
                (x_batch, theta_batch), (a_batch,) = data_batch
            else:
                x_batch, theta_batch, a_batch = data_batch

            optimizer.zero_grad()

            predictions = model(x_batch, a_batch)

            # L2 loss
            costs_batch = cost_fn(theta_batch, a_batch)
            batch_loss = ((costs_batch - predictions) ** 2).mean()
            train_loss_sum += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

            if save_sampled_actions:
                torch.save(
                    torch.hstack([x_batch, theta_batch, a_batch]),
                    path.join(model_dir, "actions", f"xtha_e{epoch}_batch{b}.pt"),
                )

        epoch += 1
        avg_train_loss = train_loss_sum / len(train_loader)
        _summary["training_losses"].append(avg_train_loss)

        # check validation performance
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for x_val_batch, theta_val_batch, a_val_batch in val_loader:
                val_preds_batch = model(x_val_batch, a_val_batch)
                val_costs_batch = cost_fn(theta_val_batch, a_val_batch)
                val_batch_loss = ((val_costs_batch - val_preds_batch) ** 2).mean()
                val_loss_sum += val_batch_loss.item()

        avg_val_loss = val_loss_sum / len(val_loader)
        _summary["validation_losses"].append(avg_val_loss)

        (
            converged,
            _best_model_state_dict,
            _epochs_since_last_improvement,
            _best_val_loss,
        ) = check_converged(
            model,
            avg_val_loss,
            _best_val_loss,
            _best_model_state_dict,
            _epochs_since_last_improvement,
            epoch,
            stop_after_epochs,
        )

        is_best = _epochs_since_last_improvement == 0
        # if epoch % ckp_interval == 0 or is_best:
        #     checkpoint = {
        #         "epoch": epoch,
        #         "state_dict": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "training_losses": torch.as_tensor(_summary["training_losses"]),
        #         "validation_losses": torch.as_tensor(_summary["validation_losses"]),
        #         "_best_val_loss": _best_val_loss,
        #         "_best_model_state_dict": _best_model_state_dict,
        #         "_epochs_since_last_improvement": _epochs_since_last_improvement,
        #     }

        if is_best:
            checkpoint_best = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_losses": torch.as_tensor(_summary["training_losses"]),
                "validation_losses": torch.as_tensor(_summary["validation_losses"]),
                "_best_val_loss": _best_val_loss,
                "_best_model_state_dict": _best_model_state_dict,
                "_epochs_since_last_improvement": _epochs_since_last_improvement,
            }

        print(
            f"{epoch}\t val_loss = {avg_val_loss:.8f}\t train_loss = {avg_train_loss:.8f}\t last_improvement = {_epochs_since_last_improvement}",
            end="\r",
        )
        if converged:
            save_checkpoint(checkpoint_best, True, ckp_interval, model_dir)
            print(
                f"\n{'-'*81}\n|\tConverged after {epoch} epochs, best achieved validation loss: {checkpoint_best['_best_val_loss']:.4f}.\t|\n{'-'*81}\n"
            )
            break
        elif max_num_epochs == epoch:
            print(
                f"Maximum number of epochs `max_num_epochs={max_num_epochs}` reached,"
                "but network has not yet fully converged. Consider increasing it."
            )

    model.load_state_dict(_best_model_state_dict)  # best model
    return (
        deepcopy(model),  # best model
        torch.as_tensor(_summary["training_losses"]),  # training loss
        torch.as_tensor(_summary["validation_losses"]),  # validation loss
    )


def check_converged(
    model: nn.Module,
    _val_loss: torch.Tensor,
    _best_val_loss: torch.Tensor,
    _best_model_state_dict: Dict,
    _epochs_since_last_improvement: int,
    epoch: int,
    stop_after_epochs: int,
):
    """check convergence during training

    Args:
        model (nn.Module): neural network
        _val_loss (torch.Tensor): current validation loss
        _best_val_loss (torch.Tensor): best observed validation loss
        _best_model_state_dict (Dict): best model
        _epochs_since_last_improvement (int): epochs since last improvement
        epoch (int): number of epochs trained
        stop_after_epochs (int): number of epochs without improvement

    Returns:
        Tuple[bool, Dict, int, float]: converged, best model, epochs since last improvement, best validation loss
    """
    converged = False

    # (Re)-start the epoch count with the first epoch or any improvement.
    if epoch == 0 or _val_loss < _best_val_loss:
        _best_val_loss = _val_loss
        _epochs_since_last_improvement = 0
        _best_model_state_dict = deepcopy(model.state_dict())
    else:
        _epochs_since_last_improvement += 1

    # If no validation improvement over many epochs, stop training.
    if _epochs_since_last_improvement > stop_after_epochs - 1:
        model.load_state_dict(_best_model_state_dict)
        converged = True

    return (
        converged,
        _best_model_state_dict,
        _epochs_since_last_improvement,
        _best_val_loss,
    )


def save_checkpoint(state: Dict, is_best: bool, save_interval: int, model_dir: str):
    """Save a given checkpoint

    Args:
        state (Dict): model state dictionary
        is_best (bool): indicator whether it is the best model
        save_interval (int): interval to save checkpoints
        model_dir (str): directory where to save the checkpoint
    """
    assert path.exists(
        path.join(model_dir, "checkpoints")
    ), "Subdirectory 'checkpoints' missing."

    # if state["epoch"] % save_interval == 0:
    #     f_path = path.join(model_dir, f"checkpoints/checkpoint_e{state['epoch']}.pt")
    #     torch.save(state, f_path)
    if is_best:
        best_fpath = path.join(model_dir, "best_model.pt")
        torch.save(state, best_fpath)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer):
    """Load checkpoint to resume training

    Args:
        checkpoint_path (str): path to the checkpoint
        model (nn.Module): instance of a model
        optimizer (_type_): optimizer to load state dict

    Returns:
        Tuple[nn.Module, nn.optim, int, torch.Tensor, float, Dict, int]: model, optimizer, training loss, best validation loss, model state dict, epochs since last improvement
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return (
        model,
        optimizer,
        checkpoint["epoch"] - 1,
        checkpoint["training_loss"],
        checkpoint["_best_val_loss"],
        checkpoint["_best_model_state_dict"],
        checkpoint["_epochs_since_last_improvement"],
    )


def load_predictors(
    task_name: str,
    dir: str,
    nsim: int = None,
    data_dir: str = "./data",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    **task_specifications,
):
    """load all available trained networks

    Args:
        task_name (str): task name
        dir (str): path to models
        data_dir (str, optional): path to data. Defaults to "./data".
        device (torch.device, optional): device. Defaults to torch.device("cuda" if torch.cuda.is_available() else "cpu").

    Returns:
        List[FeedforwardNN], List[int]: list of models and list of number of simulations
    """
    # load data
    (
        _,
        x_train,
        _,
        _,
    ) = load_data(task_name, device=device, base_dir=data_dir)

    model_files = glob.glob(path.join(dir, "best_model.pt"))
    metadata_files = [
        json.load(open(f.split("best_model.pt")[0] + "metadata.json"))
        for f in model_files
    ]

    get_nsim = lambda file: int(
        json.load(open(file.split("best_model.pt")[0] + "metadata.json"))["ntrain"]
    )

    def get_param(file: str):
        param = json.load(open(file.split("best_model.pt")[0] + "metadata.json"))[
            "parameter"
        ]
        if param == None:
            return -1
        else:
            return int(param)

    order_files = lambda file: get_nsim(file) + get_param(file)

    if nsim is None:
        model_files.sort(key=order_files)
        metadata_files.sort(
            key=lambda file: int(file["ntrain"])
            + (int(file["parameter"]) if file["parameter"] is not None else 0)
        )
    else:
        print(f"nsim={nsim}")
        model_files = [m for m in model_files if get_nsim(m) == nsim]
        print(f"model files = {model_files}")
        metadata_files = [m for m in metadata_files if m["ntrain"] == nsim]
        print(f"metadata_file={metadata_files}")

    print("Loading models:")
    str_to_tranformation = {
        "ReLU": torch.nn.ReLU(),
        "Sigmoid": torch.nn.Sigmoid(),
        "Identity": torch.nn.Identity(),
        "Softplus": torch.nn.Softplus(),
    }
    models = []
    num_simulations = []

    for file, metadata in zip(model_files, metadata_files):
        architecture = [int(n) for n in metadata["architecture"].split("-")]
        assert (
            metadata["activation"] in str_to_tranformation.keys()
        ), "Actiavtion function not implemented, one of ['ReLU', 'Sigmoid', 'Identity']."
        assert (
            metadata["output_transform"] in str_to_tranformation.keys()
        ), "Output tranformation not implemented, one of ['ReLU', 'Sigmoid', 'Identity']."
        activation = str_to_tranformation[metadata["activation"]]
        output_transform = str_to_tranformation[metadata["output_transform"]]

        actions = get_task(task_name, **task_specifications).actions
        model = build_nn(
            model=metadata["model"],
            x_train=x_train.to(device),
            action_train=actions.sample(x_train.shape[0]).to(
                device
            ),  # TODO: does the sampling here hurt?
            hidden_dims=architecture[1:-1],
            output_dim=architecture[-1],
            activation=activation,
            output_transform=output_transform,
        )
        model_ckp = torch.load(file, map_location=device)
        model.load_state_dict(model_ckp["state_dict"])
        model.eval()
        models.append(deepcopy(model))
        num_simulations.append(metadata["ntrain"])
        print(
            f"- {metadata['ntrain']} simulations,\t best val loss {model_ckp['_best_val_loss']:.4f}:\t{file}"
        )

    return models, num_simulations
