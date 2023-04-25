import torch
from torch import nn
from os import path
from typing import Callable, Dict, Tuple, Iterable, Union, Optional
from loss_calibration.loss import BCELoss_weighted
from copy import deepcopy
from torch.utils.data import DataLoader, TensorDataset

from sbi.utils.sbiutils import Standardize


class FeedforwardNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable,
        output_dim: int,
        activation: Callable = nn.Sigmoid(),
        mean: Union[torch.Tensor, float] = None,
        std: Union[torch.Tensor, float] = None,
        z_scoring: Optional[str] = "Independent",
    ):
        """initialize the neural network

        Args:
            input_dim (int): dimensionality of the network input
            hidden_dims (list): list of dimensions of hidden layers
            output_dim (int): dimensionality of the network output
        """
        if len(hidden_dims) == 0:
            raise ValueError(
                "Specify at least one hidden layer, list of hidden dims can't be empty."
            )

        if z_scoring.capitalize() in ["Independent", "Structured"]:
            assert (
                mean is not None and std is not None
            ), "Provide mean and standard deviation for z_scoring."
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

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        assert (
            x.shape[0] == a.shape[0]
        ), "Observations and actions should share the first dimension, i.e. same number of samples."
        if x.shape[1] != self.in_dim - a.shape[1]:
            raise ValueError(
                "Expected inputs of dim {}, got {}.".format(
                    self.in_dim - a.shape[1], x.shape[1]
                )
            )
        if self.z_scored:
            x = self.standardize_layer(x)
        concat = torch.concatenate([x, a], dim=1)
        out = self.input_layer(concat)
        out = self.activation(out)

        for layer in self.hidden_layers:
            out = layer(out)
            # out = nn.BatchNorm1d(out)
            out = self.activation(out)

        out = self.final_layer(out)
        return out


def get_mean_and_std(z_scoring: str, x_train: torch.Tensor, min_std: float = 1e-14):
    assert z_scoring.capitalize() in ["None", "Independent", "Structured"]
    if z_scoring == "Independent":
        std = x_train.std(dim=0)
        std[std < min_std] = min_std
        return x_train.mean(dim=0), std
    elif z_scoring == "Structured":
        std = x_train.std()
        std[std < min_std] = min_std
        return x_train.mean(), std
    else:
        return torch.zeros(1), torch.ones(1)


def build_classifier(
    model: str,
    # input_dim: int,
    x_train: Union[torch.Tensor, int],
    action_dim: int,
    hidden_dims: Iterable,
    output_dim: int,
    context=None,
    num_blocks=2,
    activation: Callable = nn.Sigmoid(),
    dropout_prob=0.0,
    use_batch_norm=False,
    z_scoring: Optional[str] = "Independent",
    mean=Optional[float],
    std=Optional[float],
):
    # check z_scoring
    if z_scoring.capitalize() in ["None", "Independent", "Structured"]:
        if type(x_train) == int:
            input_dim = x_train
            assert (
                mean is not None and std is not None
            ), "Provide training data or mean and std."
        else:
            mean, std = get_mean_and_std(z_scoring, x_train)
            input_dim = x_train.shape[1]
        input_dim += action_dim
    else:
        raise ValueError(
            "Invalid z-scoring opion, use 'None', 'Independent', 'Structured'."
        )

    if model == "fc":
        clf = FeedforwardNN(
            input_dim,
            hidden_dims,
            output_dim,
            activation,
            mean,
            std,
            z_scoring.capitalize(),
        )
    elif model == "resnet":
        clf = ResNet(
            input_dim,
            hidden_dims,
            output_dim,
            context,
            num_blocks,
            activation,
            dropout_prob,
            use_batch_norm,
            mean,
            std,
            z_scoring.capitalize(),
        )
    else:
        raise NotImplementedError
    return clf


def train(
    model: nn.Module,
    x_train: torch.Tensor,
    a_train: torch.Tensor,
    costs_train: torch.Tensor,
    x_val: torch.Tensor,
    a_val: torch.Tensor,
    costs_val: torch.Tensor,
    stop_after_epochs: int = 20,
    max_num_epochs: int = None,
    learning_rate: float = 5e-4,
    batch_size: int = 5000,
    resume_training: bool = False,
    ckp_path: str = None,
    ckp_interval: int = 20,
    model_dir: str = None,
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    seed=0,
) -> Tuple[nn.Module, torch.Tensor, torch.Tensor]:
    """train classifier until convergence

    Args:
        model (nn.Module): network instance to train
        x_train (torch.Tensor): training data - observations
        theta_train (torch.Tensor): training data - parameters
        x_val (torch.Tensor): validation data - observations
        theta_val (torch.Tensor): validation data - parameters
        costs (list): costs of misclassification, for FN and FP
        threshold (float): threshold for binarized decisions
        stop_after_epochs (int, optional): Number of epochs to wait for improvement on the validation data before terminating training. Defaults to 20.
        max_num_epochs (int, optional): Maximum number of epochs to train for. Defaults to None.
        learning_rate (float, optional): Learning rate. Defaults to 5e-4.
        batch_size (int, optional): Training batch size. Defaults to 10000.
        resume_training (bool, optional): Whether to resume training. Defaults to False.
        ckp_path (str, optional): Path to the checkpoint file in case training didn't complete. Defaults to None.
        model_dir (str, optional): Directory to save the trained classifier to. Defaults to None.
        device (str, optional): Device. Defaults to "cpu".

    Returns:
        Tuple[nn.Module, torch.Tensor, torch.Tensor]: trained classifier, training loss, validation loss
    """
    torch.manual_seed(seed)

    max_num_epochs = 2**31 - 1 if max_num_epochs is None else max_num_epochs

    _summary = dict(
        validation_losses=[],
        training_losses=[],
    )

    train_data = TensorDataset(
        costs_train.to(device), x_train.to(device), a_train.to(device)
    )
    val_data = TensorDataset(costs_val.to(device), x_val.to(device), a_val.to(device))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

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
        for costs_batch, x_batch, a_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(x_batch, a_batch)

            # L2 loss
            batch_loss = ((costs_batch - predictions) ** 2).sum(dim=0)
            train_loss_sum += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        epoch += 1
        avg_train_loss = train_loss_sum / costs_train.shape[0]
        _summary["training_losses"].append(avg_train_loss)

        # check validation performance
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for costs_batch, x_batch, a_batch in val_loader:
                preds_batch = model(x_batch, a_batch)
                val_batch_loss = ((costs_batch - preds_batch) ** 2).sum(dim=0)
                val_loss_sum += val_batch_loss.item()

        avg_val_loss = val_loss_sum / costs_val.shape[0]
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
        if epoch % ckp_interval == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_losses": torch.as_tensor(_summary["training_losses"]),
                "validation_losses": torch.as_tensor(_summary["validation_losses"]),
                "_best_val_loss": _best_val_loss,
                "_best_model_state_dict": _best_model_state_dict,
                "_epochs_since_last_improvement": _epochs_since_last_improvement,
            }

            save_checkpoint(checkpoint, is_best, ckp_interval, model_dir)

        if epoch % ckp_interval == 0:
            print(
                f"{epoch}\t val_loss = {avg_val_loss:.8f}\t train_loss = {avg_train_loss:.8f}\t last_improvement = {_epochs_since_last_improvement}",
                end="\r",
            )
        if converged:
            print(f"Converged after {epoch} epochs.")
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


def save_checkpoint(state, is_best, save_interval, model_dir):
    if state["epoch"] % save_interval == 0:
        f_path = path.join(model_dir, f"checkpoints/checkpoint_e{state['epoch']}.pt")
        torch.save(state, f_path)
    if is_best:
        best_fpath = path.join(model_dir, "best_model.pt")
        torch.save(state, best_fpath)


def load_checkpoint(checkpoint_path, model, optimizer):
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
