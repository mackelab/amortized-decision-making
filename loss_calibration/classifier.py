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

    def forward(self, x: torch.Tensor):
        if x.shape[1] != self.in_dim:
            raise ValueError(
                "Expected inputs of dim {}, got {}.".format(self._in_dim, x.shape[1])
            )
        if self.z_scored:
            x = self.standardize_layer(x)
        out = self.input_layer(x)
        out = self.activation(out)

        for layer in self.hidden_layers:
            out = layer(out)
            # out = nn.BatchNorm1d(out)
            out = self.activation(out)

        out = self.final_layer(out)
        out = self.activation(out)  # output probability of belonging to class 0
        return out


# implementation adapted from nflows/resnet.py
class ResNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable,
        output_dim: int,
        context=None,
        num_blocks=2,
        activation: Callable = nn.Sigmoid(),
        dropout_prob=0.0,
        use_batch_norm=False,
        mean: Union[torch.Tensor, float] = None,
        std: Union[torch.Tensor, float] = None,
        z_scoring: Optional[str] = "Independent",
    ):

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

        super().__init__()
        self.hidden_dims = hidden_dims
        self.context = context
        self.z_scored = z_scoring_bool
        if z_scoring_bool:
            self.standardize_layer = Standardize(mean, std)
        if context is not None:
            self.initial_layer = nn.Linear(input_dim + context, hidden_dims[0])
        else:
            self.initial_layer = nn.Linear(input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    features=hidden_dims,
                    context=context,
                    activation=activation,
                    dropout_prob=dropout_prob,
                    use_batch_norm=use_batch_norm,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward():
        pass

    def forward(self, x, context=None):
        if self.z_scored:
            x = self.standardize_layer(x)
        if context is None:
            out = self.initial_layer(x)
        else:
            out = self.initial_layer(torch.cat((x, context), dim=1))
        for block in self.blocks:
            out = block(out, context=context)
        out = self.final_layer(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(
        self,
        features,
        context,
        activation=nn.Sigmoid(),
        dropout_prob=0.0,
        use_batch_norm=False,
        zero_init=True,
    ):
        super().__init__()
        self.activation = activation

        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [
                    nn.BatchNorm1d(f, eps=1e-3) for f in features
                ]  # fixed to 2: for _ in range(2)
            )
        if context is not None:
            self.context_layer = nn.Linear(context, features)
        self.linear_layers = nn.ModuleList(
            [
                nn.Linear(f1, f2) for f1, f2 in zip(features[:-1], features[1:])
            ]  # fixed to 2: for _ in range(2)
        )
        self.dropout = nn.Dropout(p=dropout_prob)
        if zero_init:
            nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
            nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, x, context=None):
        n_layers = len(self.linear_layers)
        res = x
        if self.use_batch_norm:
            res = self.batch_norm_layers[0](res)  #! assumes fixed 2 hidden layers
        res = self.activation(res)
        res = self.linear_layers[0](res)  #! assumes fixed 2 hidden layers
        for l in range(1, n_layers):
            if self.use_batch_norm:
                res = self.batch_norm_layers[l](res)
                res = self.activation(res)
                if l == n_layers - 1:
                    res = self.dropout(res)
                res = self.linear_layers[l](res)
        if context is not None:
            res = nn.functional.glu(
                torch.cat((res, self.context_layer(context)), dim=1), dim=1
            )
        return x + res


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
    x_train: torch.Tensor,
    hidden_dims: Iterable,
    output_dim: int,
    context=None,
    num_blocks=2,
    activation: Callable = nn.Sigmoid(),
    dropout_prob=0.0,
    use_batch_norm=False,
    z_scoring: Optional[str] = "Independent",
):
    # check z_scoring
    if z_scoring.capitalize() in ["None", "Independent", "Structured"]:
        mean, std = get_mean_and_std(z_scoring, x_train)
    else:
        raise ValueError(
            "Invalid z-scoring opion, use 'None', 'Independent', 'Structured'."
        )

    input_dim = x_train.shape[1]

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
    theta_train: torch.Tensor,
    x_val: torch.Tensor,
    theta_val: torch.Tensor,
    costs: list,
    threshold: float,
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

    assert (
        theta_train.shape[1] == 1
    ), "Only decisions based on one parameter implemented."
    torch.manual_seed(seed)

    max_num_epochs = 2**31 - 1 if max_num_epochs is None else max_num_epochs

    _summary = dict(
        validation_losses=[],
        training_losses=[],
    )

    d_train = (theta_train > threshold).float()
    d_val = (theta_val > threshold).float()

    train_data = TensorDataset(
        theta_train.to(device), x_train.to(device), d_train.to(device)
    )
    val_data = TensorDataset(theta_val.to(device), x_val.to(device), d_val.to(device))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_loader = DataLoader(val_data, batch_size=min(batch_size, theta_val.shape[0]))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BCELoss_weighted(costs, threshold)

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
        for theta_batch, x_batch, d_batch in train_loader:
            optimizer.zero_grad()

            predictions = model(x_batch)
            batch_loss = criterion(predictions, d_batch, theta_batch).sum(dim=0)

            train_loss_sum += batch_loss.item()

            batch_loss.backward()
            optimizer.step()

        epoch += 1
        avg_train_loss = train_loss_sum / theta_train.shape[0]
        _summary["training_losses"].append(avg_train_loss)

        # check validation performance
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for theta_batch, x_batch, d_batch in val_loader:
                preds_batch = model(x_batch)
                val_batch_loss = criterion(preds_batch, d_batch, theta_batch).sum(dim=0)
                val_loss_sum += val_batch_loss.item()

        avg_val_loss = val_loss_sum / theta_val.shape[0]
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


def get_raw_stats(clf, theta_test, x_test, threshold):
    clf.eval()
    d_test = (theta_test > threshold).float()
    ratio_predicted = clf(x_test)
    d_predicted = (ratio_predicted > 0.5).float()
    tn = torch.logical_and(d_predicted == 0, d_test == 0)
    tp = torch.logical_and(d_predicted == 1, d_test == 1)
    fn = torch.logical_and(d_predicted == 0, d_test == 1)
    fp = torch.logical_and(d_predicted == 1, d_test == 0)
    return tp, fn, fp, tn


def get_stats(clf, theta_test, x_test, threshold):
    tp, fn, fp, tn = get_raw_stats(clf, theta_test, x_test, threshold)
    acc = (tp.sum() + tn.sum()) / (tp.sum() + fn.sum() + fp.sum() + tn.sum())
    return tp.sum(), fn.sum(), fp.sum(), tn.sum(), acc
