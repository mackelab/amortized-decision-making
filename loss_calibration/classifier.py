import torch
from torch import nn
from os import path
from typing import Callable, Dict
from loss_calibration.loss import BCELoss_weighted
from copy import deepcopy


class FeedforwardNN(nn.Module):
    def __init__(
        self, input_dim: int = 1, hidden_dims: list = [5], output_dim: int = 1
    ):
        """initialize the neural network

        Args:
            input_dim (int): dimensionality of the network input
            hidden_dims (list): list of dimensions of hidden layers
            output_dim (int): dimensionality of the network output
        """
        assert len(hidden_dims) >= 1, "Specify at least one hidden layer."
        super(FeedforwardNN, self).__init__()

        # Layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.hidden_layers = []
        if len(hidden_dims) > 1:
            for l in range(1, len(hidden_dims)):
                self.hidden_layers.append(nn.Linear(hidden_dims[l - 1], hidden_dims[l]))
                input_dim = hidden_dims[l]
            self.hidden_layers = nn.ModuleList(self.hidden_layers)
        self.final_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        out = self.input_layer(x)
        out = self.sigmoid(out)

        for layer in self.hidden_layers:
            out = layer(out)
            out = self.sigmoid(out)

        out = self.final_layer(out)
        out = self.sigmoid(out)  # output probability of belonging to class 0
        return out

    def training(
        self,
        x_train: torch.Tensor,
        th_train: torch.Tensor,
        threshold: float,
        epochs: int,
        criterion: Callable,
        optimizer,
        batch_size: int = 10000,
    ):
        d_train = (th_train > threshold).float()
        # dataset = data.TensorDataset(th_train, x_train, d_train)
        # train_loader = data.DataLoader(dataset, batch_size=batch_size)  # , shuffle=True)

        self.train()

        loss_values = []
        for epoch in range(epochs):
            # for theta_batch, x_batch, d_batch in train_loader:
            optimizer.zero_grad()

            predictions = self(x_train)
            loss = criterion(predictions, d_train, th_train)  # ! requires gt theta

            with torch.no_grad():
                loss_values.append(loss.item())

            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(
                    f"{epoch}\t c = {predictions.mean():.8f}\t L = {loss.item():.8f}",
                    end="\r",
                )
        self._summary = dict(
            epochs=epochs,
            costs=None,
            treshold=threshold,
            optimizer=optimizer,
            ntrain=th_train.shape[0],
        )


class simpleNN(nn.Module):

    # /!\ limited to 1 hidden layer

    def __init__(self, input_dim: int = 1, hidden_dim: int = 5, output_dim: int = 1):
        super(simpleNN, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # output probability of belonging to class -1
        return out


def train(
    model: nn.Module,
    x_train: torch.Tensor,
    th_train: torch.Tensor,
    x_val: torch.Tensor,
    th_val: torch.Tensor,
    costs: list,
    threshold: float,
    stop_after_epochs: int = 200,
    max_num_epochs: int = None,
    learning_rate: float = 5e-4,
    batch_size: int = 10000,
    resume_training: bool = False,
    ckp_path: str = None,
    ckp_interval: int = 500,
    model_dir: str = None,
    device: str = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """train classifier until convergence

    Args:
        model (nn.Module): network instance to train
        x_train (torch.Tensor): training data - observations
        th_train (torch.Tensor): training data - parameters
        x_val (torch.Tensor): validation data - observations
        th_val (torch.Tensor): validation data - parameters
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
        Tuple[nn.Module, torch.Tensor]: trained classifier, training loss values
    """

    max_num_epochs = 2**31 - 1 if max_num_epochs is None else max_num_epochs

    _summary = dict(
        validation_losses=[],
        training_losses=[],
    )

    d_train = (th_train > threshold).float()
    d_val = (th_val > threshold).float()
    # dataset = data.TensorDataset(th_train, x_train, d_train)
    # train_loader = data.DataLoader(dataset, batch_size=batch_size)  # , shuffle=True)

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
        # for epoch in range(start_epoch, max_num_epochs):
        # for theta_batch, x_batch, d_batch in train_loader:
        model.train()
        # train_loss_sum = 0
        optimizer.zero_grad()

        predictions = model(x_train)
        loss = criterion(predictions, d_train, th_train).mean(dim=0)

        with torch.no_grad():
            _summary["training_losses"].append(loss.item())

        loss.backward()
        optimizer.step()

        epoch += 1

        # check validation performance
        model.eval()
        with torch.no_grad():
            preds_val = model(x_val)
            val_loss = criterion(preds_val, d_val, th_val).mean()

        _summary["validation_losses"].append(val_loss)

        (
            converged,
            _best_model_state_dict,
            _epochs_since_last_improvement,
            _best_val_loss,
        ) = check_converged(
            model,
            val_loss,
            _best_val_loss,
            _best_model_state_dict,
            _epochs_since_last_improvement,
            epoch,
            stop_after_epochs,
        )

        is_best = _epochs_since_last_improvement == 0
        if (epoch + 1) % ckp_interval == 0 or is_best:
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_losses": torch.as_tensor(_summary["training_losses"]),
                "validation_losses": torch.as_tensor(_summary["validation_losses"]),
                "_best_val_loss": _best_val_loss,
                "_best_model_state_dict": _best_model_state_dict,
                "_epochs_since_last_improvement": _epochs_since_last_improvement,
            }

            save_checkpoint(checkpoint, is_best, ckp_interval, model_dir)

        if (epoch + 1) % ckp_interval == 0:
            print(
                f"{epoch}\t val_loss = {val_loss.item():.8f}\t train_loss = {loss.item():.8f}",
                end="\r",
            )
        if converged:
            print(f"Converged after {epoch} epochs.")
            # break
        elif max_num_epochs == epoch:
            print(
                f"Maximum number of epochs `max_num_epochs={max_num_epochs}` reached,"
                "but network has not yet fully converged. Consider increasing it."
            )

    return (
        deepcopy(model.load_state_dict(_best_model_state_dict)),  # best model
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


def get_stats(clf, x_test, d_test, th_test, loss_criterion):
    clf.eval()
    preds_test = clf(x_test)
    preds_test_bin = (preds_test > 0.5).float()

    tn = torch.logical_and(preds_test_bin == 0, d_test == 0).sum()
    tp = torch.logical_and(preds_test_bin == 1, d_test == 1).sum()
    fn = torch.logical_and(preds_test_bin == 0, d_test == 1).sum()
    fp = torch.logical_and(preds_test_bin == 1, d_test == 0).sum()

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    acc = (tp + tn) / (tn + tp + fn + fp)
    loss = loss_criterion(preds_test, d_test, th_test).mean(dim=0).detach()

    return tpr, tnr, acc, loss
