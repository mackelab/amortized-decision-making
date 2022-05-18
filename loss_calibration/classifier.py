import torch
from torch import nn
import shutil
from os import path
from typing import Callable
from loss_calibration.loss import BCELoss_weighted


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

        # Summary
        self._summary = dict(
            epochs=[], weights=[], treshold=[], optimizer=[], ntrain=[]
        )

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
    costs: list,
    threshold: float,
    stop_after_epochs: int,
    max_num_epochs: int,
    learning_rate: float = 5e-4,
    batch_size: int = 10000,
    resume_training: bool = False,
    ckp_path: str = None,
    model_dir: str = None,
):
    """train classifier until convergence

    Args:
        model (nn.Module): classifier to train
        x_train (torch.Tensor): training data - observations
        th_train (troch.Tensor): training data - parameters
        threshold (float): threshold for binary decisions
        stop_after_epochs (int): number of epoch to wait for improvement on the validation data before terminating training
        max_num_epochs (int): maximum number of epoch to train
        criterion (Callable): Loss criterion to compute the incurred loss given prediction, true label and theta
        optimizer (_type_): Optimizer
        batch_size (int, optional): Training batch size. Defaults to 10000.
        resume_training (bool, optional): Whether to resume training. Defaults to True.

    Returns:
        Tuple[nn.Module, torch.Tensor]: trained classifier, loss values
    """
    max_num_epochs = 2**31 - 1 if max_num_epochs is None else max_num_epochs

    d_train = (th_train > threshold).float()
    # dataset = data.TensorDataset(th_train, x_train, d_train)
    # train_loader = data.DataLoader(dataset, batch_size=batch_size)  # , shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = BCELoss_weighted(costs, threshold)

    start_epoch = 0
    loss_values = torch.empty([0])
    if resume_training:
        model, optimizer, start_epoch, loss_values = load_checkpoint(
            ckp_path, model, optimizer
        )

    for epoch in range(start_epoch, max_num_epochs):
        # for theta_batch, x_batch, d_batch in train_loader:
        model.train()
        optimizer.zero_grad()

        predictions = model(x_train)
        loss = criterion(predictions, d_train, th_train).mean(dim=0)

        with torch.no_grad():
            loss_values = torch.cat((loss_values, loss))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:  # TODO: or model is better
            checkpoint = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "training_loss": loss_values,
            }
            is_best = False  # TODO: validation performance

            save_checkpoint(checkpoint, is_best, model_dir)

            print(
                f"{epoch}\t L = {loss.item():.8f}",
                end="\r",
            )

    model._summary = dict(
        epochs=max_num_epochs,
        weights=None,
        treshold=threshold,
        optimizer=optimizer,
        ntrain=th_train.shape[0],
    )

    return model, loss_values


def save_checkpoint(state, is_best, model_dir):
    f_path = path.join(model_dir, f"checkpoints/checkpoint_e{state['epoch']}.pt")
    torch.save(state, f_path)
    if is_best:
        best_fpath = path.join(model_dir, "best_model.pt")
        shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"] - 1, checkpoint["training_loss"]
