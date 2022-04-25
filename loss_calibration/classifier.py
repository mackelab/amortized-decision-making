import torch
from torch import nn


class FeedforwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
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

    def forward(self, x):
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
        x_train,
        th_train,
        threshold,
        epochs,
        criterion,
        optimizer,
        batch_size=10000,
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
            weights=None,
            treshold=threshold,
            optimizer=optimizer,
            ntrain=th_train.shape[0],
        )


class simpleNN(nn.Module):

    # /!\ limited to 1 hidden layer

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(simpleNN, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Activation function
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # output probability of belonging to class -1
        return out


def train(
    model, x_train, th_train, threshold, epochs, criterion, optimizer, batch_size=10000
):
    d_train = (th_train > threshold).float()
    # dataset = data.TensorDataset(th_train, x_train, d_train)
    # train_loader = data.DataLoader(dataset, batch_size=batch_size)  # , shuffle=True)

    model.train()

    loss_values = []
    for epoch in range(epochs):
        # for theta_batch, x_batch, d_batch in train_loader:
        optimizer.zero_grad()

        predictions = model(x_train)
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

    model._summary = dict(
        epochs=epochs,
        weights=None,
        treshold=threshold,
        optimizer=optimizer,
        ntrain=th_train.shape[0],
    )

    return model, loss_values
