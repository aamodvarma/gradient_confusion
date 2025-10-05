import torch
import os
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import *
from confusion import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True


class LinearNN(nn.Module):
    def __init__(self, input_size, width, depth, output_size):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.Identity())  # identity activation
            in_features = width
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

        # Xavier init
        self.apply(lambda m: self._init_weights(m))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class TanhNN(nn.Module):
    def __init__(self, input_size, width, depth, output_size):
        super().__init__()
        layers = []
        in_features = input_size
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.Tanh())
            in_features = width
        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

        # Xavier init
        self.apply(lambda m: self._init_weights(m))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_loop(train_loader, model, criterion, optimizer, batches=None):
    model.train()
    total_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        X = X.view(X.size(0), -1).to(device)
        y = y.to(device)
        pred = model(X)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

        if batches and batch == batches:
            break

        if batch % 100 == 0:
            print(f"Batch {batch}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return avg_loss


def test_loop(test_loader, model, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.view(X.size(0), -1).to(device)
            y = y.to(device)
            pred = model(X)
            loss = criterion(pred, y)
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / len(test_loader.dataset)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

    return avg_loss, accuracy


if __name__ == "__main__":
    depths = [300]
    widths = [3, 10, 100]
    epoch_list = [100]
    learning_rates = [0.001]
    input_size = 784
    output_size = 10
    cycles = 1  # in each cycle all numbers are trained
    default = True

    if default:
        train_loaders, test_loaders = default_mnist(batch_size=128, num_workers=8)
        train_loaders = [train_loaders]
        test_loaders = [test_loaders]
    else:
        train_loaders, test_loaders = numbered_mnist(batch_size=128, num_workers=8)

    for width, depth, epochs, learning_rate in list(
        itertools.product(widths, depths, epoch_list, learning_rates)
    ):
        model = TanhNN(input_size, width, depth, output_size).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        cosin_sims = []
        losses = []

        for cycle in range(cycles):
            print(f"----- Cycle {cycle} -----")
            for number, train_loader in enumerate(train_loaders):
                print(f"Training on digit: {number}")
                loss = 0
                for epoch in range(epochs):
                    print(f"Epoch: {epoch} ")
                    loss = train_loop(
                        train_loader, model, criterion, optimizer, batches=None
                    )
                    losses.append(loss)
                    test_loop(test_loaders[number], model, criterion)

                    if default:
                        sim = measure_gradient_confusion(
                            model,
                            criterion,
                            train_loaders[(number) % len(train_loaders)],
                        )  # at the end of number k, compute grad confusion of number k + 1
                        # min_sim = np.min(sim)
                        min_sim = np.percentile(sim, 5)  # 5th percentile
                        cosin_sims.append(min_sim)
                    # compute grad simlairity after training each number

                if not default:
                    sim = measure_gradient_confusion(
                        model,
                        criterion,
                        train_loaders[(number) % len(train_loaders)],
                    )  # at the end of number k, compute grad confusion of number k + 1
                    # min_sim = np.min(sim)
                    min_sim = np.percentile(sim, 1)  # 1st percentile
                    cosin_sims.append(min_sim)

        for number, test_loader in enumerate(test_loaders):
            print("------- Final Results --------")
            print(f"Testing on digit: {number}")
            test_loop(test_loader, model, criterion)

        if default:
            folder = "data/default2"
            name = f"{folder}/mnist_d{depth}_w{width}_e{epochs}_lr{learning_rate}.npz"
        else:
            folder = "data/num4"
            name = f"{folder}/mnist_d{depth}_w{width}_e{epochs}_lr{learning_rate}_c{cycles}.npz"

        os.makedirs(folder, exist_ok=True)
        np.savez(
            name,
            losses=losses,
            cosin_sims=cosin_sims,
            epochs=epochs,
            cycles=cycles,
        )
