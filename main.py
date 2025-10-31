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


class MNISTMLP(nn.Module):
    def __init__(
        self,
        input_size,
        width,
        depth,
        output_size,
        activation="identity",
        init_type="glorot",
    ):
        super().__init__()
        layers = []
        in_features = input_size

        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            if activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "identity":
                layers.append(nn.Identity())
            else:
                raise ValueError("activation must be 'identity' or 'tanh'")
            in_features = width

        layers.append(nn.Linear(in_features, output_size))
        self.net = nn.Sequential(*layers)

        self.apply(lambda m: self._init_weights(m, init_type))

    def _init_weights(self, m, init_type):
        if isinstance(m, nn.Linear):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            if init_type == "glorot":
                std = (2.0 / (fan_in + fan_out)) ** 0.5
            elif init_type == "lecun":
                std = (1.0 / fan_in) ** 0.5
            else:
                raise ValueError("init_type must be 'glorot' or 'lecun'")
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_loop(train_loader, model, criterion, optimizer, scheduler, batches=None):
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

        # if batch % 100 == 0:
        # print(f"Batch {batch}, Loss: {loss.item():.4f}")

    if scheduler:
        scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
    return avg_loss


def test_loop(test_loader, model, criterion, p=True):
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
    if p:
        print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy*100:.2f}%")

    return avg_loss, accuracy


if __name__ == "__main__":
    depths = [5]
    widths = [10]
    epoch_list = [10]
    learning_rates = [0.001]
    # alphas = [0.95, 0.99, 0.9, 0.85]
    alphas = [0.9]
    input_size = 784
    output_size = 10
    cycles = 10  # in each cycle all numbers are trained
    iterations = 50
    default = False

    if default:
        train_loaders, test_loaders = default_mnist(batch_size=128, num_workers=8)
        train_loaders = [train_loaders]
        test_loaders = [test_loaders]
    else:
        train_loaders, test_loaders = numbered_mnist(batch_size=128, num_workers=8)

    for width, depth, epochs, learning_rate, alpha in list(
        itertools.product(widths, depths, epoch_list, learning_rates, alphas)
    ):
        model = MNISTMLP(input_size, width, depth, output_size).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=epochs
            * 10,  # multiply step_size by 10 to change after every cycle
            gamma=alpha,
        )
        # scheduler = None

        cosin_sims = []
        losses = []
        avg_errors = []
        cosin_sims_numbered = {n: [] for n in range(10)}

        # for cycle in range(cycles):
        for itr in range(iterations):
            # print(f"----- Cycle {cycle} -----")
            print(f"----- Iteation {itr} -----")
            number = random.randint(0, 9)
            train_loader = train_loaders[number]
            test_loader = test_loaders[number]
            print(f"Training on digit: {number}")
            loss = 0
            for epoch in range(epochs):
                print(f"Epoch: {epoch} ")
                print([x["lr"] for x in optimizer.param_groups])
                loss = train_loop(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    scheduler,
                    batches=None,
                )
                losses.append(loss)
                test_loop(test_loader, model, criterion)

                if default:
                    sim = measure_gradient_confusion(
                        model,
                        criterion,
                        train_loaders[0],
                    )  # at the end of number k, compute grad confusion of number k + 1
                    # min_sim = np.min(sim)
                    min_sim = np.percentile(sim, 1)  # 5th percentile
                    cosin_sims.append(min_sim)

            if not default:
                error = 0
                btwn_task_sims = []
                for task1, task2 in itertools.product(range(10), range(10)):
                    # for each task  compute the between task gradieent confusion
                    sim = between_task_gradient_confusion(
                        model,
                        criterion,
                        train_loaders[task1],  # the current task
                        train_loaders[task2],  # every other task
                    )  # at the end of number k, compute grad confusion of all numbers
                    min_sim = np.percentile(sim, 1)
                    btwn_task_sims.append(min_sim)

                    # for fixed current task, collect min sim across all tasks and store that.

                    if task1 == number:
                        cosin_sims_numbered[task2].append(min_sim)
                        # only compute average loss fixing current task and varying over all other tasks.
                        loss, _ = test_loop(
                            test_loaders[task2], model, criterion, p=False
                        )
                        error += loss

                cosin_sims.append(np.min(btwn_task_sims))
                avg_error = error / 10
                avg_errors.append(avg_error)

        # for number, test_loader in enumerate(test_loaders):
        #     print("------- Final Results --------")
        #     print(f"Testing on digit: {number}")
        #     test_loop(test_loader, model, criterion)

        if default:
            folder = "data/default"
            name = f"{folder}/mnist_d{depth}_w{width}_e{epochs}_lr{learning_rate}.npz"
        else:
            print("EHLO")
            folder = "data/poster-graphs-v2"
            if scheduler is not None:
                name = f"{folder}/mnist_d{depth}_w{width}_e{epochs}_lr{learning_rate}_c{cycles}-{alpha}.npz"
            else:
                name = f"{folder}/mnist_d{depth}_w{width}_e{epochs}_lr{learning_rate}_c{cycles}.npz"

        os.makedirs(folder, exist_ok=True)
        if os.path.isfile(name):
            i = 1
            base, ext = os.path.splitext(name)
            new_name = f"{base}-v{i}{ext}"
            while os.path.isfile(new_name):
                i += 1
                new_name = f"{base}-v{i}{ext}"
            name = new_name

        np.savez(
            name,
            losses=losses,
            cosin_sims=cosin_sims,
            epochs=epochs,
            cycles=cycles,
            cosin_sims_numbered=np.array(cosin_sims_numbered, dtype=object),
            avg_errors=avg_errors,
        )
