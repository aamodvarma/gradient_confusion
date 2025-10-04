import math
import random
from re import sub

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def permuted_mnist(
    total_tasks=10,
    batch_size=64,
    root="./data",
    shuffle=True,
    seed=0,
    num_workers=0,
):
    train_loaders = []
    test_loaders = []

    class PermuteTransform:
        def __init__(self, perm: torch.Tensor):
            self.perm = perm

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            x = x.view(-1)
            x = x[self.perm]
            if flatten:
                return x
            else:
                return x.view(1, 28, 28)

    permutations = []
    for t in range(total_tasks):
        gen = torch.Generator()
        gen.manual_seed(seed + t)
        perm = torch.randperm(28 * 28, generator=gen)
        permutations.append(perm)

    # create dict to store train and test dataset for each permutation
    # permutation_dict = {}
    # print(permutation_dict)
    for i, perm in enumerate(permutations):
        transform = transforms.Compose([transforms.ToTensor(), PermuteTransform(perm)])
        train_dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform,
        )
        # permutation_dict[i] = train_dataset

        test_dataset = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transform,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


def numbered_mnist(
    batch_size=64,
    root="./data",
    shuffle=True,
    seed=0,
    num_workers=0,
):
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    def split_by_digit(dataset):
        digit_dataloader = []
        # targets = torch.tensor(dataset.targets)  # labels
        targets = dataset.targets.clone().detach()
        for digit in range(10):
            mask = targets == digit
            indices = torch.where(mask)[0]
            indices = indices[: len(indices) // 4]

            subset = torch.utils.data.Subset(dataset, indices)

            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
            digit_dataloader.append(loader)
        return digit_dataloader

    train_dataloaders = split_by_digit(train_dataset)
    test_dataloaders = split_by_digit(test_dataset)
    return train_dataloaders, test_dataloaders


def default_mnist(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader
