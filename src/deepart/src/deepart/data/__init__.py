
import torch
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

__all__ = [
    "get_data",
]


def get_data():
    # Prepare MNIST data (flattened)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # flatten
    ])

    # mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform,
    mnist_train = datasets.USPS(
        root='./data',
        train=True,
        download=True,
        transform=transform,
        target_transform=transforms.Lambda(
            lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
        )
        # target_transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Lambda(lambda y: F.one_hot(y, num_classes=10)),
        # ])
    )
    # loader = DataLoader(mnist_train, batch_size=32, shuffle=True)
    loader = DataLoader(
        mnist_train,
        batch_size=1024,
        shuffle=True,
    )

    # Prepare MNIST test data
    # mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    mnist_test = datasets.USPS(
        root='./data',
        train=False,
        download=True,
        transform=transform)
    test_loader = DataLoader(
        mnist_test,
        batch_size=1024,
        shuffle=False
    )

    return loader, test_loader

