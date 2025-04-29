"""
TODO
"""

import torch
import torch.nn as nn

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import ticker
# %matplotlib inline


# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
__all__ = [
    "LocalUpdate"
]


# Set the version variable of the package
__version__ = "1.0.0"

def get_device():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    return device


class LocalUpdate(ABC):

    def __init__(self):
        self.logger = logging.getLogger(
            f"{__name__}-{self.__class__.__name__}"
        )

    @abstractmethod
    def update(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
    ):
        pass


class Hebb(LocalUpdate):
    """
    Implements a local Hebbian weight update rule:
    Δw_ij = eta * y_i * x_j
    where y_i is the output, x_j is the input, and eta is the learning rate.
    """

    def __init__(self, eta=0.01):
        super().__init__()
        self.eta = eta
        self.previous = []
        # self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        return

    # def gpu(self):
    #     # print(f"Using {device} device")

    #     return

    def update_model(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        model: nn.Sequential,
    ):
        ox = x
        activations = []
        for layer in model:
            x = layer(x)
            activations.append(x)

        inputs = [ox] + (activations[:-1])
        n_layers = len(model) - 1
        for ix, layer in enumerate(model):
            if hasattr(layer, "weight"):
                if ix == n_layers:
                    dW = self.wh_update(inputs[ix], activations[ix], layer.weight, y)
                    layer.weight += dW
                else:
                    dW = self.update(inputs[ix], activations[ix], layer.weight)
                    layer.weight += dW

    def wh_update(
        self,
        x: torch.Tensor,  # shape: (batch_size, input_dim)
        y: torch.Tensor,
        w: torch.Tensor,   # shape: (output_dim, input_dim)
        target: torch.tensor
    ):
        # OPTION 1: ZIP
        # d_ws = torch.zeros(x.size(0), *w.shape)
        # for idx, (x_sample, ys, targets) in enumerate(zip(x, y, target)):
        #     d_w = self.eta * torch.outer(targets - ys, x_sample)  # shape: (output_dim, input_dim)
        #     # d_w = 0.1 * torch.outer(targets - ys, x_sample)  # shape: (output_dim, input_dim)
        #     # torch.einsum('bp,bqr->bpqr', v, M) # batch-wise operation v.shape=(b,p) M.shape=(b,q,r)
        #     # torch.einsum('p,qr->pqr', v, M)    # cross-batch operation

        #     d_ws[idx] = d_w
        # OPTION 2: EINSUM
        d_ws = self.eta * torch.einsum('bp,bq->bpq', (target-y), x)

        return torch.mean(d_ws, dim=0)

    def update(
        self,
        x: torch.Tensor,  # shape: (batch_size, input_dim)
        y: torch.Tensor,
        w: torch.Tensor   # shape: (output_dim, input_dim)
    ):
        # OPTION 1: ZIP
        # Allocate weight update for each sample
        # d_ws = torch.zeros(x.size(0), *w.shape).to(device)
        # d_ws = torch.zeros(x.size(0), *w.shape)
        # for idx, (x_sample, ys) in enumerate(zip(x, y)):
        #     # d_w = self.eta * torch.outer(ys, x_sample - ys @ w)  # shape: (output_dim, input_dim)
        #     d_w = self.eta * torch.outer(ys, x_sample - ys @ w)  # shape: (output_dim, input_dim)
        #     d_ws[idx] = d_w

        # OPTION 2: EINSUM
        d_ws = self.eta * torch.einsum('bp,bq->bpq', y, x - y @ w)  # shape: (output_dim, input_dim)

        return torch.mean(d_ws, dim=0)


class SimpleHebbNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        nh = 100
        nh2 = 50
        self.fc = nn.Sequential(
            nn.Linear(input_dim, nh, bias=False),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.SiLU(),
            # nn.ReLU6(),

            nn.Linear(nh, nh2, bias=False),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.SiLU(),
            # nn.ReLU6(),

            nn.Linear(nh2, output_dim, bias=False),
            # nn.ReLU6(),
        )
        for p in self.fc.parameters():
            p.requires_grad = False

        # self.fc[-1].weight.uniform_()
        return

    def forward(self, x):
        return self.fc(x)


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


def get_model():
    # Instantiate model and Hebbian updater
    # input_dim = 28 * 28
    input_dim = 16 * 16
    output_dim = 10
    model = SimpleHebbNet(input_dim, output_dim)
    updater = Hebb(eta=0.05)

    # GPU = False
    # GPU = True

    # if GPU:
        # model = model.to(device)
    return model, updater

    # model.fc[-1].weight
    # a = torch.randn((1, 10))
    # b = torch.randn((1, 50))
    # c = torch.randn((1, 10))
    # d = torch.einsum('bp,bq->bpq', (a-c), b)
    # d.shape

    # mnist_train[1][0].max()
    # mnist_train[1][0].shape
    # mnist_train[1][1]
