"""
TODO
"""

# import torch
# import torch.nn as nn


# from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from tqdm.auto import tqdm

# import torch
# import torch.nn as nn
import torch.nn.functional as F



from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import ticker
# %matplotlib inline


from .optimizers import *
from .models import *
from .data import *
from .utils import *
# import optimizers

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
# __all__ = [
#     "LocalUpdate"
# ]


# Set the version variable of the package
__version__ = "1.0.0"

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


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
