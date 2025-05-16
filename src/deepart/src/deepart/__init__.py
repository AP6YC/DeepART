"""
Package containing all Python-based experiment code for the DeepART project.
"""

# import torch
# import torch.nn as nn


# from abc import ABC, abstractmethod
# from typing import List, Tuple, Dict, Any


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# from matplotlib import ticker
# %matplotlib inline


# from .optimizers import *
# from .models import *
# from .data import *
# from .utils import *

from . import optimizers
from . import models
from . import data
from . import utils
from . import experiments

__all__ = [
    "optimizers",
    "models",
    "data",
    "utils",
    "experiments",
]


# import optimizers

# device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# print(f"Using {device} device")
# __all__ = [
#     "LocalUpdate"
# ]


# Set the version variable of the package
__version__ = "1.0.0"

