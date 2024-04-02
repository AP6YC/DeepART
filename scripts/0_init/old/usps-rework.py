import torch as torch
import torchvision
import pandas as pd

# Load the USPS dataset
usps_train = torchvision.datasets.USPS(
    root='../work/data/usps/',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)

usps_test = torchvision.datasets.USPS(
    root='../work/data/usps/',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
)
