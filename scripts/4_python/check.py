import torch
import torch.nn as nn
from deepart.models import DeepARTWrapper

base = nn.Sequential(
    nn.Linear(784, 256, bias=False),
    nn.ReLU(),
    nn.Linear(256, 10, bias=False),
)

model = DeepARTWrapper(
    base,
    learning_rule="oja",
    eta=0.01,
    beta_rule="wta",
    apply_complement_coding=True,
)

x = torch.rand(32, 784)
target = torch.randint(0, 10, (32,))  # optional
y_hat = model.learn_step(x, target=target)  # forward + local weight updates
