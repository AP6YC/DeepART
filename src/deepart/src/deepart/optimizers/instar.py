# from ._base import *
from ._base import LocalUpdate

import torch
import torch.nn as nn


class Instar(LocalUpdate):
    """
    Implements a local Hebbian weight update rule:
    Δw_ij = eta * y_i * x_j
    where y_i is the output, x_j is the input, and eta is the learning rate.
    """

    def __init__(
        self,
        eta=0.01,
        decay_rate=0.975,
    ):
        super().__init__()
        self.eta = eta
        self.decay_rate = decay_rate
        self.previous = []
        # self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        return

    # def gpu(self):
    #     # print(f"Using {device} device")

    #     return

    def decay(self):
        self.eta *= self.decay_rate
        return

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
        # d_ws = self.eta * torch.einsum('bp,bq->bpq', y, x - w)  # shape: (output_dim, input_dim)

        return torch.mean(d_ws, dim=0)
