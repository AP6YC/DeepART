import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

__all__ = [
    "SimpleHebbNet",
    "ComplementCode",
]


class ComplementCode(nn.Module):
    def __init__(self, in_dim):
        self.dim = in_dim
        return

    def forward(self, x):
        return torch.cat([1 - x, x], dim=1)


class SimpleRes(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleRes, self).__init__()
        self.norm = nn.LayerNorm(out_dim)
        self.dense = nn.Linear(in_dim, out_dim, bias=False)
        self.weight = self.dense.weight
        return

    def forward(self, x):
        # out = self.dense(x)
        # out = self.norm(out)
        out = self.norm(x)
        out = self.dense(out)
        out += x
        # out = nn.Tanh(out)
        return out


class SimpleHebbNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        nh = 300
        # nh = 500
        # nh2 = 50
        # nh3 = 25
        self.fc = nn.Sequential(
            nn.Linear(input_dim, nh, bias=False),
            nn.Tanh(),
            # nn.Sigmoid(),
            # nn.SiLU(),
            # nn.ReLU6(),


            # nn.Linear(nh, nh2, bias=False),
            # nn.Tanh(),

            SimpleRes(nh, nh),
            nn.Tanh(),
            SimpleRes(nh, nh),
            nn.Tanh(),
            SimpleRes(nh, nh),
            nn.Tanh(),
            SimpleRes(nh, nh),
            nn.Tanh(),

            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),

            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),


            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),

            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),
            # SimpleRes(nh, nh),
            # nn.Tanh(),

            # SimpleRes(nh2, nh2),
            # nn.Tanh(),

            # SimpleRes(nh2, nh2),
            # nn.Tanh(),

            # SimpleRes(nh2, nh2),
            # nn.Tanh(),

            # nn.Sigmoid(),
            # nn.SiLU(),
            # nn.ReLU6(),

            # nn.Linear(nh2, nh2, bias=False),
            # nn.Tanh(),
            # nn.Linear(nh2, nh2, bias=False),
            # nn.Tanh(),
            # nn.Linear(nh2, nh2, bias=False),
            # nn.Tanh(),

            nn.Linear(nh, output_dim, bias=False),
            # nn.ReLU6(),
        )
        for p in self.fc.parameters():
            p.requires_grad = False

        # self.fc[-1].weight.uniform_()
        return

    def forward(self, x):
        return self.fc(x)
