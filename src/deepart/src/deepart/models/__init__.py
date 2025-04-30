import torch.nn as nn

# -----------------------------------------------------------------------------
# MODELS
# -----------------------------------------------------------------------------

__all__ = [
    "SimpleHebbNet"
]


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
