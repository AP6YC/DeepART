import torch

from .. import models
from .. import optimizers
from .. import data
from .. import utils

from tqdm.auto import tqdm

import torch.nn.functional as F

from matplotlib import pyplot as plt

# -----------------------------------------------------------------------------
# FUNCTIONS
# -----------------------------------------------------------------------------


def get_model():
    # Instantiate model and Hebbian updater
    # input_dim = 28 * 28
    input_dim = 16 * 16
    output_dim = 10
    model = models.SimpleHebbNet(input_dim, output_dim)
    updater = optimizers.Hebb(eta=0.01)

    # GPU = False
    # GPU = True

    # if GPU:
    #     model = model.to(device)
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


class ExpContainer():
    def __init__(self):
        train, test = data.get_data()
        model, updater = get_model()
        device = utils.get_device()

        self.train_loader = train
        self.test_loader = test
        self.model = model
        self.updater = updater
        self.device = device
        # self.GPU =
        self.GPU = torch.accelerator.is_available()
        if self.GPU:
            self.device = torch.accelerator.current_accelerator().type
            self.model = self.model.to(self.device)
        else:
            self.device = "cpu"

        self.lossi = []
        self.perfs = []

        self.n_epochs = 50
        # self.stepi = []

        return

    def test(self, toprint=False):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data_in, target in self.test_loader:

                if self.GPU:
                    data_in = data_in.to(self.device)
                    target = target.to(self.device)

                output = self.model(data_in)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        accuracy = correct / total
        if toprint:
            print(f"Test accuracy: {accuracy:.4f}")

        return accuracy

    def loss(self, logits, targets):
        loss = F.cross_entropy(logits, targets)
        self.lossi.append(loss.log10().item())
        return
        # self.stepi.append(i)

    def train(self):
        # print(self.loader)
        for ie in tqdm(range(self.n_epochs)):
            for data_in, target in self.train_loader:

                # Push the minibatch to the device
                if self.GPU:
                    data_in = data_in.to(self.device)
                    target = target.to(self.device)

                self.updater.update_model(data_in, target, self.model.fc)

                # Is this loss?
                self.loss(self.model(data_in), target)

            self.perfs.append(self.test())
            # self.updater.eta *= 0.975
            self.updater.decay()

        return self.perfs

    def plot_test(self):
        return plt.plot(range(self.n_epochs), self.perfs)

    def plot_loss(self):
        return plt.plot(range(len(self.lossi)), self.lossi)

# def test(model, test_loader, toprint=False):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data, target in test_loader:

#             if GPU:
#                 data = data.to(device)
#                 target = target.to(device)

#             output = model(data)
#             pred = output.argmax(dim=1)
#             correct += (pred == target).sum().item()
#             total += target.size(0)
#     accuracy = correct / total
#     if toprint:
#         print(f"Test accuracy: {accuracy:.4f}")
#     return accuracy


# def train(model, loader):
#     perfs = []
#     n_epochs = 50
#     for ie in range(n_epochs):
#         for data, target in tqdm(loader):
#             if GPU:
#                 data = data.to(device)
#                 target = target.to(device)
#             hebb_updater.update_model(data, target, model.fc)
#         perfs.append(test(model, test_loader))
