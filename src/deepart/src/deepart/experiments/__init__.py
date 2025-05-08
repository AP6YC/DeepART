import torch

from .. import models
from .. import optimizers
from .. import data
from .. import utils

from tqdm.auto import tqdm

import torch.nn.functional as F

from matplotlib import pyplot as plt
from matplotlib import ticker

from sklearn.manifold import TSNE

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

    # if GPU:
    #     model = model.to(device)
    return model, updater


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
        return plt.plot(range(len(self.perfs)), self.perfs)

    def plot_loss(self):
        return plt.plot(range(len(self.lossi)), self.lossi)

    def tsne(self):
        # def get_points(model, test_loader):
        def get_points(exp):
            exp.model.eval()
            outputs = []
            targets = []
            with torch.no_grad():
                for data_in, target in exp.test_loader:
                    if exp.GPU:
                        data_in = data_in.to(exp.device)
                        target = target.to(exp.device)

                    output = exp.model(data_in)
                    outputs.append(output)
                    targets.append(target)

            outputs = torch.cat(outputs)
            targets = torch.cat(targets)
            if exp.GPU:
                outputs = outputs.to("cpu")
                targets = targets.to("cpu")
            return outputs, targets

        def add_2d_scatter(ax, points, colors, title=None):
            x, y = points.T
            ax.scatter(x, y, s=50, c=colors, alpha=0.8)
            ax.set_title(title)
            ax.xaxis.set_major_formatter(ticker.NullFormatter())
            ax.yaxis.set_major_formatter(ticker.NullFormatter())

        def plot_2d(points, colors, title):
            fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
            fig.suptitle(title, size=16)
            add_2d_scatter(ax, points, colors)
            plt.show()

        x, y = get_points(self)

        t_sne = TSNE(
            n_components=2,
            perplexity=30,
            init="random",
            max_iter=250,
            random_state=0,
        )

        S_t_sne = t_sne.fit_transform(x)

        cmap = plt.get_cmap('tab10')
        colors = [cmap(i) for i in y]
        return plot_2d(S_t_sne, colors, "USPS DeepART TSNE")


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
