import torch
import torch.nn as nn


class MNIST(nn.Module):
    def __init__(self, num_class):
        super(MNIST, self).__init__()
        self.layer0 = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 784))
        self.layer = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, num_class))
        self.compute_loss = nn.CrossEntropyLoss()

    def forward(self, images, labels):  # , step_idx, mode="train"):
        # images: [B, 1, 28, 28]
        # labels: [B]
        B = images.shape[0]
        x = images.reshape(B, -1)
        x = x + self.layer0(x)
        x = self.layer(x)
        loss = self.compute_loss(x, labels)
        return x, loss
