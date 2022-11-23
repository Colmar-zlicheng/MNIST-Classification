import torch
import torch.nn as nn


class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()

    def forward(self, images, labels, step_idx, mode="train"):
        return 0, 0
