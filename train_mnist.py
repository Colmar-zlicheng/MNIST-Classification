import torch
import torch.nn as nn
import time
import argparse
import torchvision
import torchvision.transforms as transforms
from lib.model.MNIST import MNIST
from lib.utils.etqdm import etqdm
from lib.utils.misc import bar_perfixes

def SVM_worker(arg):
    return 0


def ANN_worker(arg):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=arg.batch_size,
                                               shuffle=True)
    model = MNIST()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch_idx in range(arg.epoch_size):
        model.train()
        train_bar = etqdm(train_loader)
        for bidx, (images, labels) in enumerate(train_bar):
            step_idx = epoch_idx * len(train_loader) + bidx

            pred, loss = model(images, labels, step_idx, 'train')

            train_bar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='ANN', choices=['SVM,ANN'])
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-e', '--epoch_size', type=int, default=100)

    arg = parser.parse_args()
    if arg.type == 'SVM':
        SVM_worker(arg)
    elif arg.type == 'ANN':
        ANN_worker(arg)
    else:
        raise ValueError("{} is not supported ".format(arg.type))