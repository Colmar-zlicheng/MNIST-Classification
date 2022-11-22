import torch
import torch.nn as nn
import argparse
import torchvision
import torchvision.transforms as transforms

from lib.utils.etqdm import etqdm

def main_worker():
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1,
                                               shuffle=True)

    # for i, (images, labels) in enumerate(train_loader):

if __name__=='__main__':
    arg = argparse.ArgumentParser()

    main_worker()