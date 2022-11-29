import os
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn import svm


def SVM_worker(arg):
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())
    x,y=test_dataset._load_data()
    svc = svm.SVC(C=arg.C, kernel=arg.kernel_type)
    svc.fit(x,y)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--C', type=int, default=1)
    parser.add_argument('-m', '--mode', type=str, default='ovr', choices=['ovr', 'ovo'])
    parser.add_argument('-kt', '--kernel_type', type=str, default='linear',
                        choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])

    arg = parser.parse_args()

    SVM_worker(arg)
