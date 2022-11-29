import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt


def viz_main():
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())

    example_images, example_targets = train_dataset._load_data()
    # fig = plt.figure()
    for i, c in enumerate(np.random.randint(0, 10000, 25)):
        plt.subplot(5, 5, i+1)
        plt.tight_layout()
        plt.imshow(example_images[c], cmap='gray', interpolation='none')
        plt.title("GT: {}".format(example_targets[c]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    viz_main()
