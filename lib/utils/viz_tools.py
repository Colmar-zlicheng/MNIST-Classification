import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from lib.model.MNIST import MNIST
from lib.utils.etqdm import etqdm
from lib.utils.misc import bar_perfixes


def viz_results(arg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=arg.batch_size,
                                              shuffle=False)
    example_images, example_targets = test_dataset._load_data()
    model = MNIST(num_class=10).to(device)
    model.load_state_dict(torch.load(arg.ckpt))

    with torch.no_grad():
        model.eval()
        pred_list = []
        test_bar = etqdm(test_loader)
        for bidx, (images, labels) in enumerate(test_bar):
            pred, loss = model(images.to(device), labels.to(device))
            loss_show = ('%.12f' % loss)
            test_bar.set_description(f"{bar_perfixes['test']} Loss {loss_show}")
            _, predicted = torch.max(pred.data, 1)
            pred_list.append(predicted)
        predicts = torch.stack(pred_list).reshape(-1)
        # fig = plt.figure()
        for i, c in enumerate(np.random.randint(0, 10000, arg.size**2)):
            plt.subplot(arg.size, arg.size, i + 1)
            plt.tight_layout()
            plt.imshow(example_images[c], cmap='gray', interpolation='none')
            plt.title("GT:{},pred:{}".format(example_targets[c], predicts[c]))
            plt.xticks([])
            plt.yticks([])
        plt.show()


def viz_dataset(arg):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    example_images, example_targets = train_dataset._load_data()
    # fig = plt.figure()
    for i, c in enumerate(np.random.randint(0, 60000, arg.size**2)):
        plt.subplot(arg.size, arg.size, i+1)
        plt.tight_layout()
        plt.imshow(example_images[c], cmap='gray', interpolation='none')
        plt.title("GT: {}".format(example_targets[c]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', default='test', choices=['test', 'train'], type=str)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-c', '--ckpt', type=str)
    parser.add_argument('-s', '--size', default=5, type=int)

    arg = parser.parse_args()
    # path = './exp/ANN/val_ep15_bs100_lr0.001_Adam_ds4_2022_1127_161715/model.ckpt'
    if arg.mode == 'train':
        viz_dataset(arg)
    elif arg.mode == 'test':
        if arg.ckpt is None:
            assert False, "ckpt path is needed"
        viz_results(arg)

