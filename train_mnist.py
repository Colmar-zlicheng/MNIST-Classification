import os
import torch
import torch.nn as nn
import time
import argparse
import datetime
import torchvision
import torchvision.transforms as transforms
from lib.model.MNIST import MNIST
from lib.utils.etqdm import etqdm
from lib.utils.misc import bar_perfixes

def SVM_worker(arg):
    return 0


def ANN_worker(arg, save_dir):
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    if arg.is_val is True:
        print("split train [60000] set into train+val as [50000, 10000]")
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=arg.batch_size,
                                                 shuffle=True)
    else:
        print("do not split train set")

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=arg.batch_size,
                                               shuffle=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=arg.batch_size,
                                              shuffle=False)

    model = MNIST(num_class=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=arg.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, arg.decay_step, arg.decay_gamma)
    print(f"Start training from epoch 0 to {arg.epoch_size}")
    for epoch_idx in range(arg.epoch_size):
        model.train()
        train_bar = etqdm(train_loader)
        for bidx, (images, labels) in enumerate(train_bar):
            # step_idx = epoch_idx * len(train_loader) + bidx

            pred, loss = model(images, labels)  # , step_idx, 'train')

            loss_show = ('%.12f' % loss)
            train_bar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} Loss {loss_show}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

    if arg.is_val is True:
        print("do validation")
        with torch.no_grad():
            correct = 0
            total = 0
            model.eval()
            val_bar = etqdm(val_loader)
            for bidx, (images, labels) in enumerate(val_bar):
                pred, loss = model(images, labels)
                loss_show = ('%.12f' % loss)
                val_bar.set_description(f"{bar_perfixes['val']} Loss {loss_show}")
                _, predicted = torch.max(pred.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                acc = 100 * correct / total
            print('Accuracy on val set: {} %'.format(acc))
            save_acc_path = os.path.join(save_dir, 'acc_val_txt')
            with open(save_acc_path, 'w') as ff_v:
                ff_v.write("Correct_val:" + str(correct) + '\n')
                ff_v.write("Accuracy_val:" + str(acc) + '\n')

    print("do test and compute accuracy")
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        test_bar = etqdm(test_loader)
        for bidx, (images, labels) in enumerate(test_bar):
            pred, loss = model(images, labels)
            loss_show = ('%.12f' % loss)
            test_bar.set_description(f"{bar_perfixes['test']} Loss {loss_show}")
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
        print('Accuracy on test set: {} %'.format(acc))
        save_acc_path = os.path.join(save_dir, 'acc_test_txt')
        with open(save_acc_path, 'w') as ff:
            ff.write("Correct_test:" + str(correct) + '\n')
            ff.write("Accuracy_test:" + str(acc) + '\n')

    print("beginning save checkpoints")
    save_path = os.path.join(save_dir, 'model.ckpt')
    torch.save(model.state_dict(), save_path)
    print("successfully save checkpoints")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='ANN', choices=['SVM,ANN'])
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-e', '--epoch_size', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-ds', '--decay_step', type=int, default=5)
    parser.add_argument('-dg', '--decay_gamma', type=float, default=0.1)
    parser.add_argument('-v', '--is_val', action='store_true', help="whether do validation and split train set")

    arg = parser.parse_args()
    print(f"Start training with {arg.type}")
    datetime = datetime.datetime.now()
    start = time.time()
    if arg.type == 'SVM':
        save_dir = './exp/SVM'
        SVM_worker(arg)
    elif arg.type == 'ANN':
        save_name = f"train_ep{arg.epoch_size}_bs{arg.batch_size}_lr{arg.learning_rate}_" \
                    f"ds{arg.decay_step}dg{arg.decay_gamma}_" \
                    f"{datetime.year}_{datetime.month}{datetime.day}_{datetime.hour}{datetime.minute}"
        save_dir = os.path.join('./exp/ANN', save_name)
        os.mkdir(save_dir)
        hype_dir = os.path.join(save_dir, 'Hyperparameters.txt')
        with open(hype_dir,'w') as f:
            f.write("ANN_Hyperparameters:" + '\n')
            f.write("epoch_size:" + str(arg.epoch_size) + '\n')
            f.write("batch_size:" + str(arg.batch_size) + '\n')
            f.write("decay_step:" + str(arg.decay_step) + '\n')
            f.write("decay_gamma:" + str(arg.decay_gamma) + '\n')
            f.write("is_do_validation:" + str(arg.is_val) + '\n')
        ANN_worker(arg, save_dir)
    else:
        raise ValueError("{} is not supported ".format(arg.type))
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    save_time_dir = os.path.join(save_dir, 'RunningTime.txt')
    with open(save_time_dir, 'w') as fff:
        fff.write('Running time: %s Seconds' % (end - start))