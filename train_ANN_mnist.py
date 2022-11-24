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
from lib.utils.save_results import save_results_ANN
from torch.utils.tensorboard import SummaryWriter


def save_Hyperparameters(arg):
    if arg.is_val is True:
        run_type = 'val'
    else:
        run_type = 'train'
    save_name = f"{run_type}_ep{arg.epoch_size}_bs{arg.batch_size}_lr{arg.learning_rate}_" \
                f"{arg.optimizer_type}_ds{arg.decay_step}_" \
                f"{datetime.year}_{datetime.month}{datetime.day}_{datetime.hour}{datetime.minute}{datetime.second}"
    save_dir = os.path.join('./exp/ANN', save_name)
    os.mkdir(save_dir)
    hype_dir = os.path.join(save_dir, 'Hyperparameters.txt')
    with open(hype_dir, 'w') as f:
        f.write("ANN_Hyperparameters:" + '\n')
        f.write("epoch_size:" + str(arg.epoch_size) + '\n')
        f.write("batch_size:" + str(arg.batch_size) + '\n')
        f.write("learning_rate:" + str(arg.learning_rate) + '\n')
        f.write("decay_step:" + str(arg.decay_step) + '\n')
        f.write("decay_gamma:" + str(arg.decay_gamma) + '\n')
        f.write("weight_decay:" + str(arg.weight_decay) + '\n')
        f.write("optimizer_type:" + str(arg.optimizer_type) + '\n')
        if arg.optimizer_type == 'SGD':
            f.write("SGD_momentum:" + str(arg.sgd_momentum) + '\n')
        f.write("is_do_validation:" + str(arg.is_val) + '\n')
    return save_dir


def ANN_worker(arg, save_dir, summary):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists('./data'):
        os.mkdir('./data')
    torch.manual_seed(0)
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

    model = MNIST(num_class=10).to(device)
    if arg.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay)
    elif arg.optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=arg.learning_rate, weight_decay=arg.weight_decay)
    elif arg.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=arg.learning_rate, momentum=arg.sgd_momentum, weight_decay=arg.weight_decay)
    else:
        raise ValueError(f"no such optimizer_type:{arg.optimizer_type}")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, arg.decay_step, arg.decay_gamma)
    print(f"<<<<<Start training from epoch 0 to {arg.epoch_size}>>>>>")
    for epoch_idx in range(arg.epoch_size):
        model.train()
        train_bar = etqdm(train_loader)
        correct = 0
        total = 0
        for bidx, (images, labels) in enumerate(train_bar):
            step_idx = epoch_idx * len(train_loader) + bidx
            pred, loss = model(images.to(device), labels.to(device))  # , step_idx, 'train')

            loss_show = ('%.12f' % loss)
            train_bar.set_description(f"{bar_perfixes['train']} Epoch {epoch_idx} Loss {loss_show}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_idx % arg.log_interval == 0:
                summary.add_scalar(f"scalar/loss", loss, global_step=step_idx, walltime=None)
            _, predicted = torch.max(pred.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = 100 * correct / total
        summary.add_scalar(f"scalar/train_acc", train_acc, global_step=epoch_idx, walltime=None)
        scheduler.step()
        print(f"Current LR: {[group['lr'] for group in optimizer.param_groups]}")

        if arg.is_val is True:
            # print("do validation")
            if epoch_idx % arg.eval_interval == 0:
                with torch.no_grad():
                    correct = 0
                    total = 0
                    model.eval()
                    val_bar = etqdm(val_loader)
                    for bidx, (images, labels) in enumerate(val_bar):
                        pred, loss = model(images.to(device), labels.to(device))
                        loss_show = ('%.12f' % loss)
                        val_bar.set_description(f"{bar_perfixes['val']} Loss {loss_show}")
                        _, predicted = torch.max(pred.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_acc = 100 * correct / total
                    print('Current Accuracy on val set: Epoch {}, {} %'.format(epoch_idx, val_acc))
                    save_acc_path = os.path.join(save_dir, 'acc_val_txt')
                    with open(save_acc_path, 'a') as ff_v:
                        ff_v.write("Epoch:" + str(epoch_idx) + '\n')
                        ff_v.write("Correct_val:" + str(correct) + '\n')
                        ff_v.write("Accuracy_val:" + str(val_acc) + '\n')
                        ff_v.write('\n')
                    summary.add_scalar(f"scalar/val_acc", val_acc, global_step=epoch_idx, walltime=None)

    print("do test and compute accuracy")
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        test_bar = etqdm(test_loader)
        for bidx, (images, labels) in enumerate(test_bar):
            pred, loss = model(images.to(device), labels.to(device))
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

    print("-----beginning save checkpoints and results-----")
    save_path = os.path.join(save_dir, 'model.ckpt')
    torch.save(model.state_dict(), save_path)
    print("-----successfully save checkpoints-----")
    save_results_ANN(arg=arg, val_acc=val_acc, test_acc=acc, exp=save_dir)
    print("-----successfully save results-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=100)
    parser.add_argument('-e', '--epoch_size', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-ds', '--decay_step', type=int, default=6)
    parser.add_argument('-dg', '--decay_gamma', type=float, default=0.1)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-ot', '--optimizer_type', type=str, default='Adam', choices=['Adam', 'AdamW', 'SGD'])
    parser.add_argument('-sm', '--sgd_momentum', type=float, default=0.0)
    parser.add_argument('-v', '--is_val', action='store_true', help="whether do validation and split train set")
    parser.add_argument('-log', '--log_interval', type=int, default=50)
    parser.add_argument('-eval', '--eval_interval', type=int, default=1)

    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./exp/ANN'):
        os.mkdir('./exp/ANN')

    arg = parser.parse_args()
    print("<<<<<Start training with ANN>>>>")
    datetime = datetime.datetime.now()
    start = time.time()

    save_dir = save_Hyperparameters(arg)

    summary_dir = os.path.join(save_dir, 'run')
    os.mkdir(summary_dir)
    summary = SummaryWriter(summary_dir)

    ANN_worker(arg=arg, save_dir=save_dir, summary=summary)

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    save_time_dir = os.path.join(save_dir, 'RunningTime.txt')
    with open(save_time_dir, 'w') as fff:
        fff.write('Running time: %s Seconds' % (end - start))
