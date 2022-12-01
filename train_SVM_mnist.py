import os
import argparse
import torch
import time
import datetime
import joblib
import torchvision
import torchvision.transforms as transforms
from sklearn import svm
from lib.utils.save_results import save_results_SVM


def SVM_worker(arg, save_dir):
    print("Preparing data...")
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data',
                                              train=False,
                                              transform=transforms.ToTensor())
    train_data, train_label = train_dataset._load_data()
    test_data, test_label = test_dataset._load_data()
    train_data = train_data.flatten(1, 2)
    test_data = test_data.flatten(1, 2)
    print("Start training...")
    svc = svm.SVC(C=arg.C, gamma=arg.gamma, kernel=arg.kernel_type)
    svc.fit(train_data, train_label)
    print("Starting testing...")
    test_result = svc.predict(test_data)
    test_result = torch.tensor(test_result)
    correct = (test_result == test_label).sum().item()
    acc = correct / test_label.shape[0]
    print("Accuracy:", acc)
    save_results_SVM(arg=arg, acc=acc, correct=correct, exp=save_dir)
    save_model = os.path.join(save_dir, 'model.m')
    joblib.dump(svc, save_model)
    save_acc = os.path.join(save_dir, 'acc.txt')
    with open(save_acc, 'w') as ff:
        ff.write("Correct:" + str(correct) + '\n')
        ff.write("Accuracy:" + str(acc) + '\n')
    print("-----successfully save results-----")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--C', type=float, default=1.0)
    # parser.add_argument('-m', '--mode', type=str, default='ovr', choices=['ovr', 'ovo'])
    parser.add_argument('-kt', '--kernel_type', type=str, default='linear',
                        choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'])
    parser.add_argument('-exp', '--exp_id', type=str)
    parser.add_argument('-g', '--gamma',type=str, default='scale')

    arg = parser.parse_args()

    start = time.time()
    if not os.path.exists('./exp'):
        os.mkdir('./exp')
    if not os.path.exists('./exp/SVM'):
        os.mkdir('./exp/SVM')

    print("<<<<<Beginning training SVM>>>>>")
    if arg.exp_id is not None:
        save_name = arg.exp_id
    else:
        datetime = datetime.datetime.now()
        save_name = f"{arg.kernel_type}_C{arg.C}_{arg.gamma}_" \
                    f"{datetime.year}_{datetime.month}{datetime.day}_{datetime.hour}{datetime.minute}{datetime.second}"
    save_dir = os.path.join('./exp/SVM', save_name)
    os.mkdir(save_dir)
    save_set = os.path.join(save_dir, 'setting.txt')
    argsDict = arg.__dict__
    with open(save_set, 'w') as f:
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ': ' + str(value) + '\n')

    SVM_worker(arg, save_dir)

    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    save_time_dir = os.path.join(save_dir, 'RunningTime.txt')
    with open(save_time_dir, 'w') as fff:
        fff.write('Running time: %s Seconds' % (end - start))