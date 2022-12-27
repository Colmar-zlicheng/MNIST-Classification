# MNIST-Classification

## ANN and SVM for MNIST Classification
Project 2 for CS3317-2@SJTU

### requirements
```shell
torch
torchvision
tensorboard
sklearn
numpy
joblib
datetime
```

### data
```
torchvision.datasets.MNIST()
```
Datas will be download into ./data

### SVM
Recommend to use small dataset to reduce running time(default). 

The default setting is the best performance, you can set hyperparameters in terminal.

#### run:
```shell
python train_SVM_mnist.py
```
or:
```shell
python train_SVM_mnist.py -c 11.0 -kt rbf
```
Models and relevant information will be saved into ./exp/SVM

### ANN
When do validation, the train set will be split into train+val set
(from 60000 examples into 50000+10000 examples).

The default setting is the best performance, you can set hyperparameters in terminal.
#### run:
```shell
python train_ANN_mnist.py -v
```
or:
```shell
python train_ANN_mnist.py -v -b 100 -e 15 -lr 0.001 -ds 5 -dg 0.1 -wd 0.0 -ot Adam
```

When just do train and test, the whole train set with 60000 examples will used.
#### run:
```shell
python train_ANN_mnist.py
```
or:
```shell
python train_ANN_mnist.py -b 100 -e 15 -lr 0.001 -ds 5 -dg 0.1 -wd 0.0 -ot Adam
```
Models and relevant information will be saved into ./exp/ANN
### Results
All results has been saved in ./results/SVM_results.csv and ./results/ANN_results.csv, 
you can delete the folder to run your own results or run directly(it will continue writing your results to the original file).

### Utils
#### clean dummy exp
This scripts can clean all the dummy exp whose results not saved in .csv
```shell
python lib/utils/clean_dummy_exp.py
```
#### viz results
You can use this scripts to viz dataset or your results.
For example:
```shell
python lib/utils/viz_tools.py -t ANN -m test -c exp/ANN/train_ep15_bs100_lr0.001_Adam_ds5_2022_1218_19482/model.ckpt
```
You might meet some problem with PYTHONPATH when run this script, please see the top of this script.
#### tensorboard(only for ANN)
Tensorboard is available in this project to supervise training in real time 
or review the training process.