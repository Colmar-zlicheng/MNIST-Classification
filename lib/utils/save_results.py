import csv


def save_results_ANN(arg, val_acc, test_acc, exp):
    log_path = './results/ANN_results.csv'
    file = open(log_path, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(file)

    total = len(open(log_path).readlines())
    if total == 0:
        csv_writer.writerow(['ID', 'Epoch_size', 'Batch_size', 'learning_rate',
                             'decay_step', 'decay_gamma', 'weight_decay',
                             'optimizer_type', 'sgd_momentum', 'do_val(split train)',
                             'val_acc', 'test_acc', 'exp'])
        total += 1
    if arg.optimizer_type == 'SGD':
        sgd_momentum = arg.sgd_momentum
    else:
        sgd_momentum = '--'
    if arg.is_val is True:
        do_val = 'True'
    else:
        do_val = 'False'
        val_acc = '--'
    if arg.decay_step > arg.epoch_size:
        decay_step = '--'
    else:
        decay_step = arg.decay_step
    csv_writer.writerow([str(total), str(arg.epoch_size), str(arg.batch_size), str(arg.learning_rate),
                         str(decay_step), str(arg.decay_gamma), str(arg.weight_decay),
                         str(arg.optimizer_type), sgd_momentum, do_val,
                         str(val_acc)+'%', str(test_acc)+'%', str(exp)])
    file.close()


if __name__ == '__main__':
    import argparse
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
    arg = parser.parse_args()
    save_results_ANN(arg, 50, 30, './test/path')
