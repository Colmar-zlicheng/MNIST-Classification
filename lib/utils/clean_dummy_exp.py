import os
import shutil
import csv

ANN_exp_path = './exp/ANN'
ANN_result_path = './results/ANN_results.csv'


def main():
    ANN_exp_dir = os.listdir(ANN_exp_path)
    f = open(ANN_result_path, 'r')
    ANN_csv = csv.reader(f)
    exp_list = []
    for row in ANN_csv:
        exp_list.append(row[-1])
    pop_id = ANN_exp_dir.index('.DS_Store')  # hide file for mac
    ANN_exp_dir.pop(pop_id)
    # print(exp_list)
    for idir in ANN_exp_dir:
        dir = os.path.join('./exp/ANN', idir)
        need_to_clean = False
        if dir not in exp_list:
            need_to_clean = True
        if need_to_clean is True:
            print(f"remove {dir}")
            input("Confirm ?")
            shutil.rmtree(dir)


if __name__ == '__main__':
    # clean the dummy exp not in ANN_results.csv
    main()
    print('Successfully clean dummy exp!')