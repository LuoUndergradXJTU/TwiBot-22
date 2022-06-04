import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


def lcs(dna:  str) -> None:
    global lcs_string
    global user_list
    global genes

    cnt = 0

    for user in user_list.keys():
        if dna in user_list[user]:
            cnt += 1

    if cnt >= len(lcs_string[cnt]):
        lcs_string[cnt] = dna

    if cnt <= 1:
        return

    if len(dna) >= 998:
        return

    for gene in genes:
        lcs(dna + gene)

if __name__ == '__main__':
    dataset_path = './tmp/datasets_testset'
    dataset_list = os.listdir(dataset_path)
    # dataset_list = ['Twibot-20_DNA.txt']

    genes = ['A', 'C', 'T']

    for dataset_name in dataset_list:
        if dataset_name[-8:] != '_DNA.txt':
            continue
        print(dataset_name)
        user_list = {}
        user_label = {}
        y = []
        pred = []

        f = open('./{0}/{1}'.format(dataset_path, dataset_name))
        for line in f:
            if len(line.split()) == 2:
                user_list[line.split()[0]] = line.split()[1]
            else:
                user_list[line.split()[0]] = ''
        f.close()

        # Label
        f = pd.read_csv('../../datasets/{}/label.csv'.format(dataset_name[:-8]))
        for item_id, item in enumerate(f['id']):
            user_label[item] = 1 if f['label'][item_id] == 'bot' else 0
        del f

        lcs_string = [''] * (len(user_list) + 1)

        # print('{}_LCS.txt'.format(dataset_name))
        # print(os.listdir('./tmp/datasets_testset/'))
        # print('{}_LCS.txt'.format(dataset_name) in os.listdir('./tmp/datasets_testset/'))

        if '{}_LCS.txt'.format(dataset_name[:-8]) in os.listdir('./tmp/datasets_testset/'):
            print('{} LCS exists.'.format(dataset_name[:-8]))
            f = open('./tmp/datasets_testset/{}_LCS.txt'.format(dataset_name[:-8]))
            for line in f:
                try:
                    lcs_string[int(line.split()[0])] = line.split()[1]
                except:
                    lcs_string[int(line.split()[0])] = ''
        else:
            for gene in genes:
                lcs(gene)

        for idx, item in enumerate(lcs_string):
            for i in range(idx + 1, len(lcs_string)):
                if len(lcs_string[i]) > len(lcs_string[idx]):
                    lcs_string[idx] = lcs_string[i]

        w = open('./tmp/datasets_testset/{}_LCS.txt'.format(dataset_name[:-8]), 'w')
        for item_id, item in enumerate(lcs_string):
            w.write('{} {}\n'.format(str(item_id), item))
        w.close()

        max_delta = [-1, -1]  # Max delta & Position
        for i in range(1, len(lcs_string) - 1):
            if len(lcs_string[i]) - len(lcs_string[i + 1]) >= max_delta[0]:
                max_delta[0] = len(lcs_string[i]) - len(lcs_string[i + 1])
                max_delta[1] = i
        bot_sub_string = lcs_string[max_delta[1]]
        # print(bot_sub_string)
        # print('max delta: {}'.format(max_delta))

        for uid in user_list.keys():
            if bot_sub_string in user_list[uid]:
                y.append(1)
            else:
                y.append(0)
            pred.append(user_label[uid])

        print("ACC: {}".format(accuracy_score(y, pred)))
        print("ROC: {}".format(roc_auc_score(y, pred)))
        print("F1: {}".format(f1_score(y, pred)))
        print("Precision: {}".format(precision_score(y, pred)))
        print("Recall: {}\n".format(recall_score(y, pred)))

