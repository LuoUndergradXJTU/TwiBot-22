import json
import os
import os.path as osp
from argparse import ArgumentParser
import pandas
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score


def metrics(y:  list, pred:  list):
    plog = "ACC: {}".format(accuracy_score(y, pred)) + '\n' + \
           "ROC: {}".format(roc_auc_score(y, pred)) + '\n' + \
           "F1: {}".format(f1_score(y, pred)) + '\n' + \
           "Precision: {}".format(precision_score(y, pred)) + '\n' + \
           "Recall: {}\n".format(recall_score(y, pred))
    print(plog)
    return plog


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--threshold', type=float, default=0.75)
    parser.add_argument('--type', type=str, default='english')
    args = parser.parse_args()
    u_type = args.type
    threshold = args.threshold
    path = 'tmp/output/{}.json'.format(args.dataset_name)
    if not osp.exists(path):
        raise ValueError
    pred = json.load(open(path))
    label = pandas.read_csv('../../datasets/{}/label.csv'.format(args.dataset_name))
    labels = {}
    for index, item in label.iterrows():
        labels[item['id']] = (item['label'] == 'bot')
    all_pred = []
    all_label = []
    for item in pred:
        all_pred.append(int(item[u_type] >= threshold))
        all_label.append(labels[item['id']])
    plog = metrics(all_label, all_pred)
    save_path = 'tmp/result/'
    if not osp.exists(save_path):
        os.makedirs(save_path)
    with open(osp.join(save_path, '{}.txt'.format(args.dataset_name)), 'w') as f:
        f.write(plog)






