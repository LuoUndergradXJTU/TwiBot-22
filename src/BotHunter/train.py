import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from argparse import ArgumentParser
import json
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset = args.dataset

assert dataset in ['Twibot-22', 'Twibot-20', 'midterm-2018', 'gilani-2017',
                   'cresci-stock-2018', 'cresci-rtbust-2019', 'cresci-2017',
                   'cresci-2015', 'botometer-feedback-2019']

split = pd.read_csv('../../datasets/{}/split.csv'.format(dataset))
idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
idx = {item: index for index, item in enumerate(idx)}
features = np.load('tmp/{}/features.npy'.format(dataset), allow_pickle=True)
labels = np.load('tmp/{}/labels.npy'.format(dataset))

train_idx = []
val_idx = []
test_idx = []

for index, item in tqdm(split.iterrows(), ncols=0):
    try:
        if item['split'] == 'train':
            train_idx.append(idx[item['id']])
        if item['split'] == 'val' or item['split'] == 'valid':
            val_idx.append(idx[item['id']])
        if item['split'] == 'test':
            test_idx.append(idx[item['id']])
    except KeyError:
        continue

print('loading done')

print(len(train_idx))
print(len(val_idx))
print(len(test_idx))

if __name__ == '__main__':
    train_x = features[train_idx]
    train_y = labels[train_idx]
    val_x = features[val_idx]
    val_y = labels[val_idx]
    test_x = features[test_idx]
    test_y = labels[test_idx]
    print('training......')
    cls = RandomForestClassifier(n_estimators=100)
    cls.fit(train_x, train_y)
    print('done.')

    val_pred = cls.predict(val_x)
    test_pred = cls.predict(test_x)

    val_acc = accuracy_score(val_y, val_pred)
    val_f1 = f1_score(val_y, val_pred)
    val_recall = recall_score(val_y, val_pred)
    val_precision = precision_score(val_y, val_pred)

    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred)
    test_recall = recall_score(test_y, test_pred)
    test_precision = precision_score(test_y, test_pred)

    print('validation:')
    print('acc:', val_acc)
    print('f1:', val_f1)
    print('recall:', val_recall)
    print('precision:', val_precision)

    print('test:')
    print('acc:', test_acc)
    print('f1:', test_f1)
    print('recall:', test_recall)
    print('precision:', test_precision)