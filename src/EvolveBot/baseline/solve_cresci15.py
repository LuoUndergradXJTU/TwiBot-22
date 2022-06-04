from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from pathlib import Path
import torch
import csv
import argparse

parser = argparse.ArgumentParser(description='Random seed set:')
parser.add_argument('random_seed',  type=int, help='random seed')
args = parser.parse_args()

root_dir = Path.cwd().parent.parent.parent
dataset_dir = root_dir / 'datasets'
cresci_15 = dataset_dir / 'cresci-2015'
data = torch.load(cresci_15 / 'user_info.pt')


feature = np.load('cresci-15_feature.npy')
x_train = []
y_train = []
x_test = []
y_test = []
for i in data['train_uid_with_label'].index:
    x_train.append(feature[i])
    y_train.append(data['labels'][i].item())
for i in data['test_uid_with_label'].index:
    x_test.append(feature[i])
    y_test.append(data['labels'][i].item())  



clf = RandomForestClassifier(random_state=args.random_seed)
clf.fit(X=x_train, y=y_train)
y_pred = clf.predict(x_test)
print('acc: ', accuracy_score(y_true=y_test, y_pred=y_pred))
print('precision: ', precision_score(y_true=y_test, y_pred=y_pred))
print('recall: ', recall_score(y_true=y_test, y_pred=y_pred))
print('f1: ', f1_score(y_true=y_test, y_pred=y_pred))
print('auc: ', roc_auc_score(y_true=y_test, y_score=y_pred))
