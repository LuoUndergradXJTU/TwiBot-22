from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from pathlib import Path
import torch
import csv
import pandas as pd
import argparse


parser = argparse.ArgumentParser(description='Random seed set:')
parser.add_argument('random_seed',  type=int, help='random seed')
args = parser.parse_args()

root_dir = Path.cwd().parent.parent.parent
dataset_dir = root_dir / 'datasets'
twibot_20 = dataset_dir / 'Twibot-20'
label = pd.read_csv(twibot_20 / 'label.csv')
split = pd.read_csv(twibot_20 / 'split.csv')
uid_to_user_index = {}
uid_to_user_type = {}
cnt = 0
for index, row in split.iterrows():
    if row['split'] == 'train' or row['split'] == 'test':
        uid_to_user_index[row['id']] = cnt
        if row['split'] == 'train':
            uid_to_user_type[row['id']] = 1
        else:
            uid_to_user_type[row['id']] = 2
        cnt += 1
    if row['split'] == 'support':
        break
train_num = 8278
test_num = 1183

tmp_feature = np.load('twibot_20_feature.npy')
feature = np.zeros(shape=(train_num+test_num, 7))
feature[:,0] = tmp_feature[:,0]
feature[:,1:6] = tmp_feature[:,2:7]
feature[:,6] = tmp_feature[:,10]
# feature = np.load('twibot_20_feature.npy')
feature_mean = np.mean(feature, axis=0)
feature_std = np.std(feature, axis=0)
feature = (feature-feature_mean) / feature_std
x_train = np.zeros(shape=(train_num, 7))
y_train = np.zeros(shape=train_num)
x_test = np.zeros(shape=(test_num, 7))
y_test = np.zeros(shape=test_num)
for index, row in label.iterrows():
    if not row['id'] in uid_to_user_index:
        continue
    index = uid_to_user_index[row['id']]
    if uid_to_user_type[row['id']] == 1:
        x_train[index] = feature[index]
        y_train[index] = 0 if row['label'] == 'human' else 1
    else:
        x_test[index - train_num] = feature[index]
        y_test[index - train_num] = 0 if row['label'] == 'human' else 1

clf = RandomForestClassifier(random_state=args.random_seed)
clf.fit(X=x_train, y=y_train)
y_pred = clf.predict(x_test)
print('acc: ', accuracy_score(y_true=y_test, y_pred=y_pred))
print('precision: ', precision_score(y_true=y_test, y_pred=y_pred))
print('recall: ', recall_score(y_true=y_test, y_pred=y_pred))
print('f1: ', f1_score(y_true=y_test, y_pred=y_pred))
print('auc: ', roc_auc_score(y_true=y_test, y_score=y_pred))