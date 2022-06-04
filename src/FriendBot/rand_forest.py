import os
import json
import ijson
import argparse
import pandas as pd
import numpy as np
from feature_engineering import feature_preprocess
from feature_twibot22 import preprocess
from feature_supplement import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-20', help='Choose the dataset.')
arg = parser.parse_args()
DATASET = arg.dataset

if DATASET == 'Twibot-22':
    label = pd.read_csv('../../datasets/Twibot-22/label.csv')
    split = pd.read_csv('../../datasets/Twibot-22/split.csv')
    user = list(ijson.items(open('../../datasets/Twibot-22/user.json', 'r'), 'item'))
    author = []
    tid = []
    tweet = []
    for i in range(9):
        tweet = tweet + list(ijson.items(open('../../datasets/Twibot-22/tweet_' + str(i) + '.json', 'r'), 'item.text'))
        tid = tid + list(ijson.items(open('../../datasets/Twibot-22/tweet_' + str(i) + '.json', 'r'), 'item.id'))
        author = author + list(ijson.items(open('../../datasets/Twibot-22/tweet_' + str(i) + '.json', 'r'), 'item.author_id'))
    edge = pd.read_csv('../../datasets/Twibot-22/edge.csv')
    id_tweet = dict()
    id_map = dict()
    num_user = len(user)
    for i in range(num_user):
        id_map[user[i]['id']] = i
    for i in range(len(tid)):
        if id_map[author[i]] in id_tweet.keys():
            id_tweet[id_map[author[i]]].append(tweet[i])
        else:
            id_tweet[id_map[author[i]]] = [tweet[i]]
    label_order = np.array(label['label'].values)
    split_order = np.array(split['split'].values)
    for i in range(num_user):
        label_order[id_map[label['id'][i]]] = label['label'][i]
        split_order[id_map[split['id'][i]]] = split['split'][i]
    y = (label_order == 'bot').astype(int)
    train_split = split_order[0: num_user] == 'train'
    val_split = split_order[0: num_user] == 'valid'
    test_split = split_order[0: num_user] == 'test'
    train_set = np.where(split_order == 'train')[0]
    val_set = np.where(split_order == 'valid')[0]
    test_set = np.where(split_order == 'test')[0]
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    if os.path.exists('feature_matrix_Twibot-22.csv'):
        X = pd.read_csv('feature_matrix_Twibot-22.csv').values
    else:
        X = preprocess(user, tid, author, edge, id_tweet)
    print(f"X shape: {X.shape}")
    for i in range(X.shape[0]):
        X[i][np.isnan(X[i])] = np.nanmean(X[i])
else:
    node = json.load(open('../../datasets/' + DATASET + '/node.json', 'r'))
    label = pd.read_csv('../../datasets/' + DATASET + '/label.csv')
    split = pd.read_csv('../../datasets/' + DATASET + '/split.csv')
    edge = pd.read_csv('../../datasets/' + DATASET + '/edge.csv')
    id_map = dict()
    for i in range(len(node)):
        id_map[node[i]['id']] = i
    num_user = label.shape[0]
    label_order = np.array(label['label'].values)
    split_order = np.array(split['split'].values)
    for i in range(num_user):
        label_order[id_map[label['id'][i]]] = label['label'][i]
        split_order[id_map[split['id'][i]]] = split['split'][i]
    y = (label_order == 'bot').astype(int)
    train_split = split_order[0: num_user] == 'train'
    val_split = split_order[0: num_user] == 'val'
    test_split = split_order[0: num_user] == 'test'
    train_set = np.where(split_order == 'train')[0]
    val_set = np.where(split_order == 'val')[0]
    test_set = np.where(split_order == 'test')[0]
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    if os.path.exists('feature_matrix_' + DATASET + '.csv'):
        X = pd.read_csv('feature_matrix_' + DATASET + '.csv').values[0: num_user]
    else:
        X = feature_preprocess(node, edge, DATASET)[0: num_user]
        # X = preprocessing(node, edge, DATASET)[0: num_user]
    print(f"X shape: {X.shape}")
    for i in range(X.shape[0]):
        X[i][np.isnan(X[i])] = np.nanmean(X[i])

acc = []
precision = []
recall = []
f1 = []
auc = []

clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=0)
clf.fit(X[train_set], y[train_set])
test_result = clf.predict(X[test_set])
print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
acc.append(accuracy_score(y[test_set], test_result))
precision.append(precision_score(y[test_set], test_result))
recall.append(recall_score(y[test_set], test_result))
f1.append(f1_score(y[test_set], test_result))
auc.append(roc_auc_score(y[test_set], test_result))

clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=100)
clf.fit(X[train_set], y[train_set])
test_result = clf.predict(X[test_set])
print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
acc.append(accuracy_score(y[test_set], test_result))
precision.append(precision_score(y[test_set], test_result))
recall.append(recall_score(y[test_set], test_result))
f1.append(f1_score(y[test_set], test_result))
auc.append(roc_auc_score(y[test_set], test_result))

clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=200)
clf.fit(X[train_set], y[train_set])
test_result = clf.predict(X[test_set])
print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
acc.append(accuracy_score(y[test_set], test_result))
precision.append(precision_score(y[test_set], test_result))
recall.append(recall_score(y[test_set], test_result))
f1.append(f1_score(y[test_set], test_result))
auc.append(roc_auc_score(y[test_set], test_result))

clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=300)
clf.fit(X[train_set], y[train_set])
test_result = clf.predict(X[test_set])
print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
acc.append(accuracy_score(y[test_set], test_result))
precision.append(precision_score(y[test_set], test_result))
recall.append(recall_score(y[test_set], test_result))
f1.append(f1_score(y[test_set], test_result))
auc.append(roc_auc_score(y[test_set], test_result))

clf = RandomForestClassifier(oob_score=True, bootstrap=True, random_state=400)
clf.fit(X[train_set], y[train_set])
test_result = clf.predict(X[test_set])
print(f"acc: {accuracy_score(y[test_set], test_result):.4f}, precision: {precision_score(y[test_set], test_result):.4f}, recall: {recall_score(y[test_set], test_result):.4f}, f1-score: {f1_score(y[test_set], test_result):.4f}, roc_auc: {roc_auc_score(y[test_set], test_result):.4f}")
acc.append(accuracy_score(y[test_set], test_result))
precision.append(precision_score(y[test_set], test_result))
recall.append(recall_score(y[test_set], test_result))
f1.append(f1_score(y[test_set], test_result))
auc.append(roc_auc_score(y[test_set], test_result))

if not os.path.exists('results'):
    os.mkdir('results')
results = pd.DataFrame({'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc})
results.to_csv('results/' + DATASET + '.csv', index=False)
