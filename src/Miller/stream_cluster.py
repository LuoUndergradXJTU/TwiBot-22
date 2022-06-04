import os
import json
import ijson
import pandas as pd
import numpy as np
import argparse
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Twibot-20', help='Choose the dataset.')
    arg = parser.parse_args()
    dataset = arg.dataset
    num_cluster = 0
    k = 0
    epsilon = 0
    dbscan = False

    if dataset == 'botometer-feedback-2019':
        k = 0.0208125
        num_cluster = 3
        dbscan = False
    elif dataset == 'cresci-2015':
        k = 0.235
        num_cluster = 5
        dbscan = False
    elif dataset == 'cresci-2017':
        k = 0.3
        num_cluster = 5
        dbscan = False
    elif dataset == 'cresci-rtbust-2019':
        k = 0.4
        num_cluster = 3
        dbscan = True
        epsilon = 2000
    elif dataset == 'cresci-stock-2018':
        k = 0.08
        num_cluster = 5
        dbscan = True
        epsilon = 3500
    elif dataset == 'gilani-2017':
        k = 0.2
        num_cluster = 5
        dbscan = True
        epsilon = 22000
    elif dataset == 'midterm-2018':
        k = 100
        num_cluster = 10
        dbscan = False
    elif dataset == 'Twibot-20':
        k = 0.024
        num_cluster = 10
        dbscan = False
    elif dataset == 'Twibot-22':
        k = 0.1
        num_cluster = 100
        dbscan = False
    else:
        raise ValueError(dataset + ' doesn\'t exist.')

    node = json.load(open('../../datasets/' + dataset + '/node.json', 'r'))
    # node = list(ijson.items(open('../../datasets/' + dataset + '/user.json', 'r'), 'item'))
    label = pd.read_csv('../../datasets/' + dataset + '/label.csv')
    split = pd.read_csv('../../datasets/' + dataset + '/split.csv')
    id_map = dict()
    num_user = 0
    ind_first = True
    for i in range(len(node)):
        id_map[node[i]['id']] = i
        if node[i]['id'][0] == 't' and ind_first:
            num_user = i
            ind_first = False
    if num_user == 0:
        num_user = len(node)
    label_order = np.array(label['label'].values)
    split_order = np.array(split['split'].values)
    for i in range(num_user):
        label_order[id_map[label['id'][i]]] = label['label'][i]
        split_order[id_map[split['id'][i]]] = split['split'][i]
    if os.path.exists('X_' + dataset + '.csv'):
        X = pd.read_csv('X_' + dataset + '.csv').values
    else:
        raise RuntimeError('Feature dosen\'t exist.')
    y = (label_order == 'bot').astype(int)
    train_split = split_order[0: num_user] == 'train'
    if dataset == 'Twibot-22':
        val_split = split_order[0: num_user] == 'valid'
    else:
        val_split = split_order[0: num_user] == 'val'
    test_split = split_order[0: num_user] == 'test'
    train_set = np.where(split_order == 'train')[0]
    if dataset == 'Twibot-22':
        val_set = np.where(split_order == 'valid')[0]
    else:
        val_set = np.where(split_order == 'val')[0]
    test_set = np.where(split_order == 'test')[0]
    print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")
    X_train_human = X[train_split | val_split][~y[train_split | val_split].astype(bool)]
    kmeans = KMeans(n_clusters=num_cluster, random_state=0)
    result = kmeans.fit(X_train_human)
    cluster = []
    for i in range(num_cluster):
        cluster.append(np.where(result.labels_ == i)[0])
    radius = []
    mean_dist = []
    for i in range(num_cluster):
        max_dist = 0
        sum_dist = 0
        for j in cluster[i]:
            max_dist = max(max_dist, np.linalg.norm(X_train_human[j] - result.cluster_centers_[i]))
            sum_dist += np.linalg.norm(X_train_human[j] - result.cluster_centers_[i])
        radius.append(max_dist)
        mean_dist.append(sum_dist / len(cluster[i]))
    eps = k * sum(radius) / num_cluster
    predict_y = []
    for i in range(len(test_set)):
        sign = 0
        for j in range(num_cluster):
            if np.linalg.norm(X[test_set[i]] - result.cluster_centers_[j]) < eps:
                sign = 1
        if sign == 1:
            predict_y.append(1)
        else:
            predict_y.append(0)
    predict = np.array(predict_y)
    if dbscan:
        predict_y = np.array(predict_y)
        X_train_nd = np.vstack((X_train_human, X[np.array(test_set)[~predict_y.astype(bool)]]))
        dbs = DBSCAN(eps=epsilon, min_samples=num_user // 50)
        result_2 = dbs.fit(X_train_nd)
        predict_2 = result_2.fit_predict(X[np.array(test_set)[predict_y.astype(bool)]])
        map_1 = []
        for i in range(len(predict_y.astype(bool))):
            if predict_y.astype(bool)[i]:
                map_1.append(i)
        for i in range(len(predict_2)):
            if predict_2[i] >= 0:
                predict[map_1[i]] = 0
    print(
        f"acc: {accuracy_score(y[test_set], predict):.4f}, precision: {precision_score(y[test_set], predict):.4f}, recall: {recall_score(y[test_set], predict):.4f}, f1-score: {f1_score(y[test_set], predict):.4f}, roc_auc: {roc_auc_score(y[test_set], predict):.4f}")


if __name__ == '__main__':
    main()

