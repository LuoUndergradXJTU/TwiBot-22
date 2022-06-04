import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pandas
import json
import torch_geometric.transforms as T
from torch_geometric.nn import ChebConv, GCNConv  # noqa
from tqdm import tqdm
import csv
import json
import ijson
path = '../../datasets'
dataset1 = 'Twibot-22'
path1 = os.path.join(path, dataset1)

u=0
with open(os.path.join(path1, 'user.json'), 'r', encoding = 'UTF-8') as f:
    user1 = json.load(f)
split1 = pandas.read_csv(os.path.join(path1, 'split.csv'))
label1 = pandas.read_csv(os.path.join(path1, 'label.csv'))
train_id = dict()
test_id = dict()
val_id = dict()
train=0
val=0
test=0
for index, node in tqdm(split1.iterrows()):
    if node['split'] == 'train':
        # train_id.append(userid.index(node['id']))
        train_id[node['id']]=train
        train+=1
    if node['split'] == 'test':
        # test_id.append(userid.index(node['id']))
        test_id[node['id']]=test
        test+=1
    if node['split'] == 'valid':
        # val_id.append(userid.index(node['id']))
        val_id[node['id']]=val
        val+=1
print(val)
for index, node in tqdm(label1.iterrows()):
    if node['label'] == 'bot':
        label.append(1)

    if node['label'] == 'human':
        label.append(0)
        
for user in tqdm(user1):
    if user['id'][0]=='u':
        u+=1
feature=np.zeros((u, 5))
print(u)
i = 0
with open(os.path.join(path1, 'user.json'), 'r', encoding = 'UTF-8') as f:
    stop_time = 1601510400.0
    node_id_index_dict = {}
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
     if x['id'][0]=='u':
        id = x["id"].strip()
        node_id_index_dict[id] = i
        feature[i][3] = (
            x["public_metrics"]["followers_count"]
            if x["public_metrics"]["followers_count"] is not None
            else 0
        )
        feature[i][4] = (
            x["public_metrics"]["following_count"]
            if x["public_metrics"]["following_count"] is not None
            else 0
        )
        feature[i][1] = (
            x["public_metrics"]["tweet_count"]
            if x["public_metrics"]["tweet_count"] is not None
            else 0
        )
        time_seen = x["created_at"]
        if time_seen is not None:
           real_time = stop_time - time.mktime(time.strptime(time_seen, "%Y-%m-%d %H:%M:%S+00:00"))
        else:
            real_time = 0
        feature[i][0] = real_time
        feature[i][2] = len(x["name"]) if x["name"] is not None else 0
        i+=1
# node1 = pandas.read_json(os.path.join(path1, 'node.json'))
edge1 = pandas.read_csv(os.path.join(path1, 'edge.csv'))

# {'retweeted', 'pinned', 'own', 'mentioned', 'followed', 'contain', 'discuss', 'quoted', 'following', 'membership', 'replied_to', 'followers', 'post', 'like'}
edge_index_train1 = []
edge_index_train2 = []
edge_index_test1 = []
edge_index_test2 = []
edge_index_val1 = []
edge_index_val2 = []
with open(os.path.join(path1, 'edge.csv'), 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=",", quotechar="|")
    index = 0
    for row in tqdm(spamreader):
        if row[2][0] == "u" and row[0][0] == "u" :
          if row[0] in node_id_index_dict.keys() and row[2] in node_id_index_dict.keys():
            edge_index_1 = node_id_index_dict[row[0]]
            edge_index_2 = node_id_index_dict[row[2]]
            edge_index1.append(edge_index_1)
            edge_index2.append(edge_index_2)


edge_index = np.vstack([np.array(edge_index1), np.array(edge_index2)]).T






X=pandas.DataFrame(feature)
X.to_csv('twi22X_matrix.csv', index=False)
np.savetxt('twi22edge_index.csv', edge_index, delimiter = ',')

