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
import time
import ijson
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()


path = '../../datasets'
dataset1 = 'Twibot-20'
dataset2 = 'cresci-2015'
dataset3 = 'cresci-2017'
path2 = os.path.join(path, dataset2)
path3 = os.path.join(path, dataset3)
path1 = os.path.join(path, dataset1)
with open(os.path.join(path2, 'node.json'), 'r', encoding = 'UTF-8') as f:
    node1 = json.load(f)
# node1 = pandas.read_json(os.path.join(path1, 'node.json'))
edge1 = pandas.read_csv(os.path.join(path2, 'edge.csv'))
label1 = pandas.read_csv(os.path.join(path2, 'label.csv'))
split1 = pandas.read_csv(os.path.join(path2, 'split.csv'))


source_node_index = []
target_node_index = []
i = 0  
v = 0
account_length_name = []
userid=[]
id_map = dict()

for node in tqdm(node1):
    if (node['id'][0] == 'u'):
        # age.append(age_calculate(node.created_at,time_now))
        account_length_name.append(len(str(node['name'])))
        userid.append(str(node['id']))
        id_map[node['id']] = i
        i = i+ 1


statuses_count = np.zeros(i) 
followers_count = np.zeros(i)
friends_count = np.zeros(i)
age=np.zeros(i)
stop_time = 1601510400.0
with open(os.path.join(path1, 'node.json'), 'r', encoding = 'UTF-8') as f:
    j = 0
    obj = ijson.items(f, "item")
    for x in obj:
       time_seen = x["created_at"]
       if time_seen is not None:
           real_time = stop_time - time.mktime(time.strptime(time_seen, "%a %b %d %H:%M:%S +0000 %Y "))
       else:
         real_time = 0
       age[j] = real_time
       j+=1
       if j == i:
           break
# edge2 = edge1.values[edge1['relation'].values != 'post']
for node in tqdm(edge1['source_id']):
    if node in id_map.keys():
        if (edge1['relation'][v] == 'post'):
            # statuses_count[userid.index(node)]+=1
            statuses_count[id_map[node]] += 1
            v += 1
        else:
          if(edge1['target_id'][v] in id_map.keys()):
            if (edge1['relation'][v] == 'follow'):
                # followers_count[userid.index(str(edge1.target_id[v]))]+=1
                followers_count[id_map[edge1['target_id'][v]]] += 1
                # source_node_index.append(userid.index(node))
                source_node_index.append(id_map[node])
                # target_node_index.append(userid.index(edge1.target_id[v]))
                target_node_index.append(id_map[edge1['target_id'][v]])
                v = v + 1
            if (edge1['relation'][v] == 'friend'):
                # friends_count[node1['id'].index(node)] += 1
                friends_count[id_map[node]] += 1
                # source_node_index.append(userid.index(node))
                source_node_index.append(id_map[node])
                # target_node_index.append(userid.index(edge1.target_id[v]))
                target_node_index.append(id_map[edge1['target_id'][v]])
                v = v + 1
          else:
            v=v+1
    else:
        v = v + 1
        continue

X_matrix=np.vstack([age, statuses_count,np.array(account_length_name),followers_count,friends_count]).T
edge_index = np.vstack(([np.array(source_node_index), np.array(target_node_index)]))
X=pandas.DataFrame(X_matrix)
X.to_csv('X_matrix2.csv', index=False)
edge_Index=pandas.DataFrame(edge_index)
edge_Index.to_csv('edge_index2.csv', index=False)