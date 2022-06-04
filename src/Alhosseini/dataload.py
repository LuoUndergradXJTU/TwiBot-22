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
with open(os.path.join(path1, 'node.json'), 'r', encoding = 'UTF-8') as f:
    node1 = json.load(f)
# node1 = pandas.read_json(os.path.join(path1, 'node.json'))
edge1 = pandas.read_csv(os.path.join(path1, 'edge.csv'))
label1 = pandas.read_csv(os.path.join(path1, 'label.csv'))
split1 = pandas.read_csv(os.path.join(path1, 'split.csv'))
# time_now=time.strftime(time.localtime(time.time()))
'''def age_calculate(time1,now):
    if(int(now[5:6])<int(time1[5:6])):
        if(int(now[8:9])<int(time1[8:9])):
            days = (int(now[0:3]) - int(time1[0:3])) * 365-360+(int(now[5:6])-int(time1[5:6])+12)*30-30+(int(now[8:9])-int(time1[8:9])+30)
        else:
            days = (int(now[0:3]) - int(time1[0:3])) * 365-360+(int(now[5:6])-int(time1[5:6])+12)*30+(int(now[8:9])-int(time1[8:9]))
    else:
        if (int(now[8:9]) < int(time1[8:9])):
            days = (int(now[0:3]) - int(time1[0:3])) * 365  + (int(now[5:6]) - int(time1[5:6]) ) * 30 - 30 + (
                        int(now[8:9]) - int(time1[8:9]) + 30)
        else:
            days = (int(now[0:3]) - int(time1[0:3])) * 365  + (int(now[5:6]) - int(time1[5:6]) ) * 30 + (
                        int(now[8:9]) - int(time1[8:9]))

    return days
'''

source_node_index = []
target_node_index = []
i = 0  # user点的数量
v = 0
age = []
account_length_name = []
userid=[]

for node in tqdm(node1):
    if (node['id'][0] == 'u'):
        # age.append(age_calculate(node.created_at,time_now))
        account_length_name.append(len(str(node['name'])))
        userid.append(str(node['id']))
        i = i+ 1

statuses_count = np.zeros(i) # 发的推文数
followers_count = np.zeros(i)
friends_count = np.zeros(i)


for node in tqdm(edge1['source_id']):
    if (edge1['relation'][v] == 'post'):
        statuses_count[userid.index(node)]+=1
        v += 1
        continue
    if (edge1['relation'][v] == 'follow'):
        followers_count[userid.index(str(edge1.target_id[v]))]+=1
        source_node_index.append(userid.index(node))
        target_node_index.append(userid.index(edge1.target_id[v]))
        v = v+ 1
    if (edge1['relation'][v] == 'friend'):
        friends_count[userid.index(node)] += 1
        source_node_index.append(userid.index(node))
        target_node_index.append(userid.index(edge1.target_id[v]))
        v = v+ 1

X_matrix=np.hstack([statuses_count.T,np.array(account_length_name).T,followers_count.T,friends_count.T])
edge_index = torch.LongTensor(np.vstack(([np.array(source_node_index), np.array(target_node_index)])))
A_matrix = torch.sparse.FloatTensor(edge_index, torch.LongTensor(np.ones(i)), torch.Size([i, i])).to_dense()
np.savetxt(fname="./X_matrix.csv", X=X_matrix)
np.savetxt(fname="./A_matrix.csv", X=np.array(A_matrix))
np.savetxt(fname="./edge_index.csv", X=np.array(edge_index))
