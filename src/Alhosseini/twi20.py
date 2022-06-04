import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
import pandas as pd
import json
import torch_geometric.transforms as T
from tqdm import tqdm
import random
import spacy
from torch_geometric.data import Data
import scipy.io as sio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def gen_structure_outliers(data1,data2, m, n):

    node_set1 = set(range(data1.x.shape[0]))
    node_set2 = set(range(data2.x.shape[0]))
    new_edges = []
    sample_indices1 = []
    sample_indices2 = []
    for i in range(0, n):

        sampled_idx1 = np.random.choice(list(node_set1), size=m, replace=False)

        sample_indices1 += sampled_idx1.tolist()
        sampled_idx2 = np.random.choice(list(node_set2), size=m, replace=False)

        sample_indices2 += sampled_idx2.tolist()

        for j, id1 in enumerate(sampled_idx1):
            for k, id2 in enumerate(sampled_idx2+data1.x.shape[0]):
                if j != k:
                    new_edges.append(
                        torch.tensor([[id1, id2]], dtype=torch.long))

        node_set1 = node_set1.difference(set(sampled_idx1))
        node_set2 = node_set2.difference(set(sampled_idx2))
    y_outlier1 = torch.zeros(data1.x.shape[0], dtype=torch.long)
    y_outlier1[sample_indices1] = 1
    y_outlier2 = torch.zeros(data2.x.shape[0], dtype=torch.long)
    y_outlier2[sample_indices2] = 1
    y_outlier=torch.hstack([y_outlier1,y_outlier2])
    data1.edge_index = torch.cat([data1.edge_index, torch.cat(new_edges).T],
                                dim=1)

    return data1, y_outlier
path = '../../datasets'
dataset1 = 'Twibot-20'
path1 = os.path.join(path, dataset1)
with open(os.path.join(path1, 'node.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
# node1 = pandas.read_json(os.path.join(path1, 'node.json'))
edge1 = pd.read_csv(os.path.join(path1, 'edge.csv'))
label1 = pd.read_csv(os.path.join(path1, 'label.csv'))
split1 = pd.read_csv(os.path.join(path1, 'split.csv'))
source_node_index = []
target_node_index = []
nlp = spacy.load('en_core_web_trf')
i = 0  
v = 0
o = 0
t = 0
tweet = []
id_map = dict()
tweet_map = dict()
text_map = dict()
tweet_index=dict()
support=[]
usernum=0
index_new=dict()

for index, node in split1.iterrows():
    if node['split'] == 'support':
        support.append(node['id'])
    else:
        usernum+=1
for node in tqdm(node1):
    if node['id'][0] == 'u' and node['id'] not in support:
        id_map[node['id']] = i
        i=i+1
    if node['id'][0] == 't':
        text_map[node['text']]=node['id']
        tweet_index[node['id']]=t
        t=t+1
chosen=random.sample(range(1,t),20000)
feature = np.zeros((20000, 16))
h=0
for node in tqdm(edge1['source_id']):
    if node in id_map.keys():
      if (edge1['relation'][v] == 'post'):
        if edge1['target_id'][v] in tweet_index.keys() and tweet_index[edge1['target_id'][v]] in chosen:
          index_new[edge1['target_id'][v]]=h
          h+=1
          tweet_map[edge1['target_id'][v]] = node
          source_node_index.append(id_map[node])
          target_node_index.append(h+usernum)
    v += 1
count = 0
print(len(source_node_index))
for text in tqdm(text_map.keys()):
    if type(text)==str and text_map[text] in tweet_map.keys():
        Text=nlp(text)
        pos = [token.pos_ for token in Text]
        feature[index_new[text_map[text]],1] += text.count('@')
        feature[index_new[text_map[text]],2] += text.count('#')
        for s in text:
            if s.isupper():
                feature[index_new[text_map[text]],4]+=1
        feature[index_new[text_map[text]],14] += len(pos)
        feature[index_new[text_map[text]],15]+=SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        for token in Text:
            if token.like_url:
                feature[index_new[text_map[text]],0] += 1
        for p in pos:
            if p=='PUNCT':
                feature[index_new[text_map[text]],3] += 1
            elif p=='NOUN':
                feature[index_new[text_map[text]],5] += 1
            elif p=='PRON':
                feature[index_new[text_map[text]],6] += 1
            elif p=='VERB':
                feature[index_new[text_map[text]],7] += 1
            elif p=='ADV':
                feature[index_new[text_map[text]],8] += 1
            elif p=='ADJ':
                feature[index_new[text_map[text]],9] += 1
            elif p=='ADP':
                feature[index_new[text_map[text]],10] += 1
            elif p=='CCONJ' or 'SCONJ':
                feature[index_new[text_map[text]],11] += 1
            elif p=='NUM':
                feature[index_new[text_map[text]],12] += 1
            elif p=='INTJ':
                feature[index_new[text_map[text]],13] += 1
    count+=1
edge_index = np.vstack([np.array(source_node_index), np.array(target_node_index)])
np.savetxt('X_tweet.csv', feature, delimiter = ',')
np.savetxt('edge_index.csv', edge_index, delimiter = ',')
path='../../data'
x1=np.array(torch.load(os.path.join(path,'cat_properties_tensor.pt')))[:usernum,:]
x2=np.array(torch.load(os.path.join(path,'des_tensor.pt')))[:usernum,:]
x3=np.array(torch.load(os.path.join(path,'num_properties_tensor.pt')))[:usernum,:]
x4=np.array(torch.load(os.path.join(path,'tweets_tensor.pt')))[:usernum,:]
X_user=np.hstack([x1,x2,x3,x4])
T=np.zeros((feature.shape[0],X_user.shape[1]-feature.shape[1]))
X_tweet=np.hstack([feature,T])
X=np.vstack([X_user,X_tweet])

# data, ya = gen_attribute_outliers(data1, n=500, k=50)
# data, ys = gen_structure_outliers(data, m=25, n=20)
# data.y = torch.logical_or(ys, ya).int()
label=np.zeros((X.shape[0]))
for index, node in label1.iterrows():
  if node['id'] in id_map.keys() :
    if node['label'] == 'bot':
      label[id_map[node['id']]]=1
data1=Data(x=torch.FloatTensor(X_user),edge_index=torch.LongTensor(edge_index))
data2=Data(x=torch.FloatTensor(X_tweet),edge_index=torch.LongTensor(edge_index))
data1, ys = gen_structure_outliers(data1, data2, m=10, n=5)
y = torch.logical_or(ys, torch.IntTensor(label)).int()
data=Data(x=torch.FloatTensor(X),edge_index=torch.LongTensor(data1.edge_index),y=y)
sio.savemat('twi20.mat', {'X1':X_user,'X2':X_tweet,'edge_index':np.array(data1.edge_index),'label':np.array(y)})