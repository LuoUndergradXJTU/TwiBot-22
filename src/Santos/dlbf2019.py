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
import spacy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


path = '../../datasets'
dataset1 = 'Twibot-20'
dataset2 = 'cresci-2015'
dataset3 = 'cresci-2017'
dataset4 = 'midterm-2018'
dataset5 = 'gilani-2017'
dataset6 = 'cresci-stock-2018'
dataset7 = 'cresci-rtbust-2019'
dataset8 = 'botometer-feedback-2019'
path1 = os.path.join(path, dataset1)
path2 = os.path.join(path, dataset2)
path3 = os.path.join(path, dataset3)
path4 = os.path.join(path, dataset4)
path5 = os.path.join(path, dataset5)
path6 = os.path.join(path, dataset6)
path7 = os.path.join(path, dataset7)
path8 = os.path.join(path, dataset8)
with open(os.path.join(path8, 'node.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
# node1 = pandas.read_json(os.path.join(path1, 'node.json'))
label1 = pd.read_csv(os.path.join(path8, 'label.csv'))
split1 = pd.read_csv(os.path.join(path8, 'split.csv'))
nlp = spacy.load('en_core_web_trf')
print(len(node1))
feature = np.zeros((len(node1), 16))
i = 0  
v = 0
userid = []
tweet = []
id_map = dict()
tweet_map = dict()
text_map = dict()
for node in tqdm(node1):
    if node['id'][0] == 'u' and type(node['description'])==str:
        userid.append(str(node['id']))
        id_map[node['id']] = i
        doc=nlp(node['description'])
        pos = [token.pos_ for token in doc]
        feature[i,1] += node['description'].count('@')
        feature[i,2] += node['description'].count('#')
        for s in node['description']:
            if s.isupper():
                feature[i,4]+=1
        feature[i,14] += len(pos)
        feature[i,15]+=SentimentIntensityAnalyzer().polarity_scores(node['description'])['compound']
        for token in doc:
            if token.like_url:
                feature[i,0] += 1
        for p in pos:
            if p=='PUNCT':
                feature[i,3] += 1
            elif p=='NOUN':
                feature[i,5] += 1
            elif p=='PRON':
                feature[i,6] += 1
            elif p=='VERB':
                feature[i,7] += 1
            elif p=='ADV':
                feature[i,8] += 1
            elif p=='ADJ':
                feature[i,9] += 1
            elif p=='ADP':
                feature[i,10] += 1
            elif p=='CCONJ' or 'SCONJ':
                feature[i,11] += 1
            elif p=='NUM':
                feature[i,12] += 1
            elif p=='INTJ':
                feature[i,13] += 1
    i=i+1
# feature_save = pd.DataFrame(feature)
# feature_save.to_csv('featurem2018.csv')
np.savetxt('featurecf2019.csv', feature, delimiter = ',')