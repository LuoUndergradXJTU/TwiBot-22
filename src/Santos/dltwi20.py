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
dataset8 = 'cresci-feedback-2019'
path1 = os.path.join(path, dataset1)
path2 = os.path.join(path, dataset2)
path3 = os.path.join(path, dataset3)
path4 = os.path.join(path, dataset4)
path5 = os.path.join(path, dataset5)
path6 = os.path.join(path, dataset5)
path7 = os.path.join(path, dataset7)
path8 = os.path.join(path, dataset8)
with open(os.path.join(path1, 'node.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
# node1 = pandas.read_json(os.path.join(path1, 'node.json'))
edge1 = pandas.read_csv(os.path.join(path1, 'edge.csv'))
label1 = pandas.read_csv(os.path.join(path1, 'label.csv'))
split1 = pandas.read_csv(os.path.join(path1, 'split.csv'))

nlp = spacy.load('en_core_web_trf')
i = 0  
v = 0
o = 0
userid = []
tweet = []
id_map = dict()
tweet_map = dict()
text_map = dict()
support=[]

for index, node in split1.iterrows():
    if node['split'] == 'support':
        support.append(node['id'])
for node in tqdm(node1):
    if node['id'][0] == 'u' and node['id'] not in support:
        userid.append(str(node['id']))
        id_map[node['id']] = i
        i=i+1
    if node['id'][0] == 't':
        text_map[node['text']]=node['id']
    
feature = np.zeros((i, 16))
for node in tqdm(node1):
   if node['id'][0] == 'u' and type(node['description'])==str:
     doc=nlp(node['description'])
     pos = [token.pos_ for token in doc]
     feature[o,1] += node['description'].count('@')
     feature[o,2] += node['description'].count('#')
     for s in node['description']:
         if s.isupper():
             feature[o,4]+=1
     feature[o,14] += len(pos)
     feature[o,15]+=SentimentIntensityAnalyzer().polarity_scores(node['description'])['compound']
     for token in doc:
         if token.like_url:
             feature[o,0] += 1
     for p in pos:
         if p=='PUNCT':
             feature[o,3] += 1
         elif p=='NOUN':
             feature[o,5] += 1
         elif p=='PRON':
             feature[o,6] += 1
         elif p=='VERB':
             feature[o,7] += 1
         elif p=='ADV':
             feature[o,8] += 1
         elif p=='ADJ':
             feature[o,9] += 1
         elif p=='ADP':
             feature[o,10] += 1
         elif p=='CCONJ' or 'SCONJ':
             feature[o,11] += 1
         elif p=='NUM':
             feature[o,12] += 1
         elif p=='INTJ':
             feature[o,13] += 1
   o=o+1
   if o==i:
     break

for node in tqdm(edge1['source_id']):
    if node in id_map.keys():
      if (edge1['relation'][v] == 'post'):
        if edge1['target_id'][v] in tweet_map.keys():
          tweet_map[edge1['target_id'][v]] = node
    v += 1
count = 0

for text in tqdm(text_map.keys()):
    if type(text)==str and text_map[text] in tweet_map.keys():
        # feature[id_map[tweet_map[tweet[count]]]][0] += len(urls)
        Text=nlp(text)
        pos = [token.pos_ for token in Text]
        feature[id_map[tweet_map[text_map[text]]],1] += text.count('@')
        feature[id_map[tweet_map[text_map[text]]],2] += text.count('#')
        for s in text:
            if s.isupper():
                feature[id_map[tweet_map[tweet[count]]]][4]+=1
        feature[id_map[tweet_map[text_map[text]]],14] += len(pos)
        feature[id_map[tweet_map[text_map[text]]],15]+=SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        for token in Text:
             if token.like_url:
               feature[id_map[tweet_map[text_map[text]]],0] += 1
        for p in pos:
            if p=='PUNCT':
                feature[id_map[tweet_map[text_map[text]]],3] += 1
            elif p=='NOUN':
                feature[id_map[tweet_map[text_map[text]]],5] += 1
            elif p=='PRON':
                feature[id_map[tweet_map[text_map[text]]],6] += 1
            elif p=='VERB':
                feature[id_map[tweet_map[text_map[text]]],7] += 1
            elif p=='ADV':
                feature[id_map[tweet_map[text_map[text]]],8] += 1
            elif p=='ADJ':
                feature[id_map[tweet_map[text_map[text]]],9] += 1
            elif p=='ADP':
                feature[id_map[tweet_map[text_map[text]]],10] += 1
            elif p=='CCONJ' or 'SCONJ':
                feature[id_map[tweet_map[text_map[text]]],11] += 1
            elif p=='NUM':
                feature[id_map[tweet_map[text_map[text]]],12] += 1
            elif p=='INTJ':
                feature[id_map[tweet_map[text_map[text]]],13] += 1
    count+=1
np.savetxt('featuretwi20.csv', feature, delimiter = ',')