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
import ijson
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
# {'retweeted', 'pinned', 'own', 'mentioned', 'followed', 'contain', 'discuss', 'quoted', 'following', 'membership', 'replied_to', 'followers', 'post', 'like'}
path = '../../datasets'
dataset1 = 'Twibot-22'

path1 = os.path.join(path, dataset1)

with open(os.path.join(path1, 'user.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)

edge1 = pandas.read_csv(os.path.join(path1, 'edge.csv'))
label1 = pandas.read_csv(os.path.join(path1, 'label.csv'))
split1 = pandas.read_csv(os.path.join(path1, 'split.csv'))

nlp = spacy.load('en_core_web_trf')
i = 0  
v = 0
o = 0
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
        id_map[node['id']] = i
        i=i+1

    
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

tweet_index=0
text_user=dict()
with open("/data2/whr/czl/TwiBot22-baselines/src/twibot22_Botrgcn_feature/id_tweet.json", 'r', encoding='UTF-8') as f:
    idindex_tweet=json.load(f)
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_0.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        Text.append(x['text'])
        text_user[x['text']]=id_map[x['author_id']]
        
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_1.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        Text.append(x['text'])
        text_user[x['text']]=id_map[x['author_id']]
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_2.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        Text.append(x['text'])
        text_user[x['text']]=id_map[x['author_id']]
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_3.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        Text.append(x['text'])
        text_user[x['text']]=id_map[x['author_id']]
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_4.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        text_user[x['text']]=id_map[x['author_id']]
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_5.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        text_user[x['text']]=id_map[x['author_id']]
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_6.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        text_user[x['text']]=id_map[x['author_id']]
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_7.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        text_user[x['text']]=id_map[x['author_id']]
with open("/data2/whr/czl/TwiBot22-baselines/src/Twibot-22/tweet_8.json", 'r', encoding='UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj):
      if x['author_id'] not in support and x['author_id'] in id_map.keys():
        text_user[x['text']]=id_map[x['author_id']]
for text in tqdm(text_user.keys()):
    Text=nlp(text)
    pos = [token.pos_ for token in Text]
    feature[text_user[text],1] += text.count('@')
    feature[text_user[text],2] += text.count('#')
    for s in text:
        if s.isupper():
            feature[text_user[text],4]+=1
    feature[text_user[text],14] += len(pos)
    feature[text_user[text],15]+=SentimentIntensityAnalyzer().polarity_scores(text)['compound']
    for token in Text:
         if token.like_url:
           feature[text_user[text],0] += 1
    for p in pos:
        if p=='PUNCT':
            feature[text_user[text],3] += 1
        elif p=='NOUN':
            feature[text_user[text],5] += 1
        elif p=='PRON':
            feature[text_user[text],6] += 1
        elif p=='VERB':
            feature[text_user[text],7] += 1
        elif p=='ADV':
            feature[text_user[text],8] += 1
        elif p=='ADJ':
            feature[text_user[text],9] += 1
        elif p=='ADP':
            feature[text_user[text],10] += 1
        elif p=='CCONJ' or 'SCONJ':
            feature[text_user[text],11] += 1
        elif p=='NUM':
            feature[text_user[text],12] += 1
        elif p=='INTJ':
            feature[text_user[text],13] += 1
'''
for text in tqdm(text_map.keys()):
    if type(text)==str and text_map[text] in tweet_map.keys():
        # feature[id_map[tweet_map[tweet[count]]]][0] += len(urls)
        Text=nlp(text)
        pos = [token.pos_ for token in Text]
        feature[text_user[text],1] += text.count('@')
        feature[text_user[text],2] += text.count('#')
        for s in text:
            if s.isupper():
                feature[id_map[tweet_map[tweet[count]]]][4]+=1
        feature[text_user[text],14] += len(pos)
        feature[text_user[text],15]+=SentimentIntensityAnalyzer().polarity_scores(text)['compound']
        for token in Text:
             if token.like_url:
               feature[text_user[text],0] += 1
        for p in pos:
            if p=='PUNCT':
                feature[text_user[text],3] += 1
            elif p=='NOUN':
                feature[text_user[text],5] += 1
            elif p=='PRON':
                feature[text_user[text],6] += 1
            elif p=='VERB':
                feature[text_user[text],7] += 1
            elif p=='ADV':
                feature[text_user[text],8] += 1
            elif p=='ADJ':
                feature[text_user[text],9] += 1
            elif p=='ADP':
                feature[text_user[text],10] += 1
            elif p=='CCONJ' or 'SCONJ':
                feature[text_user[text],11] += 1
            elif p=='NUM':
                feature[text_user[text],12] += 1
            elif p=='INTJ':
                feature[text_user[text],13] += 1
    count+=1
'''
np.savetxt('featuretwi22.csv', feature, delimiter = ',')