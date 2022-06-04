import pandas as pd
import numpy as np
import math
import json
from sklearn.metrics import roc_auc_score as auc
from numpy import histogram_bin_edges as bin
from sklearn.ensemble import RandomForestClassifier
import torch
from tqdm import tqdm
from scipy.stats import entropy
import argparse

from datetime import datetime as dt
t0 = dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
def handle_data(df):
    for i in range(len(df)):
        #print(i)
        if  'protected' not in df[i].keys() or df[i]['protected'] == 'False' :
            df[i]['protected'] = 0
        else:
            df[i]['protected'] = 1 
        
        if 'verified' not in df[i].keys() or df[i]['verified'] == 'False' :
            df[i]['verified'] = 0
        else:
            df[i]['verified'] = 1
            if type(df[i]['id']) == type('str'):
                df[i]['id'] = int(df[i]['id'][1:])
        if 'name' not in df[i].keys() or df[i]['name'] is None:
            df[i]['name'] = 0   
        if type(df[i]['name']) == type('str'):
            df[i]['name'] = len(df[i]['name'])
        else:
            df[i]['name'] = 0
        if 'description' not in df[i].keys() or df[i]['description'] is None:
            df[i]['description'] = 0
        else: 
            df[i]['description'] = len(df[i]['description'])
        if 'created_at' not in df[i].keys() or df[i]['created_at'] is None:
            df[i]['created_at'] = 0
        else:
            if args.datasets == 'Twibot-22':
                df[i]['created_at'] = (t0-dt.strptime(data[i]['created_at'],'%Y-%m-%d %X%z')).days
            elif args.datasets == 'cresci-2015':
                df[i]['created_at'] = (t0-dt.strptime(data[i]['created_at'],'%a %b %d %X %z %Y')).days
            else:
                df[i]['created_at'] = (t0-dt.strptime(data[i]['created_at'],'%a %b %d %X %z %Y ')).days
                
        if 'url' not in df[i].keys() or df[i]['url'] is None or df[i]['url'] == '':
            df[i]['url'] = 0
        else:
            df[i]['url'] = 1
        if 'location' not in df[i].keys() or df[i]['location'] is None or df[i]['location'] == '':
            df[i]['location'] = 0
        else:
            df[i]['location'] = 1   

    return df

parser = argparse.ArgumentParser(description="Reproduction of Kudugunta et al. with SMOTENN and rain forest")
parser.add_argument("--datasets", type=str, default="Twibot-22", help="dataset name")
args = parser.parse_args()
dir = "../../datasets/"

print("loading data...")
labels = pd.read_csv(dir+args.datasets+"/label.csv")
user_num = len(labels)
y = np.zeros(user_num)
dict={}

if args.datasets == "Twibot-22":
    data = json.load(open(dir+args.datasets+"/user.json"))
else:
    data = json.load(open(dir+args.datasets+"/node.json"))

split= pd.read_csv(dir+args.datasets+'/split.csv')
for i in range(user_num):
    dict[split['id'][i]] = i


for i in range(user_num):
    if split['split'][i] == "valid" or split['split'][i] == 'val':
        j=i
        break
for i in range(j,user_num):
    if split['split'][i] == "test":
        k=i
        break
for i in range(user_num):
    if labels['label'][i]=="bot":
        y[dict[labels['id'][i]]] = 1
train_mask=range(j)
val_mask=range(j,k)
test_mask=range(k,user_num)

print("begin to process data...")
data = handle_data(data)

feature_num = 11
tot = user_num
print(len(data))
feature = ['followers_count','following_count','tweet_count','listed_count',
           'id','name', 'description','created_at','url','verified','location']

X = np.zeros((tot,feature_num))
for i in tqdm(range(tot)):
    s ='u'+str(data[i]['id'])
    if s not in dict.keys():
        s = data[i]['id']
    for j in range(4):         
        X[dict[s]][j] = data[i]['public_metrics'][feature[j]]
    for j in range(4,feature_num):
        X[dict[s]][j]= data[i][feature[j]]

print("begin to load edge...")
edge = pd.read_csv(dir+args.datasets+'/edge.csv')
edge_fr = edge.loc[edge['relation']=="friend"]

friend = np.zeros((user_num,1000))
count = np.zeros(user_num,dtype=np.int)
for i in tqdm(range(len(edge_fr))):
    if edge_fr['source_id'].iloc[i] in dict.keys() and edge_fr['target_id'].iloc[i] in dict.keys():
        #print(edge['source_id'][i])
        friend[dict[edge_fr['source_id'].iloc[i]]][count[dict[edge_fr['source_id'].iloc[i]]]] = dict[edge_fr['target_id'].iloc[i]]
        count[dict[edge_fr['source_id'].iloc[i]]] +=1
        
b = bin(X[:,10],bins='fd')
l = np.zeros(11,dtype=np.int)
for i in range(11):
    l[i] = len(bin(X[:,i],bins='fd'))

b = []
for i in range(11):
    b.append(bin(X[:,i],bins='fd'))
p = []
for i in range(11):
    p.append(np.zeros(l[i]))
for i in tqdm(range(11)):
    for j in range(user_num):
        pos = np.searchsorted(b[i],X[j][i],side='right')
        p[i][pos-1] += 1     
        

for i in range(11):
    sum = p[i].sum()
    for j in range(l[i]):
        p[i][j] /= sum

friend = friend.astype(np.int)

print("begin to calculate friend preference...")
fp = X.copy()
for i in tqdm(range(user_num)):
    for j in range(11):

        op = np.zeros(l[j])
        fp[i][j]=0
        for k in range(count[i]):
            pos = np.searchsorted(b[j],X[friend[i][k]][j],side='right')
            op[pos-1] += 1
        sum = op.sum()

        if sum != 0:   
            for k in range(l[j]):
                op[k] /= sum
            fp[i][j] = entropy(op,p[j])
 

true_X = np.concatenate((fp,X[:,5].reshape((-1,1)),X[:,6].reshape((-1,1))
                         ,X[:,1].reshape((-1,1)),X[:,0].reshape((-1,1))),axis=1)

np.save('./'+args.datasets+'.npy',true_X)   