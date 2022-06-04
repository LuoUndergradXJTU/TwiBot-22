import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import os
import os.path as osp
from tqdm import tqdm
import json
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
X = pd.read_csv('featurebf2019.csv', header=None).values
with open(os.path.join(path8, 'node.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
label1 = pd.read_csv(os.path.join(path8, 'label.csv'))
split1 = pd.read_csv(os.path.join(path8, 'split.csv'))

print(X.shape)

count=0

i=0
label=np.zeros(X.shape[0])
userid=[]
id_map=dict()
for node in node1:
    userid.append(node['id'])
    id_map[node['id']]=i
    i+=1
for index, node in label1.iterrows():
  if node['id'] in userid:
    if node['label'] == 'bot':
      label[id_map[node['id']]]=1
train_id = []
test_id = []
val_id = []
for index, node in tqdm(split1.iterrows()):
    if node['split'] == 'train':
        # train_id.append(userid.index(node['id']))
        train_id.append(id_map[node['id']])
    if node['split'] == 'test':
        # test_id.append(userid.index(node['id']))
        test_id.append(id_map[node['id']])
    if node['split'] == 'val':
        # val_id.append(userid.index(node['id']))
        val_id.append(id_map[node['id']])

clf=tree.DecisionTreeClassifier(max_depth=4,max_leaf_nodes=7)
clf=clf.fit(X[train_id],label[train_id])
res=clf.predict(X[test_id])
prob=clf.predict_proba(X[test_id])
Acc=accuracy_score(label[test_id],res)
Auc=roc_auc_score(label[test_id], prob[:,1], average='macro')
Prec=precision_score(label[test_id],res)
F1=f1_score(label[test_id], res)
Rec=recall_score(label[test_id],res)
print(f'Acc:{Acc:.4f}, Prec:{Prec:.4f}, Rec:{Rec:.4f}, F1:{F1:.4f}, AUC:{Auc:.4f}')
'''
Acc=np.average(cross_val_score(clf, X, label, scoring='accuracy', cv=10))
Prec=np.average(cross_val_score(clf, X, label, scoring='precision', cv=10))
Rec=np.average(cross_val_score(clf, X, label, scoring='recall', cv=10))
F1=np.average(cross_val_score(clf, X, label, scoring='f1_macro', cv=10))
Auc=np.average(cross_val_score(clf, X, label, scoring='roc_auc', cv=10))
'''
