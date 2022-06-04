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
import json
from tqdm import tqdm
path = '../../datasets'
dataset1 = 'Twibot-22'
path1 = os.path.join(path, dataset1)
X1 = pd.read_csv('featuretwi22.0.csv', header=None).values
X2 = pd.read_csv('featuretwi22.0.csv', header=None).values
X3 = pd.read_csv('featuretwi22.0.csv', header=None).values
X4 = pd.read_csv('featuretwi22.0.csv', header=None).values
X=X1+X2+X3+X4
with open(os.path.join(path1, 'user.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
label1 = pd.read_csv(os.path.join(path1, 'label.csv'))
clf=tree.DecisionTreeClassifier(max_depth=4,max_leaf_nodes=7)
kf=KFold(n_splits=10,shuffle=True)
count=0
Prec=0
Acc=0
F1=0
Auc=0
Rec=0
i=0
label=np.zeros(X.shape[0])
userid=[]
id_map=dict()
for node in tqdm(node1):
    userid.append(node['id'])
    id_map[node['id']]=i
    i+=1
for index, node in tqdm(label1.iterrows()):
    if node['label'] == 'bot':
      label[id_map[node['id']]]=1
print(set(list(label)))
for X_train_i,X_test_i in kf.split(X):
    clf.fit(X[X_train_i],label[X_train_i])
    res=clf.predict(X[X_test_i])
    print(res.shape)
    print(set(list(res)))
    prob=clf.predict_proba(X[X_test_i])
    Acc+=accuracy_score(label[X_test_i],res)/10.0
    Auc+=roc_auc_score(label[X_test_i], prob[:,1], average='macro')/10.0
    Prec+=precision_score(label[X_test_i],res)/10.0
    F1+=f1_score(label[X_test_i],res)/10.0
    Rec+=recall_score(label[X_test_i],res)/10.0
'''
Acc=np.average(cross_val_score(clf, X, label, scoring='accuracy', cv=10))
Prec=np.average(cross_val_score(clf, X, label, scoring='precision', cv=10))
Rec=np.average(cross_val_score(clf, X, label, scoring='recall', cv=10))
F1=np.average(cross_val_score(clf, X, label, scoring='f1_macro', cv=10))
Auc=np.average(cross_val_score(clf, X, label, scoring='roc_auc', cv=10))
'''
# print(f'Acc:{Acc:.4f}, Prec:{Prec:.4f}, Rec:{Rec:.4f}, F1:{F1:.4f}, AUC:{Auc:.4f}')