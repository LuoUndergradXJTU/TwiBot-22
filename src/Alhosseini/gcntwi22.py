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
from torch_geometric.nn import ChebConv, GCNConv, Linear  # noqa
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import preprocessing
import random
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()
path = '../../datasets'
dataset1 = 'Twibot-22'
path1 = os.path.join(path, dataset1)
with open(os.path.join(path1, 'user.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
# edge1 = pandas.read_csv(os.path.join(path1, 'edge.csv'))
label1 = pandas.read_csv(os.path.join(path1, 'label.csv'))
split1 = pandas.read_csv(os.path.join(path1, 'split.csv'))

source_node_index = []
target_node_index = []
i = 0  
v = 0

userid = []

id_map = dict()
X = pandas.read_csv('twi22X_matrix.csv').values
edge_Index=pandas.read_csv('twi22edge_index.csv').values.T

print(label1.shape)
print(split1.shape)
print(X.shape)
for node in tqdm(node1):
    if node['id'][0] == 'u' and node['id'] not in id_map.keys():
        userid.append(str(node['id']))
        id_map[node['id']] = i
        i = i+ 1
print(i)
label = []
for i in tqdm(range(X.shape[0])):
    X[i,0]=(X[i,0]/86400.0) + 577.0
# X = preprocessing.minmax_scale(X, axis=0)

for index, node in tqdm(label1.iterrows()):
    if node['label'] == 'bot':
        label.append(1)

    if node['label'] == 'human':
        label.append(0)


Label = np.array(label)
print(Label.shape)
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

print(len(train_id))
print(len(test_id))
print(len(val_id))
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = GCNConv(5, 16, cached=True,
                             # normalize=not args.use_gdc)
        # self.conv2 = GCNConv(16, 16, cached=True,
                             # normalize=not args.use_gdc)
        self.lin = Linear(5,2)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = torch.FloatTensor(X), torch.LongTensor(edge_Index)
        # x = self.conv1(x, edge_index)
        # x = F.dropout(x, training=self.training)
        # x = F.relu(self.conv2(x, edge_index))
        x = self.lin(x)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(torch.device('cpu'))
optimizer = torch.optim.Adam([
    # dict(params=model.conv1.parameters(), weight_decay=5e-4),
    # dict(params=model.conv2.parameters(), weight_decay=0),
    dict(params=model.lin.parameters(), weight_decay=0)
], lr=0.005)  # Only perform weight-decay on first convolution.

# train_set = model.forward()[train_id]
train_label = Label[train_id]
# val_set = model.forward()[val_id]
val_label = Label[val_id]
# test_set = model.forward()[test_id]
test_label = Label[test_id]
human=[]
bot=[]
for i in train_id:
    if Label[i]==0:
        human.append(i)
    if Label[i]==1:
        bot.append(i)
sample=random.sample(human,len(bot))
Train=sample+bot
Train_label=Label[Train]
print(len(Train))
print(Train_label.shape)
def train():
    model.train()
    optimizer.zero_grad()
    prob_train=model()[train_id]
    pred_train = prob_train.max(1)[1]
    acc_train = accuracy_score(train_label, pred_train)
    loss = F.nll_loss(model()[Train], torch.LongTensor(Train_label))
    loss.backward()
    optimizer.step()
    return float(loss), acc_train

@torch.no_grad()
def test():
    model.eval()
    accs = []
    precs=[]
    recs=[]
    f1s=[]
    pred_train = model()[train_id].max(1)[1]
    acc_train = accuracy_score(train_label, pred_train)
    accs.append(acc_train)
    pred_val = model()[val_id].max(1)[1]
    acc_val = accuracy_score(val_label, pred_val)
    accs.append(acc_val)
    pred_test = model()[test_id].max(1)[1]
    prob_test = model()[test_id][:,1].T
    acc_test = accuracy_score(test_label, pred_test)
    prec=precision_score(test_label, pred_test, average='macro')
    rec=recall_score(test_label, pred_test, average='macro')
    f1=f1_score(test_label, pred_test, average='macro')
    accs.append(acc_test)
    auc=roc_auc_score(test_label, prob_test, average='macro')

    return accs, prec, rec, f1, auc

for i in range(5):
    best_val=0
    best_test=0
    best_f1=0
    Prec=0
    Rec=0
    for epoch in range(1, 101):
        loss=train()
        train_acc, val_acc, test_acc = test()[0]
        test_prec, test_rec, test_f1, auc=test()[1], test()[2],test()[3],test()[4]
        if val_acc>best_val:
          best_val=val_acc
          best_test=test_acc
          best_f1=test_f1
          Prec=test_prec
          Rec=test_rec
        # print(f'Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, '
              # f'Val: {val_acc:.4f}, Test:{test_acc:.4f}, Prec:{test_prec:.4f}, Rec:{test_rec:.4f}, F1:{test_f1:.4f}, AUC:{auc:.4f}')
    print(best_test, Prec, Rec, best_f1)