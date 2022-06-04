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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

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
with open(os.path.join(path3, 'node.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
# node1 = pandas.read_json(os.path.join(path1, 'node.json'))
edge1 = pandas.read_csv(os.path.join(path3, 'edge.csv'))
label1 = pandas.read_csv(os.path.join(path3, 'label.csv'))
split1 = pandas.read_csv(os.path.join(path3, 'split.csv'))
# time_now=time.strftime(time.localtime(time.time()))

source_node_index = []
target_node_index = []
i = 0  
v = 0
age = []
account_length_name = []
userid = []

id_map = dict()

for node in tqdm(node1):
    if (node['id'][0] == 'u'):
        # age.append(age_calculate(node.created_at,time_now))
        account_length_name.append(len(str(node['name'])))
        userid.append(str(node['id']))
        id_map[node['id']] = i
        i = i+ 1

X = pandas.read_csv('X_matrix3.csv').values
edge_Index = pandas.read_csv('edge_index3.csv').values
print(X.shape)
print(edge_Index.shape)
label = []

for index, node in label1.iterrows():
  if node['id'] in userid:
    if node['label'] == 'bot':
        label.append(1)

    if node['label'] == 'human':
        label.append(0)


Label = np.array(label)
train_id = []
test_id = []
val_id = []
for index, node in split1.iterrows():
  if node['id'] in userid:
    if node['split'] == 'train':
        # train_id.append(userid.index(node['id']))
        train_id.append(id_map[node['id']])
    if node['split'] == 'test':
        # test_id.append(userid.index(node['id']))
        test_id.append(id_map[node['id']])
    if node['split'] == 'val':
        # val_id.append(userid.index(node['id']))
        val_id.append(id_map[node['id']])


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 16, cached=True,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(16, 4, cached=True,
                             normalize=not args.use_gdc)
        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self):
        x, edge_index = torch.FloatTensor(X), torch.LongTensor(edge_Index)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(torch.device('cpu'))
optimizer = torch.optim.Adam([
    dict(params=model.conv1.parameters(), weight_decay=5e-4),
    dict(params=model.conv2.parameters(), weight_decay=0)
], lr=0.01)  # Only perform weight-decay on first convolution.

# train_set = model.forward()[train_id]
train_label = Label[train_id]
# val_set = model.forward()[val_id]
val_label = Label[val_id]
# test_set = model.forward()[test_id]
test_label = Label[test_id]

def train():
    model.train()
    optimizer.zero_grad()
    loss = F.cross_entropy(model()[train_id], torch.LongTensor(train_label))
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    accs = []
    precs=[]
    recs=[]
    f1s=[]
    pred_train = model()[train_id].max(1)[1]
    acc_train = f1_score(train_label, pred_train, average='macro')
    accs.append(acc_train)
    pred_val = model()[val_id].max(1)[1]
    acc_val = f1_score(val_label, pred_val, average='macro')
    accs.append(acc_val)
    pred_test = model()[test_id].max(1)[1]
    acc_test = f1_score(test_label, pred_test, average='macro')
    prec, rec, f1, _ = precision_recall_fscore_support(test_label,pred_test,average='micro')
    accs.append(acc_test)
    auc=roc_auc_score(test_label, pred_test, average='micro')

    return accs, prec, rec, f1, auc


best_val_acc = test_acc = 0
for epoch in range(1, 201):
    loss=train()
    train_acc, val_acc, tmp_test_acc = test()[0]
    test_prec, test_rec, test_f1, auc=test()[1], test()[2],test()[3],test()[4]
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    print(f'Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {best_val_acc:.4f}, Test:{test_acc:.4f}, Prec:{test_prec:.4f}, Rec:{test_rec:.4f}, F1:{test_f1:.4f}, AUC:{auc:.4f}')