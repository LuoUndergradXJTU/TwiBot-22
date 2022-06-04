import argparse
import os
import os.path as osp
import time
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import json
from torch_geometric.nn import ChebConv, GCNConv, Linear
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Choose a dataset.')
args = parser.parse_args()

dataset = args.dataset
path = '../../../datasets/' + dataset
with open(os.path.join(path, 'node.json'), 'r') as f:
    node1 = json.load(f)
label = pd.read_csv(os.path.join(path, 'label.csv'))
split = pd.read_csv(os.path.join(path, 'split.csv'))

num_user = 0
v = 0
age = []
account_length_name = []
id_map = dict()

for node in tqdm(node1):
    if node['id'][0] == 'u':
        account_length_name.append(len(str(node['name'])))
        id_map[node['id']] = num_user
        num_user += 1

if dataset == 'Twibot-20':
    X = pd.read_csv('../X_matrix.csv').values
elif dataset == 'cresci-2015':
    X = pd.read_csv('../X_matrix2.csv').values
else:
    raise ValueError('Dataset' + dataset + 'doesn\'t exist.')
for i in range(X.shape[0]):
    X[i][0] = X[i][0] / 86400.0

num_user = label.shape[0]
label_order = np.array(label['label'].values)
split_order = np.array(split['split'].values)
for i in range(num_user):
    label_order[id_map[label['id'][i]]] = label['label'][i]
    split_order[id_map[split['id'][i]]] = split['split'][i]
y = (label_order == 'bot').astype(int)
Y = torch.LongTensor(y)
train_split = split_order[0: num_user] == 'train'
val_split = split_order[0: num_user] == 'val'
test_split = split_order[0: num_user] == 'test'
train_set = np.where(split_order == 'train')[0]
val_set = np.where(split_order == 'val')[0]
test_set = np.where(split_order == 'test')[0]
print(f"train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)}")

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
Y = Y.to(device)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = Linear(5, 16)
        # self.conv1 = Linear(16, 16)
        # self.conv2 = Linear(16, 16)
        self.lin2 = Linear(16, 2)

    def forward(self):
        x = torch.FloatTensor(X).to(device)
        # x = self.lin1(x)
        # x = F.selu(self.conv1(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        # x = F.selu(self.conv2(x))
        # x = F.dropout(x, p=0.2, training=self.training)
        x = F.selu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)


seed = 500
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
model = Net().to(device)
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=5e-4, lr=0.005)


def train():
    model.train()
    optimizer.zero_grad()
    loss = F.nll_loss(model()[train_set], Y[train_set])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test():
    model.eval()
    accs = []
    pred_train = model()[train_set].max(1)[1].cpu()
    accs.append(accuracy_score(y[train_set], pred_train.numpy()))
    pred_val = model()[val_set].max(1)[1].cpu()
    accs.append(accuracy_score(y[val_set], pred_val.numpy()))
    pred_test = model()[test_set].max(1)[1].cpu()
    prob_test = model()[test_set].cpu()[:, 1].view(-1)
    accs.append(accuracy_score(y[test_set], pred_test.numpy()))
    pre = precision_score(y[test_set], pred_test.numpy())
    recall = recall_score(y[test_set], pred_test.numpy())
    f1 = f1_score(y[test_set], pred_test.numpy())
    auc = roc_auc_score(y[test_set], prob_test.numpy())
    return accs, pre, recall, f1, auc


if not osp.exists('results_rmg_' + dataset):
    os.mkdir('results_rmg_' + dataset)

best_val = 0
best_acc = 0
best_f1 = 0
best_pre = 0
best_recall = 0
best_auc = 0
train_acc_all = []
val_acc_all = []
test_acc_all = []
precision_all = []
recall_all = []
f1_all = []
auc_all = []
for epoch in range(1, 501):
    loss = train()
    accs, pre, recall, f1, auc = test()
    if accs[1] > best_val:
        best_val = accs[1]
        best_acc = accs[2]
        best_f1 = f1
        best_pre = pre
        best_recall = recall
        best_auc = auc
    train_acc_all.append(accs[0])
    val_acc_all.append(accs[1])
    test_acc_all.append(accs[2])
    precision_all.append(pre)
    recall_all.append(recall)
    f1_all.append(f1)
    auc_all.append(auc)
    print(f'Epoch: {epoch:03d}, Loss:{loss:.4f}, Train: {accs[0]:.4f}, Val: {accs[1]:.4f}, Test:{accs[2]:.4f}, Pre:{pre:.4f}, Rec:{recall:.4f}, F1:{f1:.4f}, AUC:{auc:.4f}', flush=True)
print(f"ACC: {best_acc:.4f}, Pre: {best_pre:.4f}, Rec: {best_recall:.4f}, F1: {best_f1:.4f}, AUC: {best_auc:.4f}")
result = pd.DataFrame({'train_acc': train_acc_all, 'val_acc': val_acc_all, 'test_acc': test_acc_all, 'precision': precision_all, 'recall': recall_all, 'f1': f1_all, 'auc': auc_all})
result.to_csv('results_rmg_' + dataset + '/result_4.csv', index=False)

