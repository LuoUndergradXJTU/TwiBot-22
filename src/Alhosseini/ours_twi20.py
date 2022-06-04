import os.path as osp
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv
import heapq
import random
from scipy.spatial.distance import pdist
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
import scipy.io as sio
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

data2 = sio.loadmat(f'twi20.mat')

data1 = HeteroData()
data1['user'].x = torch.FloatTensor(data2['X1'])
data1['tweet'].x = torch.FloatTensor(data2['X2'][:,:16])
data1.node_types=['user', 'tweet']
data1.edge_types=[('user', 'to', 'tweet'), ('tweet', 'to', 'user')]
for i in range(data2['edge_index'].shape[1]):
    data2['edge_index'][1, i] -= data2['X1'].shape[0]
data1[('user', 'to', 'tweet')].edge_index = torch.LongTensor(data2['edge_index'])
arr = np.copy(data2['edge_index'])
arr[[0, 1], :] = arr[[1, 0], :]
data1[('tweet', 'to', 'user')].edge_index = torch.LongTensor(arr)
# data1[('user', 'to', 'director')].edge_index = torch.LongTensor(data[('user', 'to', 'director')]['edge_index'])
# data1[('director', 'to', 'user')].edge_index = torch.LongTensor(data[('director', 'to', 'user')]['edge_index'])
node_types=['user', 'tweet']
gnd = data2['label'][0, :]
gnd1 = gnd[0:data2['X1'].shape[0]]
gnd2 = gnd[data2['X1'].shape[0]:]
T=dict()
T['user']=torch.zeros((data1['user'].x.shape[0],2))
T['tweet']=torch.zeros((data1['tweet'].x.shape[0],2))
for i in range(data1['user'].x.shape[0]):
    T['user'][i,0]=1
for i in range(data1['tweet'].x.shape[0]):
    T['tweet'][i,1]=1
X = dict()
for node_type in data1.node_types:
    X[node_type] = data1[node_type].x
A = dict()
flag_user = []
for i in range(data1.x_dict['user'].shape[1]):
    flag_user.append(random.randint(1, 4))
flag_user = np.array(flag_user)
flag_tweet = []
for i in range(data1.x_dict['tweet'].shape[1]):
    flag_tweet.append(random.randint(1, 2))
flag_tweet = np.array(flag_tweet)
def multiview(x, num1, num2):
    X_hat_views = dict()
    X_hat_views['user'] = []
    X_hat_views['tweet'] = []
    index_user = []
    index_tweet = []
    n = 0
    m = 0
    for i in range(num1):
        while n < len(flag_user):
            if flag_user[n] == i + 1:
                index_user.append(n)
            n += 1
        X_hat_views['user'].append(x['user'][:, index_user])
        index_user = []
        n=0
    for i in range(num2):
        while m < len(flag_tweet):
            if flag_tweet[m] == i + 1:
                index_tweet.append(m)
            m += 1
        X_hat_views['tweet'].append(x['tweet'][:, index_tweet])
        index_tweet = []
        m=0
    return X_hat_views
X_views=multiview(X, 4, 2)
y_dict1 = dict.copy(data1.x_dict)
y_dict1['user'] = X_views['user'][0]
y_dict1['tweet'] = X_views['tweet'][0]
y_dict2 = dict.copy(data1.x_dict)
y_dict2['user'] = X_views['user'][1]
y_dict2['tweet'] = X_views['tweet'][0]
y_dict3 = dict.copy(data1.x_dict)
y_dict3['user'] = X_views['user'][2]
y_dict3['tweet'] = X_views['tweet'][0]
y_dict4 = dict.copy(data1.x_dict)
y_dict4['user'] = X_views['user'][3]
y_dict4['tweet'] = X_views['tweet'][0]
y_dict5 = dict.copy(data1.x_dict)
y_dict5['user'] = X_views['user'][0]
y_dict5['tweet'] = X_views['tweet'][1]
y_dict6 = dict.copy(data1.x_dict)
y_dict6['user'] = X_views['user'][1]
y_dict6['tweet'] = X_views['tweet'][1]
y_dict7 = dict.copy(data1.x_dict)
y_dict7['user'] = X_views['user'][2]
y_dict7['tweet'] = X_views['tweet'][1]
y_dict8 = dict.copy(data1.x_dict)
y_dict8['user'] = X_views['user'][3]
y_dict8['tweet'] = X_views['tweet'][1]
list1 = [y_dict1, y_dict2, y_dict3, y_dict4, y_dict5, y_dict6, y_dict7, y_dict8]

# A = torch.sparse.FloatTensor(i, v, torch.Size([4057, 14328])).to_dense()
for edge_type in data1.edge_types:
    A[edge_type] = torch.sparse.FloatTensor(data1.edge_index_dict[edge_type],
                                            torch.LongTensor(np.ones(data1.edge_index_dict[edge_type].shape[1])),
                                            torch.Size([data1[edge_type[0]].x.shape[0],
                                                        data1[edge_type[2]].x.shape[0]])).to_dense()


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, num_view):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_view), requires_grad=True)
        self.weight_node_type = torch.nn.Parameter(torch.randn(2), requires_grad=True)
        # self.w_FC = torch.nn.Parameter(torch.randn(out_channels, data.x_dict['user'].shape[1]), requires_grad=True)
        # self.b_FC = torch.nn.Parameter(torch.randn(data.x_dict['user'].shape), requires_grad=True)
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data1.node_types:
            self.lin_dict[node_type] = torch.nn.ModuleList()
            for i in range(8):
                if i == 0:
                    self.lin_dict[node_type].append(Linear(y_dict1[node_type].shape[1], hidden_channels))
                if i == 1:
                    self.lin_dict[node_type].append(Linear(y_dict2[node_type].shape[1], hidden_channels))
                if i == 2:
                    self.lin_dict[node_type].append(Linear(y_dict3[node_type].shape[1], hidden_channels))
                if i == 3:
                    self.lin_dict[node_type].append(Linear(y_dict4[node_type].shape[1], hidden_channels))
                if i == 4:
                    self.lin_dict[node_type].append(Linear(y_dict5[node_type].shape[1], hidden_channels))
                if i == 5:
                    self.lin_dict[node_type].append(Linear(y_dict6[node_type].shape[1], hidden_channels))
                if i == 6:
                    self.lin_dict[node_type].append(Linear(y_dict7[node_type].shape[1], hidden_channels))
                if i == 7:
                    self.lin_dict[node_type].append(Linear(y_dict8[node_type].shape[1], hidden_channels))

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data1.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        self.out_dict = torch.nn.ModuleDict()
        for node_type in data1.node_types:
            self.out_dict[node_type] = Linear(hidden_channels, out_channels)
        self.lin = torch.nn.ModuleDict()
        for node_type in data1.node_types:
            self.lin[node_type] = Linear(out_channels, data1.x_dict[node_type].shape[1])
        self.Tlin = torch.nn.ModuleDict()
        for node_type in data1.node_types:
            self.Tlin[node_type] = Linear(data1.x_dict[node_type].shape[1], 2)
        # self.Tlin= Linear(data.x_dict['user'].shape[1], 4)

    def forward(self, x_dict, edge_index_dict):
        list3 = dict()
        for node_type, _ in x_dict[0].items():
            list3[node_type] = []
        for i in range(8):
            for node_type, x in x_dict[i].items():
                x_dict[i][node_type] = self.lin_dict[node_type][i](x).relu_()

            for conv in self.convs:
                x_dict[i] = conv(x_dict[i], edge_index_dict)

            for node_type, _ in x_dict[0].items():
                list3[node_type].append(self.out_dict[node_type](x_dict[i][node_type]))

        weight_norm = F.softmax(self.weight, dim=0)
        weight_type = F.softmax(self.weight_node_type, dim=0)
        Z_dict = dict()
        A_head = dict()
        X_head = dict()
        t_head = dict()
        T_head = dict()
        for node_type, _ in x_dict[0].items():
            # Z_dict[node_type] = weight_norm[0] * list3[node_type][0] + weight_norm[1] * list3[node_type][1]
            Z_dict[node_type] = weight_norm[0] * list3[node_type][0] + weight_norm[1] * list3[node_type][1] + \
                                weight_norm[2] * list3[node_type][2] + weight_norm[3] * list3[node_type][3] + \
                                weight_norm[4] * list3[node_type][4] + weight_norm[5] * list3[node_type][5] + \
                                weight_norm[6] * list3[node_type][6] + weight_norm[7] * list3[node_type][7]
            X_head[node_type] = self.lin[node_type](Z_dict[node_type])
            t_head[node_type] = self.Tlin[node_type](X_head[node_type])
            T_head[node_type] = torch.zeros((t_head[node_type].shape[0], t_head[node_type].shape[1]))
            for i in range(T_head[node_type].shape[0]):
                T_head[node_type][i, 0] = weight_type[0] * t_head[node_type][i, 0]
                T_head[node_type][i, 1] = weight_type[1] * t_head[node_type][i, 1]
            T_head[node_type] = F.softmax(T_head[node_type], dim=0)
        for edge_type in edge_index_dict.keys():
            A_head[edge_type] = torch.sigmoid(torch.mm(Z_dict[edge_type[0]], Z_dict[edge_type[2]].T))

        # Z_user = weight_norm[0] * list3[0] + weight_norm[1] * list3[1]
        # Z_tweet = self.lin(x_dict[0]['tweet'])
        # A_head = torch.sigmoid(torch.mm(Z_user, Z_tweet.T))
        # X_head = torch.relu_(torch.mm(Z_user, self.w_FC) + self.b_FC)
        # T_head = F.softmax(X_head , dim=1)
        return A_head, X_head, T_head, weight_norm, weight_type


model = HGT(hidden_channels=128, out_channels=16, num_heads=2, num_layers=2, num_view=8)
device = torch.device('cpu')
data, model = data1.to(device), model.to(device)
'''
with torch.no_grad():  # Initialize lazy modules.
    out = model(list1, data.edge_index_dict)
'''
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.00001)


def train():
    model.train()
    optimizer.zero_grad()
    input_list = copy.deepcopy(list1)
    A_hat, X_hat, T_hat, wX, wT = model(input_list, data.edge_index_dict)
    X_hat_view = multiview(X_hat, 4, 2)
    loss = 0
    loss += pow(torch.norm(A_hat[('user', 'to', 'tweet')] - A[('user', 'to', 'tweet')]), 2)
    loss += pow(torch.norm(A_hat[('tweet', 'to', 'user')] - A[('tweet', 'to', 'user')]), 2)
    # loss += F.cross_entropy(out[0][('user', 'to', 'director')], torch.LongTensor(gnd1))
    # loss += F.cross_entropy(out[0][('director', 'to', 'user')], torch.LongTensor(np.zeros(data['director'].x.shape[0])))
    # loss += torch.norm(out[1]['user'] - X['user'])/float(gnd1.shape[0])
    # loss += torch.norm(out[1]['tweet'] - X['tweet'])/float(gnd2.shape[0])
    # loss += torch.norm(out[1]['director'] - X['director'])
    # for node_type in data1.node_types:
        #loss += (torch.norm(out[2][node_type] - torch.FloatTensor(np.array(T[node_type]))))/data1[node_type].x.shape[0]
    for node_type in node_types:
        loss += pow(torch.norm(T_hat[node_type] - T[node_type]), 2)
    for i in range(4):
        loss += pow(torch.norm(X_hat_view['user'][i] - X_views['user'][i]), 2)
    for i in range(2):
        loss += pow(torch.norm(X_hat_view['tweet'][i] - X_views['tweet'][i]), 2)
    loss=loss/float(gnd.shape[0])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    input_list = copy.deepcopy(list1)
    out = model(input_list, data.edge_index_dict)
    ano_score = []
    for i in range(data1['user'].x.shape[0]):
        ano_score.append(0.48 * torch.norm(out[0][('user', 'to', 'tweet')][i] - A[('user', 'to', 'tweet')][i])
                         + 0.48 * torch.norm(out[1]['user'][i] - X['user'][i])+0.04 * torch.norm(out[2]['user'][i] - torch.FloatTensor(np.array(T['user']))))
    for i in range(data1['tweet'].x.shape[0]):
        ano_score.append(0.48 * torch.norm(out[0][('tweet', 'to', 'user')][i] - A[('tweet', 'to', 'user')][i])
                         + 0.48 * torch.norm(out[1]['tweet'][i] - X['tweet'][i])+0.04 * torch.norm(out[2]['tweet'][i] - torch.FloatTensor(np.array(T['tweet']))))
    index_ano = list(map(ano_score.index, heapq.nlargest(int(np.sum(gnd)), ano_score)))
    label_ano = np.zeros(gnd.shape[0])
    for i in range(gnd.shape[0]):
        if i in index_ano:
            label_ano[i]=1
    # acc = len(set(index_ano) & set(selected)) / 300.0
    auc = roc_auc_score(gnd, np.array(ano_score) / max(ano_score))
    Prec=precision_score(gnd, label_ano)
    Rec=recall_score(gnd, label_ano)
    F1=f1_score(gnd, label_ano)
    acc=accuracy_score(gnd, label_ano)
    return acc, Prec, Rec, F1, auc


for epoch in range(1, 101):
    loss = train()
    acc, Prec, Rec, F1, auc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}, Prec:{Prec:.4f}, Rec:{Rec:.4f}, F1:{F1:.4f}, AUC:{auc:.4f}')
