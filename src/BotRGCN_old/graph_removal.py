from model import BotMLP
from dataset2 import training_data,val_data,test_data
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch.utils.data import DataLoader
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
from tqdm import tqdm
import numpy as np

device = 'cuda:4'
embedding_size,dropout,lr,weight_decay=128,0.1,0.0001,5e-4
batch_size=500

root='./data_twi20/'
print('loading train loader')
train_loader =DataLoader(training_data(root=root),batch_size,shuffle=True)
#val_loader = DataLoader(val_data(root=root),batch_size,shuffle=True)
print('loading test loader')
test_loader = DataLoader(test_data(root=root),len(torch.load(root+'test_idx.pt')),shuffle=True)

model=BotMLP().to(device=device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)


def train(epoch):
    model.train()
    for des_tensor,tweet_tensor,num_prop,cat_prop,labels in tqdm(train_loader):
        des_tensor,tweet_tensor,num_prop,cat_prop,labels=des_tensor.to(device=device),tweet_tensor.to(device=device),num_prop.to(device=device),cat_prop.to(device=device),labels.to(device=device)
        output = model(des_tensor,tweet_tensor,num_prop,cat_prop)
        loss_train = loss(output, labels)
        acc_train = accuracy(output, labels)
        #acc_val = accuracy(output[val_idx], labels[val_idx])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        )
    return acc_train,loss_train

def test():
    for des_tensor,tweet_tensor,num_prop,cat_prop,labels in tqdm(test_loader):
        des_tensor,tweet_tensor,num_prop,cat_prop,labels=des_tensor.to(device=device),tweet_tensor.to(device=device),num_prop.to(device=device),cat_prop.to(device=device),labels.to(device=device)
        output = model(des_tensor,tweet_tensor,num_prop,cat_prop)
        loss_test = loss(output, labels)
        acc_test = accuracy(output, labels)
        output=output.max(1)[1].to('cpu').detach().numpy()
        label=labels.to('cpu').detach().numpy()
        f1=f1_score(label,output)
        #mcc=matthews_corrcoef(label, output)
        precision=precision_score(label,output)
        recall=recall_score(label,output)
        fpr, tpr, thresholds = roc_curve(label, output, pos_label=1)
        Auc=auc(fpr, tpr)
    print("Test set results:",
            "test_loss= {:.4f}".format(loss_test.item()),
            "test_accuracy= {:.4f}".format(acc_test.item()),
            "f1_score= {:.4f}".format(f1.item()),
            )
    
model.apply(init_weights)

epochs=300
for epoch in range(epochs):
    train(epoch)
    test()
