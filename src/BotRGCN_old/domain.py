from model import BotRGCN
from Dataset import Twibot20
import torch
from torch import nn
from utils import accuracy,init_weights

from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc

from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData

import pandas as pd
from tqdm import tqdm
import numpy as np

device = 'cuda:2'
embedding_size,dropout,lr,weight_decay=128,0.1,5e-4,5e-4

dataset=Twibot20(root='./data_twi22/',device='cpu',process=False,save=False)
des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,labels,_,_,_=dataset.dataloader()

model=BotRGCN(cat_prop_size=3,embedding_dimension=embedding_size).to(device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),
                    lr=lr,weight_decay=weight_decay)

def train(batch_id,des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,label_train,optimizer):
    model.train()
    des_tensor=des_tensor.to(device)
    tweets_tensor=tweets_tensor.to(device)
    num_prop=num_prop.to(device)
    category_prop=category_prop.to(device)
    edge_index=edge_index.to(device)
    edge_type=edge_type.to(device)
    label_train=label_train.to(device)
    output = model(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type)
    loss_train = loss(output, label_train)
    acc_train = accuracy(output, label_train)
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    return acc_train,loss_train

def test(des_tensor,tweets_tensor,num_prop,category_prop,edge_index,edge_type,label_test):
    model.eval()
    output = model(des_tensor.to(device),tweets_tensor.to(device),num_prop.to(device),category_prop.to(device),edge_index.to(device),edge_type.to(device))
    loss_test = loss(output, label_test.to(device))
    acc_test = accuracy(output, label_test.to(device))
    return acc_test.item()

def run(epochs,train_loader,test_loader,optimizer):
    model.apply(init_weights)
    acc_final=0.0
    for epoch in range(epochs):
        for i,sampled_data in enumerate(train_loader):
            des=sampled_data.x
            n_id=sampled_data.n_id
            tweet=tweets_tensor[n_id]
            num=num_prop[n_id]
            cat=category_prop[n_id]
            e_index=sampled_data.edge_index
            e_type=sampled_data.edge_type
            label=labels[n_id]
            train(i,des,tweet,num,cat,e_index,e_type,label,optimizer)
            
        for i,sampled_data in enumerate(test_loader):
            des=sampled_data.x
            n_id=sampled_data.n_id
            tweet=tweets_tensor[n_id]
            num=num_prop[n_id]
            cat=category_prop[n_id]
            e_index=sampled_data.edge_index
            e_type=sampled_data.edge_type
            label=labels[n_id]
            acc_test=test(des,tweet,num,cat,e_index,e_type,label)
        
        if acc_test>acc_final:
            acc_final=acc_test
            
    return acc_final


results=[]
for i in range(10):
    if i<9:
        continue
    single_result=[]
    for j in tqdm(range(10)):
        if j<8:
            continue
        print(i,j)
        train_idx=torch.load('./domain_split/user'+str(i)+'.pt')
        test_idx=torch.load('./domain_split/user'+str(j)+'.pt')
        graph = Data(x=des_tensor, edge_index=edge_index.to('cpu'),edge_type=edge_type,n_id = torch.arange(len(des_tensor)))
        train_loader = NeighborLoader(graph, num_neighbors=[256] * 2, input_nodes=train_idx, batch_size=5000)
        test_loader = NeighborLoader(graph, num_neighbors=[-1] * 2, input_nodes=test_idx, batch_size=10000)
        acc_test=run(400,train_loader,test_loader,optimizer)
        single_result.append(acc_test)
        np.save('./domain_split/results/result'+str(i)+'_'+str(j)+'.npy',acc_test)
    np.save('./domain_split/results/result'+str(i)+'.npy',single_result)
    results.append(single_result)
    
'''
df=pd.DataFrame(results,index=list(range(i+1)),columns=list(range(j+1)))
df.to_excel('./domain_split/results/result.xlsx')
df.to_csv('./domain_split/results/result.csv')
'''
