from model import DeeProBot
from dataset import training_data,val_data,test_data
import torch
from torch import nn
from utils import accuracy,init_weights
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve,auc
from tqdm import tqdm
import pandas as pd
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='name of the dataset used for training')
args = parser.parse_args()

device = 'cuda:0'
embedding_size,dropout,lr,weight_decay=128,0.1,1e-3,1e-2
batch_size=40

root='./'+args.dataset+'/'
train_loader =DataLoader(training_data(root=root),batch_size,shuffle=True)
val_loader = DataLoader(val_data(root=root),batch_size,shuffle=True)
test_loader = DataLoader(test_data(root=root),batch_size,shuffle=True)

model=DeeProBot().to(device=device)
loss=nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)


def train(epoch):
    model.train()
    for des_tensor,num_prop,labels in tqdm(train_loader):
        des_tensor,num_prop,labels=des_tensor.to(device=device),num_prop.to(device=device),labels.to(device=device)
        output = model(des_tensor,num_prop)
        loss_train = loss(output, labels)
        acc_train = accuracy(output, labels)
        #acc_val = accuracy(output[val_idx], labels[val_idx])
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
    print('Epoch: {:04d}'.format(epoch+1),
        'loss_train: {:.4f}'.format(loss_train.item()),
        'acc_train: {:.4f}'.format(acc_train.item()),
        #'acc_val: {:.4f}'.format(acc_val.item()),
        )
    return acc_train,loss_train

def val():
    for des_tensor,num_prop,labels in tqdm(val_loader):
        des_tensor,num_prop,labels=des_tensor.to(device=device),num_prop.to(device=device),labels.to(device=device)
        output = model(des_tensor,num_prop)
        loss_test = loss(output, labels)
        acc_test = accuracy(output, labels)
    print("Test set results:",
            "val_loss= {:.4f}".format(loss_test.item()),
            "val_accuracy= {:.4f}".format(acc_test.item()),
            )
    
def test():
    for des_tensor,num_prop,labels in tqdm(test_loader):
        des_tensor,num_prop,labels=des_tensor.to(device=device),num_prop.to(device=device),labels.to(device=device)
        output = model(des_tensor,num_prop)
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
            "precision= {:.4f}".format(precision.item()),
            "recall= {:.4f}".format(recall.item()),
            "f1_score= {:.4f}".format(f1.item()),
            #"mcc= {:.4f}".format(mcc.item()),
            "auc= {:.4f}".format(Auc.item()),
            )
    
model.apply(init_weights)

epochs=200
for epoch in range(epochs):
    train(epoch)
    val()
    
test()