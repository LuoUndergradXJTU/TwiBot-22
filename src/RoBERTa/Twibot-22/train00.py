import os
from tkinter import W
import numpy as np
import pandas as pd
import torch
os.environ['CUDA_VISIBLE_DEVICE'] = '5'
import math
from tqdm import tqdm
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

path0 = Path('src/RoBERTa/Twibot-22/data')


def eval(preds, labels):
    return accuracy_score(labels, preds)

            
class Twibot20Dataset(Dataset):
    def __init__(self, split, device='cuda:5'):
        self.device = torch.device(device)
        path1 = Path("data/twibot22")
        path2 = Path("src/RoBERTa/Twibot-22")
        
        tweets_tensor = torch.load(path1 / 'tweets_tensor.pt')
        des_tensor = torch.load(path1 / 'des_tensor.pt')
        label = 1 - torch.load(path2 / 'label_list.pt')
        
        self.tweet_feature = tweets_tensor[split]
        self.des_feature = des_tensor[split]
        self.label = label[split]
        self.length = len(self.tweet_feature)
        """
        batch_size here is useless
        """
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        return self.tweet_feature[index], self.des_feature[index], self.label[index]
    
    
class MLPclassifier(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=128,
                 dropout=0.5):
        super(MLPclassifier, self).__init__()
        self.dropout = dropout
        
        self.pre_model1 = nn.Linear(input_dim, input_dim // 2)
        self.pre_model2 = nn.Linear(input_dim, input_dim // 2)
        
        self.linear_relu_tweet = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        self.linear_relu2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_dim, 2)
    
    def forward(self,tweet_feature, des_feature):
        pre1 = self.pre_model1(tweet_feature)
        pre2 = self.pre_model2(des_feature)
        x = torch.cat((pre1,pre2), dim=1)
        x = self.linear_relu_tweet(x)
        # x = self.linear_relu1(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
class RobertaTrianer:
    def __init__(self,
                 train_loader,
                 test_loader,
                 epochs=300,
                 input_dim=768,
                 hidden_dim=128,
                 dropout=0.5,
                 activation='relu',
                 optimizer=torch.optim.Adam,
                 weight_decay=1e-5,
                 lr=1e-4,
                 device='cuda:4'):    
        self.epochs = epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        self.model = MLPclassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=dropout)
        self.device = device
        self.model.to(self.device)
        
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self, i, j):
        self.model.train()
        train_loader = self.train_loader
        best_acc = 0
        for epoch in range(self.epochs):
            loss_avg = 0
            preds = []
            labels = []
            with tqdm(train_loader) as progress_bar:
                for batch in progress_bar:
                    tweet = batch[0].to(self.device)
                    des = batch[1].to(self.device)
                    label = batch[2].to(self.device)
                    pred = self.model(tweet, des)
                    loss = self.loss_func(pred, label)
                    loss_avg += loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    progress_bar.set_description(desc=f'epoch={epoch}')
                    progress_bar.set_postfix(loss=loss.item())
                    
                    preds.append(pred.argmax(dim=-1).cpu().numpy())
                    labels.append(label.cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            labels = np.concatenate(labels, axis=0)
            loss_avg = loss_avg / len(train_loader)   
            print('{' + f'loss={loss_avg.item()}' + '}' + 'eval=', end='')    
            best_acc = self.test(best_acc)
        with open(path0 / 'result1.txt', 'a') as file:
            file.write(' '.join([str(i), str(j), '\n']))
            file.write(''.join([str(best_acc), '\n\n']))
            file.close()
    
        
    @torch.no_grad()
    def test(self, best_acc):
        preds = []
        labels = []
        test_loader = self.test_loader
        for batch in test_loader:
            tweet = batch[0].to(self.device)
            des = batch[1].to(self.device)
            label = batch[2].to(self.device)
            pred = self.model(tweet, des)
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            labels.append(label.cpu().numpy())
            
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)

        acc = eval(preds, labels)
        print(acc)
        if acc > best_acc:
            best_acc = acc
        return best_acc
  
     
        
if __name__ == '__main__':
    split = np.load(path0 / 'id.npy')
    
    for i in range(10):
        for j in range(10):
            train_dataset = Twibot20Dataset(split=split[i])
            test_dataset = Twibot20Dataset(split=split[j])
            
            print(len(train_dataset))
            print(len(test_dataset))
            
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
            
            trainer = RobertaTrianer(train_loader, test_loader)
            trainer.train(i, j)