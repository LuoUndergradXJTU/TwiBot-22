import os
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

split = [[], [], []]
path0 = Path('datasets/Twibot-22')
split_list = pd.read_csv(path0 / 'split.csv')
label = pd.read_csv(path0 / 'label.csv')

users_index_to_uid = list(label['id'])
uid_to_users_index = {x : i for i, x in enumerate(users_index_to_uid)}
for id in split_list[split_list['split'] == 'train']['id']:
    split[0].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'val']['id']:
    split[1].append(uid_to_users_index[id])
for id in split_list[split_list['split'] == 'test']['id']:
    split[2].append(uid_to_users_index[id])

def eval(preds_auc, preds, labels):
    print("ACC:{}".format(accuracy_score(labels, preds)), end=",")
    print("F1:{}".format(f1_score(labels, preds)), end=",")
    print("ROC:{}".format(roc_auc_score(labels, preds_auc)))
    print("precision_score:{}".format(precision_score(labels, preds)), end=",")
    print("recall_score:{}".format(recall_score(labels, preds)))

            
class Twibot20Dataset(Dataset):
    def __init__(self, name, device='cuda:5'):
        self.device = torch.device(device)
        path1 = Path("data/twibot22/T5-tweet")
        path2 = Path("src/RoBERTa/Twibot-22")
        
        tweets_tensor = torch.load(path1 / 'tweets_tensor.pt')
        des_tensor = torch.load(path1 / 'des_tensor.pt')
        label = 1 - torch.load(path2 / 'label_list.pt')
        
        if name == 'train':
            self.tweet_feature = tweets_tensor[split[0]]
            self.des_feature = des_tensor[split[0]]
            self.label = label[split[0]]
            self.length = len(self.tweet_feature)
        elif name == 'val':
            self.tweet_feature = tweets_tensor[split[1]]
            self.des_feature = des_tensor[split[1]]
            self.label = label[split[1]]
            self.length = len(self.tweet_feature)
        else:
            self.tweet_feature = tweets_tensor[split[2]]
            self.des_feature = des_tensor[split[2]]
            self.label = label[split[2]]
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
                 input_dim=512,
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
                 val_loader,
                 test_loader,
                 epochs=50,
                 input_dim=512,
                 hidden_dim=128,
                 dropout=0.5,
                 activation='relu',
                 optimizer=torch.optim.Adam,
                 weight_decay=1e-5,
                 lr=1e-4,
                 device='cuda:5'):    
        self.epochs = epochs
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        self.model = MLPclassifier(input_dim=self.input_dim, hidden_dim=self.hidden_dim, dropout=dropout)
        self.device = device
        self.model.to(self.device)
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self):
        train_loader = self.train_loader
        for epoch in range(self.epochs):
            self.model.train()
            loss_avg = 0
            preds = []
            preds_auc = []
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
                    preds_auc.append(pred[:,1].detach().cpu().numpy())
                    labels.append(label.cpu().numpy())
            
            preds = np.concatenate(preds, axis=0)
            preds_auc = np.concatenate(preds_auc, axis=0)
            labels = np.concatenate(labels, axis=0)
            loss_avg = loss_avg / len(train_loader)   
            print('{' + f'loss={loss_avg.item()}' + '}' + 'eval=', end='')
            eval(preds_auc, preds, labels)     
            self.valid()
            self.test()
        
    @torch.no_grad()
    def valid(self):
        self.model.eval()
        preds = []
        preds_auc = []
        labels = []
        val_loader = self.val_loader
        for batch in val_loader:
            tweet = batch[0].to(self.device)
            des = batch[1].to(self.device)
            label = batch[2].to(self.device)
            pred = self.model(tweet, des)
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(label.cpu().numpy())
        

        preds = np.concatenate(preds, axis=0)
        preds_auc = np.concatenate(preds_auc, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        eval(preds_auc, preds, labels)
        
    @torch.no_grad()
    def test(self):
        self.model.eval()
        preds = []
        preds_auc = []
        labels = []
        test_loader = self.test_loader
        for batch in test_loader:
            tweet = batch[0].to(self.device)
            des = batch[1].to(self.device)
            label = batch[2].to(self.device)
            pred = self.model(tweet, des)
            preds.append(pred.argmax(dim=-1).cpu().numpy())
            preds_auc.append(pred[:,1].detach().cpu().numpy())
            labels.append(label.cpu().numpy())
            
        preds = np.concatenate(preds, axis=0)
        preds_auc = np.concatenate(preds_auc, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        eval(preds_auc, preds, labels)
  
     
        
if __name__ == '__main__':
    train_dataset = Twibot20Dataset('train')
    val_dataset = Twibot20Dataset('val')
    test_dataset = Twibot20Dataset('test')
    
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    for i in range(5):
        trainer = RobertaTrianer(train_loader, val_loader, test_loader)
        trainer.train()