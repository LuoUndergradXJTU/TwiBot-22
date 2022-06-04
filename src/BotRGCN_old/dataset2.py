import torch
import numpy as np
import pandas as pd
import json
import os
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm

class training_data(Dataset):
    def __init__(self,root='./data_twi22/'):
        self.root=root
        self.idx=torch.load(root+'train_idx.pt')
        self.des=torch.load(root+'des_tensor.pt')[self.idx]
        self.tweet=torch.load(root+'tweets_tensor.pt')[self.idx]
        self.cat_prop=torch.load(root+'cat_properties_tensor.pt')[self.idx]
        self.num_prop=torch.load(root+'num_properties_tensor.pt')[self.idx]
        self.label=torch.load(root+'label.pt')[self.idx]
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self,idx):
        des=self.des[idx]
        tweet=self.tweet[idx]
        cat_prop=self.cat_prop[idx]
        num_prop=self.num_prop[idx]
        label=self.label[idx]
        
        return des,tweet,num_prop,cat_prop,label
    
class val_data(Dataset):
    def __init__(self,root='./data_twi22/'):
        self.root=root
        self.idx=torch.load(root+'val_idx.pt')
        self.des=torch.load(root+'des_tensor.pt')[self.idx]
        self.tweet=torch.load(root+'tweets_tensor.pt')[self.idx]
        self.cat_prop=torch.load(root+'cat_properties_tensor.pt')[self.idx]
        self.num_prop=torch.load(root+'num_properties_tensor.pt')[self.idx]
        self.label=torch.load(root+'label.pt')[self.idx]
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self,idx):
        des=self.des[idx]
        tweet=self.tweet[idx]
        cat_prop=self.cat_prop[idx]
        num_prop=self.num_prop[idx]
        label=self.label[idx]
        
        return des,tweet,num_prop,cat_prop,label
    
class test_data(Dataset):
    def __init__(self,root='./data_twi22/'):
        self.root=root
        self.idx=torch.load(root+'test_idx.pt')
        self.des=torch.load(root+'des_tensor.pt')[self.idx]
        self.tweet=torch.load(root+'tweets_tensor.pt')[self.idx]
        self.cat_prop=torch.load(root+'cat_properties_tensor.pt')[self.idx]
        self.num_prop=torch.load(root+'num_properties_tensor.pt')[self.idx]
        self.label=torch.load(root+'label.pt')[self.idx]
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self,idx):
        des=self.des[idx]
        tweet=self.tweet[idx]
        cat_prop=self.cat_prop[idx]
        num_prop=self.num_prop[idx]
        label=self.label[idx]
        
        return des,tweet,num_prop,cat_prop,label