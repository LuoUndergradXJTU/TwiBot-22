import torch
from torch import nn
import torch.nn.functional as F

class DeeProBot(nn.Module):
    def __init__(self,des_size=50,num_prop_size=9,dropout=0.1):
        super(DeeProBot, self).__init__()
        self.dropout = dropout
        self.des_size = des_size
        self.num_prop_size = num_prop_size
        
        self.lstm =nn.Sequential(
            nn.LSTM(input_size=self.des_size,hidden_size=32,num_layers=2,batch_first=True),
        )
        
        self.linear1 =nn.Sequential(
            nn.Linear(32+self.num_prop_size,128),
            nn.ReLU()
        )
        
        self.linear2 =nn.Sequential(
            nn.Linear(128,32),
            nn.ReLU()
        )
        
        self.output =nn.Linear(32,2)
        
    def forward(self,des,num_prop):
        des_out=self.lstm(des)[0]
        des_out=F.relu(des_out.sum(dim=1))
        x=torch.cat((des_out,num_prop),dim=1)
        
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.linear1(x)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.linear2(x)
        x=self.output(x)
        
        return x
    

    