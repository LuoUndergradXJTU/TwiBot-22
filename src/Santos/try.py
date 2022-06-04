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
import ijson
parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use GDC preprocessing.')
args = parser.parse_args()


path = '../../datasets'
dataset1 = 'Twibot-22'
i=0
path1 = os.path.join(path, dataset1)
with open(os.path.join(path1, 'tweet_0.json'), 'r', encoding = 'UTF-8') as f:
    obj = ijson.items(f, "item")
    for x in tqdm(obj): 
      i+=1     
      if i==2:
        print(x)
        break