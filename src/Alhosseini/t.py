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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn import preprocessing

path = '../../datasets'
dataset1 = 'Twibot-22'
path1 = os.path.join(path, dataset1)
with open(os.path.join(path1, 'user.json'), 'r', encoding='UTF-8') as f:
    node1 = json.load(f)
# edge1 = pandas.read_csv(os.path.join(path1, 'edge.csv'))
label1 = pandas.read_csv(os.path.join(path1, 'label.csv'))
split1 = pandas.read_csv(os.path.join(path1, 'split.csv'))
print(set(split1['split']))