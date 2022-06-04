import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, HeteroData
from datetime import datetime as dt
# dataset
# from dataset import fast_merge,df_to_mask

from tqdm import tqdm
import os
print(os.getcwd())
# node=pd.read_json("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/node.json")
edge=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/edge.csv")
label=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/label.csv")
split=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/split.csv")
print(edge.shape) # (7086134, 3) 
print(edge.head())
print(label.shape)
print(label.head())
print(split.shape)
print(split.head())
# /data2/whr/czl/TwiBot22-baselines/src/THAVASIMANI et al./test1.py