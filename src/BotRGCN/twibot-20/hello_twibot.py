import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from dataset_tool import fast_merge,df_to_mask
import os

def main():
    print('loading raw data')
    
    if not os.path.exists("processed_data"):
        print("processed_data exists")
    else:
        print("processed_data does not exist")
        
    node=pd.read_json("../datasets/Twibot-20/node.json")
    edge=pd.read_csv("../datasets/Twibot-20/edge.csv")
    label=pd.read_csv("../datasets/Twibot-20/label.csv")
    split=pd.read_csv("../datasets/Twibot-20/split.csv")
    
        
if __name__ == '__main__':
    main()