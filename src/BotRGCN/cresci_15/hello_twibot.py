import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from dataset_tool import fast_merge,df_to_mask
import os

def main():
    print('loading raw data')
    paths = ["../datasets/cresci-2015/node.json", "../datasets/cresci-2015/edge.csv", "../datasets/cresci-2015/label.csv", "../datasets/cresci-2015/split.csv"]
    for path in paths:
        if os.path.exists(path):
            print(f"{path} exists")
        else:
            print("processed_data does not exist")
    
        
if __name__ == '__main__':
    main()