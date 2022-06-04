from pathlib import Path
import pandas as pd
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import torch
import numpy as np
import os
from torch_geometric.data import Data, HeteroData
import ijson


def get_data_dir(server_id):
    if server_id == "206":
        return Path("/new_temp/fsb/Twibot22-baselines/datasets")
    elif server_id == "208":
        return Path("")
    elif server_id == "209":
        return Path("/data2/whr/TwiBot22-baselines/datasets")
    else:
        raise NotImplementedError

dataset_names = [
    'botometer-feedback-2019', 'botwiki-2019', 'celebrity-2019', 'cresci-2015', 'cresci-2017', 'cresci-rtbust-2019', 'cresci-stock-2018', 'gilani-2017', 'midterm-2018', 'political-bots-2019', 'pronbots-2019', 'vendor-purchased-2019', 'verified-2019', "Twibot-20","Twibot-22"]

def merge(dataset="Twibot-20", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    
    node_info = pd.read_csv(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    node_info = pd.merge(node_info, label)

def split_user_and_tweet(df):
    """
    split user and tweet from df, and ignore entries whose id is `None`
    """
    df = df[df.id.str.len() > 0]
    return df[df.id.str.contains("^u")], df[df.id.str.contains("^t")]
    

def getdict(dicttemp):
    dict1 = dicttemp
    dict2 = ["id", "text"]
    dict3 = {}
    for i in dict2:
        dict3[i] = dict1[i]
    
    return dict3

def gettweet(datanum, dataset="Twibot-20", server_id="209"):

    dataset_dir = get_data_dir(server_id) / dataset
  
    with open(str(dataset_dir) + "/" + "tweet_" + str(datanum) + ".json", "rb") as df_del:
        data =ijson.items(df_del, 'item')
        nodejson1 = [getdict(deld) for deld in data]
    
    return nodejson1
        

    
def fast_merge( dataset="Twibot-20", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    
    if dataset == "Twibot-22":
        tweetor = []
        for i in range(9):
            tweettemp = gettweet(datanum = i, dataset = dataset)
            tweetor += tweettemp

        tweet = {}
        for tweetdic in tweetor:
            tweet[tweetdic["id"]] = tweetdic["text"]        
    else:    
        node_info = pd.read_json(dataset_dir / "node.json")
        _, tweet = split_user_and_tweet(node_info)
        tweetkey = tweet["id"].tolist()
        tweetvalue = tweet["text"].tolist()
        tweet = dict(zip(tweetkey, tweetvalue))
        
    
    return tweet
        
    


def merge_and_split(dataset="botometer-feedback-2019", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    if dataset == "Twibot-22":
        node_info = pd.read_json(dataset_dir / "user.json")
    else:
        node_info = pd.read_json(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")
    node_info = pd.merge(node_info, label)
    node_info = pd.merge(node_info, split)
    
    
    train = node_info[node_info["split"] == "train"]
    valid = node_info[node_info["split"] == "val"]
    test = node_info[node_info["split"] == "test"]
    
    
    return train, valid, test


    



