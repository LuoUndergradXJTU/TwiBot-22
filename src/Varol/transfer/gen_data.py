import os
import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
import torch.multiprocessing as mp
import sys
sys.path.append("..")
from preprocess import *
from tqdm import tqdm

def generate_transfer_data():
    data_path = Path("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22")
    node = pd.read_json(data_path / "user.json")
    node.set_index("id", inplace=True)
    label = pd.read_csv(data_path / "label.csv")
    label.set_index("id", inplace=True)

    friend_adj_list = torch.load("../Twibot-22/friend_adj_list.pt")
    tweet_list = torch.load("../Twibot-22/tweet_list.pt")

    def generate_from_split(split):
        data = torch.zeros([10_000, 201])
        l = torch.zeros([10_000])
        for ptr, idx in tqdm(enumerate(split)):
            feats = torch.tensor(np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)]))
            data[ptr].copy_(feats)
            l[ptr] = (label.loc[idx].label == "bot")
        return data, l
    
    def parallel_generate_from_split(split, friend_adj_list, node, tweet_list, label):
        data = torch.zeros([10_000, 201]).share_memory_()
        l = torch.zeros([10_000]).share_memory_()
        def add(idx, lock, data, l, ptr):
            feats = torch.tensor(np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)]))
            la = label.loc[idx].label == "bot"
            with lock:
                data[ptr[0]].copy_(feats)
                l[ptr[0]] = la
                ptr[0] += 1
                print(ptr[0])
        
        ptr = mp.Manager().list([0])
        lock = mp.Manager().Lock()
        pool = mp.Pool(10)
        for idx in split:
            pool.apply_async(add, (idx, lock, data, l, ptr), error_callback=print)
        pool.close()
        pool.join()
        return data, l


    base_path = Path("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/domain")
    file_names = ["user{}.json".format(i) for i in range(0, 1)]
    for file in file_names:
        file_path = base_path / file
        split = json.load(open(file_path, "r"))
        data, l = generate_from_split(split)
        torch.save({"data": data, "label": np.array(l, dtype=np.int32)}, f"./{file[4]}.pt")
if __name__ == "__main__":
    generate_transfer_data()
        


