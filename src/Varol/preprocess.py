import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from tqdm import tqdm
import nltk
from nltk import word_tokenize
from collections import defaultdict
import torch
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="cresci-2015", type=str)
parser.add_argument("--first_time", default=False, type=bool)
args = parser.parse_args()       

eps = 1e-5

def edge_to_list(edge_index):
    follow_adj_list = defaultdict(list)
    friend_adj_list = defaultdict(list)
    tweet_list = defaultdict(list)
    
    edge_index = edge_index[edge_index.relation == "followers"]
    print(len(edge_index))
    
    for i, edge in tqdm(edge_index.iterrows()):
        if edge.relation == "post":
            tweet_list[edge.source_id].append(edge.target_id)
        elif edge.relation == "follow":
            follow_adj_list[edge.source_id].append(edge.target_id)
        else:
            friend_adj_list[edge.source_id].append(edge.target_id)
    
    # src = list(edge_index.source_id)
    # dst = list(edge_index.target_id)
    # for s, d in tqdm(zip(src, dst)):
    #     friend_adj_list[s].append(d)
            
    return friend_adj_list, follow_adj_list, tweet_list

def skewness(samples):
    if isinstance(samples, list):
        samples = np.array(samples)
    mean = np.mean(samples)
    var = np.var(samples)
    a = np.mean((samples - mean) ** 3)
    b = var ** 1.5
    if abs(b) < eps:
        return 0.0
    return a/b

def kurtosis(samples):
    if isinstance(samples, list):
        samples = np.array(samples)
    mean = np.mean(samples)
    var = np.var(samples)
    a = np.mean((samples - mean) ** 4)
    b = var ** 2
    if abs(b) < eps:
        return 0.0
    return a/b

def featurize_dist(samples):
    feat = np.zeros(8)
    if len(samples) == 0:
        return feat
    feat[0] = min(samples)
    feat[1] = max(samples)
    feat[2] = np.median(samples)
    feat[3] = np.mean(samples)
    feat[4] = np.std(samples)
    feat[5] = np.var(samples)
    feat[6] = skewness(samples)
    feat[7] = kurtosis(samples)
    return feat
    
def featurize_friend(uid, node, adj_list):
    statistics = [[], [], [], [], []]
    
    for fid in adj_list[uid]:
        try:
            friend = node.loc[fid]
        except KeyError:
            continue
        statistics[0].append(time() - friend.created_at.timestamp())
        statistics[1].append(friend.public_metrics["followers_count"])
        statistics[2].append(friend.public_metrics["following_count"])
        statistics[3].append(friend.public_metrics["tweet_count"])
        if friend.description is None:
            statistics[4].append(0)
        else:
            statistics[4].append(len(friend.description))
        
    feats = np.concatenate(list(map(featurize_dist, statistics)))
    return feats
    
def featurize_user_metadata(uid, node):
    feats = []
    user = node.loc[uid]
    if user.name is not None:
        feats.append(len(user.name))
    else:
        feats.append(0)
    feats.append(sum(map(lambda x: x.isdigit(), user.name)))
    feats.append(len(user.username))
    feats.append((time() - user.created_at.timestamp()) // 86400)
    if user.description is not None:
        feats.append(len(user.description))
    else:
        feats.append(0)
    feats.append(user.public_metrics["followers_count"])
    feats.append(user.public_metrics["following_count"])
    feats.append(user.public_metrics["tweet_count"])
    feats.append(user.public_metrics["listed_count"])
    return np.array(feats)

tags = set(["VB", "PDT", "NN", "JJ", "MD", "UH", "RB", "WP", "PRP"])
tags_to_index = {}
for i, word in enumerate(list(tags)):
    tags_to_index[word] = i

def featurized_tweet(uid, node, tweet_list):
    pos_freq = [[]] * 9
    pos_portion = [[]] * 9
    num_words = []
    for tweet in tweet_list[uid]:
        freq = [0] * 9
        if tweet is None: 
            continue
        words = word_tokenize(tweet)
        num_words.append(len(words))
        pos_tags = nltk.pos_tag(words)
        for _, tag in pos_tags:
            if tag in tags:
                freq[tags_to_index[tag]] += 1
        for idx, f in enumerate(freq):
            pos_freq[idx].append(f)
            pos_portion[idx].append(f / len(words))
    pos_freq = list(map(featurize_dist, pos_freq))
    pos_portion = list(map(featurize_dist, pos_portion))
    feats = np.concatenate(pos_freq + pos_portion + [featurize_dist(num_words)])
    return feats

def featurize_graph(uid, node, adj_list):
    feats = []
    feats.append(len(adj_list[uid]))
    
def generate_data(dataset="cresci-2015"):
    # data_path = Path("../dataset/") / dataset
    data_path = Path(".")
    node = pd.read_json("node.json")
    node.set_index("id", inplace=True)
    edge_index = pd.read_csv(data_path / "edge.csv")
    label = pd.read_csv(data_path / "label.csv")
    label.set_index("id", inplace=True)
    split = pd.read_csv(data_path / "split.csv")
    split.set_index("id", inplace=True)
    
    # friend_adj_list, follow_adj_list, tweet_list = edge_to_list(edge_index)
    # torch.save(friend_adj_list, "friend_adj_list.pt")
    # torch.save(follow_adj_list, "follow_adj_list.pt")
    # torch.save(tweet_list, "tweet_list.pt")
    
    friend_adj_list = torch.load("friend_adj_list.pt")
    print(len(friend_adj_list))
    tweet_list = torch.load("tweet_list.pt")
    
    train_data = []
    valid_data = []
    test_data = []
    train_label = []
    valid_label = []
    test_label = []
    for idx, entry in tqdm(label.iterrows()):
        sp = split.loc[idx].split
        if sp == "train":
            train_data.append(np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)]))
            train_label.append(entry.label == "bot")
        elif sp == "val":
            valid_data.append(np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)]))
            valid_label.append(entry.label == "bot")
        else:
            test_data.append(np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)]))
            test_label.append(entry.label == "bot")
            
            
    train_data = np.stack(train_data, axis=0)
    valid_data = np.stack(valid_data, axis=0)
    test_data = np.stack(test_data, axis=0)
    train_label = np.array(train_label, dtype=np.int32)
    valid_label = np.array(valid_label, dtype=np.int32)
    test_label = np.array(test_label, dtype=np.int32)
    
    torch.save(train_data, "train_data.pt")
    torch.save(train_label, "train_label.pt")
    torch.save(valid_data, 'valid_data.pt')
    torch.save(valid_label, "valid_label.pt")
    torch.save(test_data, "test_data.pt")
    torch.save(test_label, "test_label.pt")
    
    return train_data, train_label, valid_data, valid_label, test_data, test_label


def non_parallel_generate_data():
    dataset = "Twibot-22"
    # data_path = Path(f"../datasets/{dataset}")
    data_path = Path(f"/data2/whr/TwiBot22-baselines/datasets/{dataset}")
    print(data_path)
    node = pd.read_json(data_path / "user.json")
    node.set_index("id", inplace=True)
    edge_index = pd.read_csv(data_path / "edge.csv")
    label = pd.read_csv(data_path / "label.csv")
    label.set_index("id", inplace=True)
    
    split = pd.read_csv(data_path / "split.csv")
    split.set_index("id", inplace=True)

    print("data load finished!")    
    
    # friend_adj_list, follow_adj_list, _  = edge_to_list(edge_index)
    # torch.save(friend_adj_list, f"./{dataset}/friend_adj_list.pt")
    # torch.save(follow_adj_list, f"./{dataset}/follow_adj_list.pt")
    # torch.save(tweet_list, f"./{dataset}/tweet_list.pt")

    friend_adj_list = torch.load(f"./{dataset}/friend_adj_list.pt")
    tweet_list = torch.load(f"./{dataset}/tweet_list.pt")

    train_data = mp.Manager().list([])
    valid_data = mp.Manager().list([])
    test_data = mp.Manager().list([])
    train_label = mp.Manager().list([])
    valid_label = mp.Manager().list([])
    test_label = mp.Manager().list([])

    def add(idx, entry, lock, train_data, train_label, valid_data, valid_label, test_data, test_label):
        sp = split.loc[idx].split
        # print(sp)
        feats = np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)])
        with lock:
            # print(number)
            print(len(train_label))
            if sp == "train":
                train_data.append(feats)
                train_label.append(entry.label == "bot")
            elif sp == "val":
                valid_data.append(feats)
                valid_label.append(entry.label == "bot")
            elif sp == "test":
                test_data.append(feats)
                test_label.append(entry.label == "bot")
            else:
                return
                            
    lock = mp.Manager().Lock()
    pool = mp.Pool(30)
    inputs = list(map(lambda x: list(x) + [lock], label.iterrows()))
    # print(inputs[0])
    # pool.apply_async(add, inputs, error_callback=lambda err: print(err))
    for arg in tqdm(inputs):
        pool.apply_async(add, tuple(arg + [train_data, train_label, valid_data, valid_label, test_data, test_label]), error_callback=lambda x: print(x))
    pool.close()
    pool.join()

    train_data = np.stack(train_data, axis=0)
    valid_data = np.stack(valid_data, axis=0)
    test_data = np.stack(test_data, axis=0)
    train_label = np.array(train_label, dtype=np.int32)
    valid_label = np.array(valid_label, dtype=np.int32)
    test_label = np.array(test_label, dtype=np.int32)

    torch.save(train_data, f"./{dataset}/train_data.pt")
    torch.save(train_label, f"./{dataset}/train_label.pt")
    torch.save(valid_data, f'./{dataset}/valid_data.pt')
    torch.save(valid_label, f"./{dataset}/valid_label.pt")
    torch.save(test_data, f"./{dataset}/test_data.pt")
    torch.save(test_label, f"./{dataset}/test_label.pt")


def chunk_based_saving():
    from tsai.all import create_empty_array, progress_bar
    

if __name__ == "__main__":
# def parallel_generate_data(dataset="cresci-2015"):
    # dataset = "Twibot-20"
    dataset = "cresci-2015"
    dataset = args.dataset
    # data_path = Path(f"../datasets/{dataset}")
    data_path = Path(f"/data2/whr/TwiBot22-baselines/datasets/{dataset}")
    print(data_path)
    node = pd.read_json(data_path / "node.json")
    node.set_index("id", inplace=True)
    edge_index = pd.read_csv(data_path / "edge.csv")
    label = pd.read_csv(data_path / "label.csv")
    label.set_index("id", inplace=True)
    
    split = pd.read_csv(data_path / "split.csv")
    split.set_index("id", inplace=True)

    print("data load finished!")    
    
    friend_adj_list, follow_adj_list, tweet_list = edge_to_list(edge_index)
    
    if args.first_time:
        friend_adj_list, follow_adj_list, _  = edge_to_list(edge_index)
        torch.save(friend_adj_list, f"./{dataset}/friend_adj_list.pt")
        torch.save(follow_adj_list, f"./{dataset}/follow_adj_list.pt")
        torch.save(tweet_list, f"./{dataset}/tweet_list.pt")

    friend_adj_list = torch.load(f"./{dataset}/friend_adj_list.pt")
    tweet_list = torch.load(f"./{dataset}/tweet_list.pt")

    if dataset == "cresci-2015":
        train_data = mp.Manager().list([])
        valid_data = mp.Manager().list([])
        test_data = mp.Manager().list([])
        train_label = mp.Manager().list([])
        valid_label = mp.Manager().list([])
        test_label = mp.Manager().list([])
    
    else:
        train_data = torch.zeros([70_0000, 201]).share_memory_()
        train_label = torch.zeros([70_0000]).share_memory_()
        valid_data = torch.zeros([20_0000, 201]).share_memory_()
        valid_label = torch.zeros([20_0000]).share_memory_()
        test_data = torch.zeros([10_0000, 201]).share_memory_()
        test_label = torch.zeros([10_0000]).share_memory_()

        ptr = mp.Manager().list([0, 0, 0])
    
    if dataset == "cresci-2015":
        def add(idx, entry, lock, train_data, train_label, valid_data, valid_label, test_data, test_label):
            sp = split.loc[idx].split
            # print(sp)
            feats = np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)])
            with lock:
                # print(number)
                print(len(train_label))
                if sp == "train":
                    train_data.append(feats)
                    train_label.append(entry.label == "bot")
                elif sp == "val":
                    valid_data.append(feats)
                    valid_label.append(entry.label == "bot")
                elif sp == "test":
                    test_data.append(feats)
                    test_label.append(entry.label == "bot")
                else:
                    return
    else:
        def add(idx, entry, lock, train_data, train_label, valid_data, valid_label, test_data, test_label, ptr):
            sp = split.loc[idx].split
            # print(sp)
            feats = torch.tensor(np.concatenate([featurize_friend(idx, node, friend_adj_list), featurize_user_metadata(idx, node), featurized_tweet(idx, node, tweet_list)]))
            
            if sp == "train":
                data = train_data
                label = train_label
                index = 0
            elif sp == "valid":
                data = valid_data
                label = valid_label
                index = 1
            elif sp == "test":
                data = test_data
                label = test_label
                index = 2
            
            with lock:
                # print(number)
                print(ptr)
                # if sp == "train":
                #     train_data[ptr[0]].copy_(feats)
                #     train_label[ptr[0]] = (entry.label == "bot")
                #     ptr[0] += 1
                # elif sp == "val":
                #     valid_data[ptr[1]].copy_(feats)
                #     valid_label[ptr[1]] = (entry.label == "bot")
                #     ptr[1] += 1
                # elif sp == "test":
                #     test_data[ptr[2]].copy_(feats)
                #     test_label[ptr[2]] = (entry.label == "bot")
                #     ptr[2] += 1
                # else:
                #     return
                data[ptr[index]].copy_(feats)
                label[ptr[index]] = (entry.label == "bot")
                ptr[index] += 1
                            
    lock = mp.Manager().Lock()
    pool = mp.Pool(30)
    # label = label[:5000]
    inputs = list(map(lambda x: list(x) + [lock], label.iterrows()))
    # print(inputs[0])
    # pool.apply_async(add, inputs, error_callback=lambda err: print(err))
    
    if dataset != "cresci-2015":
        for arg in tqdm(inputs):
            pool.apply_async(add, tuple(arg + [train_data, train_label, valid_data, valid_label, test_data, test_label, ptr]), error_callback=lambda x: print(x))
    
    else:
        for arg in tqdm(inputs):
            pool.apply_async(add, tuple(arg + [train_data, train_label, valid_data, valid_label, test_data, test_label]), error_callback=lambda x: print(x))
    pool.close()
    pool.join()
    
    
    if dataset == "cresci-2015":
        train_data = np.stack(train_data, axis=0)
        valid_data = np.stack(valid_data, axis=0)
        test_data = np.stack(test_data, axis=0)
    train_label = np.array(train_label, dtype=np.int32)
    valid_label = np.array(valid_label, dtype=np.int32)
    test_label = np.array(test_label, dtype=np.int32)

    torch.save(train_data, f"./{dataset}/train_data.pt")
    torch.save(train_label, f"./{dataset}/train_label.pt")
    torch.save(valid_data, f'./{dataset}/valid_data.pt')
    torch.save(valid_label, f"./{dataset}/valid_label.pt")
    torch.save(test_data, f"./{dataset}/test_data.pt")
    torch.save(test_label, f"./{dataset}/test_label.pt")

    # return train_data, train_label, valid_data, valid_label, test_data, test_label