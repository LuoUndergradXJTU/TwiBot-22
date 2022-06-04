import numpy as np
import pandas as pd
import os
import torch
import datetime
from torch_geometric.data import Data, HeteroData
import nltk
import pickle

import sys
sys.path.append("..")

from dataset import merge_and_split, get_data_dir, fast_merge, split_user_and_tweet


def preprocess_dataset(dataset, server_id="209"):
    train, valid, test = merge_and_split(dataset=dataset, server_id=server_id)
    train = pd.concat([train, valid])
    return train, test


    
def get_directdata(dataset, server_id = "209"):
    train, valid, testdata = merge_and_split(dataset=dataset, server_id=server_id)
    #print(train)
    traindata = pd.concat([train, valid])

    train_createtime = list(traindata["created_at"])
    trainpublic_metrics = list(traindata["public_metrics"])
   
    test_createtime = list(testdata["created_at"])
    testpubic_metrics = list(testdata["public_metrics"])
  
    
    createtime = train_createtime + test_createtime
    public_metrics = trainpublic_metrics + testpubic_metrics
    friend = []
    followers = []
    tcount = []
    listcount = []
    
    for dict in public_metrics:
        friend.append(dict["following_count"])
        followers.append(dict["followers_count"])
        tcount.append(dict["tweet_count"])
        listcount.append(dict["listed_count"])
    
    
    if isinstance(createtime[0], datetime.datetime) == False:
        a = 0
        for i, strtime in enumerate(createtime):
            if createtime[i][-1] == "L":
                strtime = "Thu Mar 06 02:37:29 +0000 2014"
            strtime = datetime.datetime.strptime(strtime , "%a %b %d %H:%M:%S %z %Y")
            createtime[i] = strtime
        

               
    datasetdatedict = {"botometer-feedback-2019":"2019-01-01 00:00:00+00:00", "cresci-2015":"2015-01-01 00:00:00+00:00", "cresci-2017":"2017-01-01 00:00:00+00:00", "cresci-rtbust-2019":"2019-01-01 00:00:00+00:00", "cresci-stock-2018":"2018-01-01 00:00:00+00:00", "gilani-2017": "2017-01-01 00:00:00+00:00", "midterm-2018":"2018-01-01 00:00:00+00:00", "Twibot-20":"2020-01-01 00:00:00+00:00", "Twibot-22": "2022-01-01 00:00:00+00:00"}
    
    seconds = []
    
    if dataset in datasetdatedict.keys():
        nowtime = datetime.datetime.strptime(datasetdatedict[dataset], "%Y-%m-%d %H:%M:%S%z")
        for trtime in createtime:
            seconds.append((nowtime - trtime).total_seconds())

        
        
        
    return friend, followers, tcount, listcount, seconds



def text_feature(dataset, server_id="209"):
    dataset_dir = get_data_dir(server_id) / dataset
    
    
    tweet = fast_merge(dataset, server_id)
    
    #print(tweet)
    
    
    train, valid, test = merge_and_split(dataset)
    
    allid = list(train["id"]) + list(valid["id"]) + list(test["id"])
    
     
    
    edge = pd.read_csv(dataset_dir / "edge.csv")
    edge_types = edge["relation"]
    sourceid = edge["source_id"]
    targetid = edge["target_id"]
    src2dst ={}
    for i,edge_type in enumerate(edge_types):
        usertweet = [] 
        if edge_type == "post":
            if sourceid[i] not in src2dst.keys():
                usertweet.append(targetid[i])
                src2dst[sourceid[i]] =  usertweet
            else:
                src2dst[sourceid[i]].append(targetid[i])
    
    hashtags = []
    urls = []
    mentions = []
    uniqueurl = []
    uniquementions = []
    uniquehashtag = []
    for uid in allid:
        tags_list = []
        mentions_list = []
        urls_list = []
        if uid not in src2dst.keys():
            hashtags.append(0)
            mentions.append(0)
            urls.append(0)
            uniqueurl.append(0)
            uniquementions.append(0)
            uniquehashtag.append(0)
            continue
        else:
            tlist = src2dst[uid]
            for tid in tlist:
                tokens = []
                if tid in tweet.keys():
                    if tweet[tid] is not None:
                        tokens += tweet[tid].split()
                        tweet.pop(tid)
                        for token in tokens:
                            if "@" in token:
                                mentions_list.append(token.strip(":"))
                            if "#" in token:
                                tags_list.append(token.strip(":"))
                            if "http" in token:
                                urls_list.append(token)
                                    
            uniquementionlist = set(mentions_list)
            uniquetagslist = set(tags_list)
            uniqueurllist = set(urls_list)
            
            mentions.append(len(mentions_list))
            hashtags.append(len(tags_list))
            urls.append(len(urls_list))
            uniquementions.append(len(uniquementionlist))  
            uniquehashtag.append(len(uniquetagslist))
            uniqueurl.append(len(uniqueurllist))
    
    return  mentions, hashtags, urls, uniquementions, uniquehashtag, uniqueurl

if __name__ == '__main__':
    text_feature("Twibot-22", server_id="209")
    
    
    