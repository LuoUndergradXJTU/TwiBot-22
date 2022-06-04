import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from dataset import fast_merge

user, tweet = fast_merge("Twibot-20", '209')

user_index_to_uid = list(user.id)
tweet_index_to_tid = list(tweet.id)
        
uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}
tid_to_tweet_index = {x : i for i, x in enumerate(tweet_index_to_tid)}

edge=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/edge.csv")

edge=edge[edge.relation=='post']

src = list(edge[edge["relation"] == 'post']["source_id"])
dst = list(edge[edge["relation"] == 'post']["target_id"])

new_src = []
new_dst = []
            
for s, t in tqdm(zip(src, dst)):
    new_src.append(s)
    new_dst.append(t)
                
src = new_src
dst = new_dst
            
src = list(map(lambda x: uid_to_user_index[x], src))
dst = list(map(lambda x: tid_to_tweet_index[x], dst))

edge['target_id']=list(map(lambda x:tid_to_tweet_index[x],edge['target_id'].values))

edge['source_id']=list(map(lambda x:uid_to_user_index[x],edge['source_id'].values))



dict={i:[] for i in range(len(user))}
for i in tqdm(range(len(user))):
    try:
        edge['source_id'][i]
    except KeyError:
        continue
    else:
        dict[edge['source_id'][i]].append(tweet['text'][edge['target_id'][i]+len(user)])

np.save('user_tweets.npy',dict)