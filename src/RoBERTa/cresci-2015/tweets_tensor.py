import os
import json
import torch
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import *

pretrained_weights = 'roberta'
# tokenizer = RoBERTaModel.from_pretrained(pretrained_weights)
# model = RoBERTaModel.from_pretrained(pretrained_weights)
feature_extractor = pipeline('feature-extraction', model='roberta-base',tokenizer='roberta-base', device=2, padding=True, truncation=True, max_length=50)

users_tweets = np.load('src/T5/cresci-2017/user_tweets_dict.npy', allow_pickle=True).tolist()

tweets_tensor = []
for i in tqdm(range(len(users_tweets))):
    user_tweets_tensor = []
    try:
        for tweet in users_tweets[i]:
            tweet_tensor = torch.tensor(feature_extractor(tweet)).squeeze(0)
            tweet_tensor = torch.mean(tweet_tensor, dim=0)
            user_tweets_tensor.append(tweet_tensor)
        user_tweets_tensor = torch.mean(torch.stack(user_tweets_tensor), dim=0)
    except:
        user_tweets_tensor = torch.zeros(768)
    tweets_tensor.append(user_tweets_tensor)

path3 = Path('src/RoBERTa/cresci-2017')
tweets_tensor = torch.stack(tweets_tensor)
torch.save(tweets_tensor, path3 / 'tweets_tensor.pt')