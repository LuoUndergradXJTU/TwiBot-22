import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import ijson
import json

f1=open("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/tweet_4.json",'rb')
f2=open("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/tweet_4.json",'rb')
out_file=open("/data2/whr/czl/TwiBot22-baselines/src/twibot22-feature-extraction/tweet_data/tweet_4.json",'w')
tweet = ijson.items(f1, 'item.text')
tid=ijson.items(f2,'item.id')
tweet_dict={}
for i,(t_id,text) in enumerate(tqdm(zip(tid,tweet))):
    tweet_dict[t_id] = text

json.dump(tweet_dict, out_file,ensure_ascii=False)
f1.close()
f2.close()
out_file.close()

