import json
from tqdm import tqdm

user = json.load(open("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/user.json", "r"))
id_tweet = json.load(open("/data2/whr/czl/TwiBot22-baselines/src/twibot22_Botrgcn_feature/id_tweet.json", "r"))

ret = {}
for u, t in tqdm(zip(user, id_tweet.values())):
    ret[u["id"]] = t
    
import torch as th

th.save(ret, "./Twibot-22/tweet_list.pt")