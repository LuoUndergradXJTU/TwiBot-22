import json
import numpy as np
from nlp_features import *
file='train'
name='/data2/whr/lyh/project3/Twibot-20/'+file +'.json'
data=np.load('Twibot-20/node_fea.npy')
with open(name,'r') as f:
    users=json.load(f)
    for user in users:
        tweets=user['tweet']
        sent=sentiment(tweets)
        print()
