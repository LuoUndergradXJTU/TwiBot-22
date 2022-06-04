import re
import numpy as np
import fasttext
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm import tqdm
import os
from nlp_features import *
f1=open('Twibot-22/data_include.json','r')
f2=open('Twibot-22/tweet_include.json','r')
data_include=json.load(f1)
tweet_include=json.load(f2)
def nlp_fea(data,tweets):
    fea=[]
    model,classifier=init()
    
    for i in tqdm(range(len(data))):
            user=data[i]
            tweet=tweets[i]
            try:
                l_no=links_no(tweet)
                l_per=l_no/len(tweet)
            except:
                l_no=0
                l_per=0
            try:
                m_no=mentions_no(tweet)
                m_per=m_no/len(tweet)
            except:
                m_no=0
                m_per=0
            try:
                tweets_no=eval(user['profile']['statuses_count'])
            except:
                tweets_no=0
            try:
                av_tweet_len,std_tweet_len=tweet_len(tweet)
            except:
                av_tweet_len,std_tweet_len=0,0
            try:
                no_langs,per_en,no_odd_langs,per_legit=lan(tweet[:50],model)
            except:
                no_langs,per_en,no_odd_langs,per_legit=0,0,0,0
            try:
                av_sent,std_sent,pos=sentiment(tweet[:20],classifier)
            except:
                av_sent,std_sent,pos=0,0,0
            fea.append([l_no,m_no,tweets_no,l_per,m_per,av_tweet_len,std_tweet_len,no_langs,per_en,no_odd_langs,per_legit,av_sent,std_sent,pos])
       
        
    
    return np.concatenate(fea)


fea=nlp_fea(data_include,tweet_include)

np.save('Twibot-22/nlp_22.npy',fea)
print('nlp_feature saved!')