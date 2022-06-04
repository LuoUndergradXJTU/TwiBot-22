
import re
import numpy as np
import fasttext
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm import tqdm
import os

dataset_name='Twibot-22'
id_include=(np.load(dataset_name+'/id_include.npy',allow_pickle=True))
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

id_include=list(id_include.item())
train_id=np.load(r'/data2/whr/lyh/baseline2/Twibot-22/train_id.npy')
test_id=np.load(r'/data2/whr/lyh/baseline2/Twibot-22/test_id.npy')
val_id=np.load(r'/data2/whr/lyh/baseline2/Twibot-22/val_id.npy')
id_list=np.load('/data2/whr/lyh/twibot22_baseline/'+dataset_name+'/id.npy')
id_list=list(id_list)
    
def links_no(tweets):
    patt=r'https'
    pattern=re.compile(patt)
    count=0
    for tweet in tweets:
        count=count+len(pattern.findall(tweet))
    return count

def mentions_no(tweets):
    patt='@'
    pattern=re.compile(patt)
    count=0
    for tweet in tweets:
        count=count+len(pattern.findall(tweet))
    return count

def tweet_len(tweets):
    total_num=len(tweets)
    count=np.zeros(total_num)
    for i,tweet in enumerate(tweets):
        count[i]=len(tweet.split(' '))
    return count.mean(),count.var()

def lan(tweets,model):
    
    #tweets=[tweet.rstrip() for tweet in tweets]
    predict=[]
    for tweet in tweets:
        twi=tweet.split('\n')
        tmp=[]
        for t in twi:
            if t == '':
                continue
            pred,_=model.predict(t)
            tmp=tmp+list(pred)
        predict=predict+tmp
    #predict=model.predict(tweets)
    no_languages=len(set(predict))
    per_en=predict.count('__label__en')/len(predict)
    d_langs=set(predict)
    no_odd_langs=0
    twi=0
    for lang in d_langs:
        num_lan=predict.count(lang)
        if num_lan>0.1*len(predict):
            no_odd_langs=no_odd_langs+1
            twi=twi+num_lan
    per_legit=twi/len(predict)
    return no_languages,per_en,no_odd_langs,per_legit
def init():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"     # 选择想要的模型
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer,device = 6)
    PRETRAINED_MODEL_PATH = '/data2/whr/lyh/baseline2/tmp/lid.176.bin'
    model = fasttext.load_model(PRETRAINED_MODEL_PATH)
    print("init finished")
    return model,classifier
    
def sentiment(tweets,classifier):
   
    sent=np.zeros(len(tweets))
    pos=0
    for i,tweet in enumerate(tweets):
        res=classifier(tweet)[0]
        if(res['label']=='LABEL_2'):
            pos=pos+1
        sent[i]=res['score']
    

    return sent.mean(),sent.var(),pos/len(tweets)


def nlp_fea(data):
    fea=[]
    model,classifier=init()
    
    for tweet in tqdm(data):
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
            no_langs,per_en,no_odd_langs,per_legit=lan(tweet[:10],model)
        except:
            no_langs,per_en,no_odd_langs,per_legit=0,0,0,0
        try:
            av_sent,std_sent,pos=sentiment(tweet[:10],classifier)
        except:
            av_sent,std_sent,pos=0,0,0
        fea.append([l_no,m_no,tweets_no,l_per,m_per,av_tweet_len,std_tweet_len,no_langs,per_en,no_odd_langs,per_legit,av_sent,std_sent,pos])
    return np.array(fea)

if __name__ == '__main__':
    files=['train','val','test']
    #files=['node']
    data=[]
    id_tweet=json.load(open(r"/data2/whr/czl/TwiBot22-baselines/src/twibot22_Botrgcn_feature/id_tweet.json",'r'))
    tweet=[]


   
    nlp=nlp_fea(id_tweet)
    np.save(dataset_name+'/'+'nlp_revised.npy',nlp)
    print('nlp_feature saved!')
    
        