
import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline, T5Tokenizer, T5EncoderModel
import os
os.environ['CUDA_VISIBLE_DEVICE'] = '1'
import json
DEVICE = 1

each_user_tweets=json.load(open("src/twibot22_Botrgcn_feature/id_tweet.json",'r'))

pretrained_weights = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(pretrained_weights)
model = T5EncoderModel.from_pretrained(pretrained_weights)

feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer,device=DEVICE,padding=True, truncation=True,max_length=50, add_special_tokens = True)

def tweets_embedding():
        print('Running feature2 embedding')
        path='data/twibot22/T5-tweet/'
        if True:
            tweets_list=[]
            for i in tqdm(range(len(each_user_tweets))):
                if i>600000 and i<=700000:
                    if len(each_user_tweets[str(i)])==0:
                        total_each_person_tweets=torch.zeros(512)
                    else:
                        for j in range(len(each_user_tweets[str(i)])):
                            each_tweet=each_user_tweets[str(i)][j]
                            if each_tweet is None:
                                total_word_tensor=torch.zeros(512)
                            else:
                                each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
                                for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
                                    if k==0:
                                        total_word_tensor=each_word_tensor
                                    else:
                                        total_word_tensor+=each_word_tensor
                                total_word_tensor/=each_tweet_tensor.shape[1]
                            if j==0:
                                total_each_person_tweets=total_word_tensor
                            elif j==20:
                                break
                            else:
                                total_each_person_tweets+=total_word_tensor
                        if (j==20):
                            total_each_person_tweets/=20
                        else:
                            total_each_person_tweets/=len(each_user_tweets[str(i)])
                            
                    tweets_list.append(total_each_person_tweets)
                        
                    if i%1000==0 and i!=0:
                        tweet_tensor=torch.stack(tweets_list)
                        file_name="tweets_tensor"+str(i)+'.pt'
                        torch.save(tweet_tensor,path+file_name)
                        tweets_list=[]
                        
        else:
            tweets_tensor=torch.load(path)
        print('Finished')
        
tweets_embedding()