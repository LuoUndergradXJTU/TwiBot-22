import torch
from tqdm import tqdm
from dataset import hetero_graph_vectorize,fast_merge
import numpy as np
from transformers import pipeline
import os

user, tweet = fast_merge("cresci-2017", "209")

tweet_text = [text for text in tweet.text]

each_user_tweets=np.load('./each_user_tweets.npy',allow_pickle=True)

feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=1,padding=True, truncation=True,max_length=50, add_special_tokens = True)

def tweets_embedding():
        print('Running feature2 embedding')
        path="./data_17/tweets_tensor.pt"
        if not os.path.exists(path):
            tweets_list=[]
            for i in tqdm(range(len(each_user_tweets))):
                if i>6000 and i<=9000:
                    if len(each_user_tweets[i])==0:
                        total_each_person_tweets=torch.zeros(768)
                    else:
                        for j in range(len(each_user_tweets[i])):
                            each_tweet=tweet_text[each_user_tweets[i][j]]
                            if each_tweet is None:
                                total_word_tensor=torch.zeros(768)
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
                            total_each_person_tweets/=len(each_user_tweets[i])
                            
                    tweets_list.append(total_each_person_tweets)
                    if i%1000==0 and i!=0:
                        tweet_tensor=torch.stack(tweets_list)
                        file_name="tweets_tensor"+str(i)+'.pt'
                        torch.save(tweet_tensor,"./data_17/"+file_name)
                        tweets_list=[]
            
        else:
            tweets_tensor=torch.load(path)
        print('Finished')
        
tweets_embedding()