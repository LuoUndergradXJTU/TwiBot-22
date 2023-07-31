import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from dataset_tool import fast_merge,df_to_mask
import os
from transformers import pipeline

print('loading raw data')
node=pd.read_json("../datasets/cresci-2015/node.json")
edge=pd.read_csv("../datasets/cresci-2015/edge.csv")
label=pd.read_csv("../datasets/cresci-2015/label.csv")
split=pd.read_csv("../datasets/cresci-2015/split.csv")
print('processing raw data')
user,tweet=fast_merge(dataset='cresci-2015')
path='processed_data/'

if not os.path.exists("processed_data"):
    os.mkdir("processed_data")

#labels
print('extracting labels and splits')
label_list=[]
for i in range(len(label)):
    if label['label'][i]=='human':
        label_list.append(int(0))
    else:
        label_list.append(int(1))
label_tensor=torch.tensor(label_list,dtype=torch.long)
torch.save(label_tensor,path+'label.pt')

train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]
user_index_to_uid = list(user.id)
tweet_index_to_tid = list(tweet.id)
unique_uid=set(user.id)
uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}
tid_to_tweet_index = {x : i for i, x in enumerate(tweet_index_to_tid)}
train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")
torch.save(train_mask,path+"train_idx.pt")
torch.save(valid_mask,path+"val_idx.pt")
torch.save(test_mask,path+"test_idx.pt")

#graph
print('extracting graph info')
edge_index=[]
edge_type=[]
for i in tqdm(range(len(edge))):
    if edge['relation'][i]=='post':
        continue
    elif edge['relation'][i]=='friend':
        try:
            source_id=uid_to_user_index[edge['source_id'][i]]
            target_id=uid_to_user_index[edge['target_id'][i]]
        except KeyError:
            continue
        else:
            edge_index.append([uid_to_user_index[edge['source_id'][i]],uid_to_user_index[edge['target_id'][i]]])
            edge_type.append(0)
    else:
        try:
            source_id=uid_to_user_index[edge['source_id'][i]]
            target_id=uid_to_user_index[edge['target_id'][i]]
        except KeyError:
            continue
        else:
            edge_index.append([uid_to_user_index[edge['source_id'][i]],uid_to_user_index[edge['target_id'][i]]])
            edge_type.append(1)
torch.save(torch.tensor(edge_index,dtype=torch.long).t(),path+'edge_index.pt')
torch.save(torch.tensor(edge_type,dtype=torch.long),path+'edge_type.pt')

#num_properties
print('extracting num_properties')
following_count=[]
for i,each in enumerate(node['public_metrics']):
    if i==len(user):
        break
    if each is not None and isinstance(each,dict):
        if each['following_count'] is not None:
            following_count.append(each['following_count'])
        else:
            following_count.append(0)
    else:
        following_count.append(0)
        
statues=[]
for i,each in enumerate(node['public_metrics']):
    if i==len(user):
        break
    if each is not None and isinstance(each,dict):
        if each['listed_count'] is not None:
            statues.append(each['listed_count'])
        else:
            statues.append(0)
    else:
        statues.append(0)
        
followers_count=[]
for each in user['public_metrics']:
    if each is not None and each['followers_count'] is not None:
        followers_count.append(int(each['followers_count']))
    else:
        followers_count.append(0)

tweet_count=[]
for each in user['public_metrics']:
    if each is not None and each['tweet_count'] is not None:
        tweet_count.append(int(each['tweet_count']))
    else:
        tweet_count.append(0)
        
screen_name_length=[]
for each in user['username']:
    if each is not None:
        screen_name_length.append(len(each))
    else:
        screen_name_length.append(int(0))
        

created_at=user['created_at']
created_at=pd.to_datetime(created_at,unit='s')


df_followers_count=pd.DataFrame(followers_count)
followers_count=(df_followers_count-df_followers_count.mean())/df_followers_count.std()
followers_count=torch.tensor(np.array(followers_count),dtype=torch.float32)

date0=dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
active_days=[]
for each in created_at:
    active_days.append((date0-each).days)
    
active_days=pd.DataFrame(active_days)
active_days.fillna(int(0))
active_days=active_days.fillna(int(0)).astype(np.float32)
active_days=(active_days-active_days.mean())/active_days.std()
active_days=torch.tensor(np.array(active_days),dtype=torch.float32)

screen_name_length=pd.DataFrame(screen_name_length)
screen_name_length=(screen_name_length-screen_name_length.mean())/screen_name_length.std()
screen_name_length=torch.tensor(np.array(screen_name_length),dtype=torch.float32)

following_count=pd.DataFrame(following_count)
following_count=(following_count-following_count.mean())/following_count.std()
following_count=torch.tensor(np.array(following_count),dtype=torch.float32)

tweet_count=pd.DataFrame(tweet_count)
tweet_count=(tweet_count-tweet_count.mean())/tweet_count.std()
tweet_count=torch.tensor(np.array(tweet_count),dtype=torch.float32)

statues=pd.DataFrame(statues)
statues=(statues-statues.mean())/statues.std()
statues=torch.tensor(np.array(statues),dtype=torch.float32)

num_properties_tensor=torch.cat([followers_count,active_days,screen_name_length,following_count,statues, tweet_count],dim=1)
torch.save(num_properties_tensor,path+'num_properties_tensor.pt')

#cat_properties
print('extracting cat_properties')
default_profile_image=[]
for each in user['profile_image_url']:
    if each is not None:
        if each=='http://a0.twimg.com/sticky/default_profile_images/default_profile_0_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://a0.twimg.com/sticky/default_profile_images/default_profile_1_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://a0.twimg.com/sticky/default_profile_images/default_profile_2_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://a0.twimg.com/sticky/default_profile_images/default_profile_3_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://a0.twimg.com/sticky/default_profile_images/default_profile_4_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://a0.twimg.com/sticky/default_profile_images/default_profile_5_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://a0.twimg.com/sticky/default_profile_images/default_profile_6_normal.png':
            default_profile_image.append(int(1))
        else:
            default_profile_image.append(int(0))
    else:
        default_profile_image.append(int(1))
default_profile_image_tensor=torch.tensor(default_profile_image,dtype=torch.float)
default_profile_image_tensor=default_profile_image_tensor.reshape([5301,1])

location_one_hot = []
for location in user['locations']:
    one_hot_vector = [0] * len(unique_locations)
    if location is not None:
        location_index = location_to_index[location]
        one_hot_vector[location_index] = 1
    location_one_hot.append(one_hot_vector)
location_tensor = torch.tensor(location_one_hot, dtype=torch.float32)

cat_properties_tensor = torch.cat([default_profile_image_tensor, location_tensor], dim=1)
    
torch.save(cat_properties_tensor,path+'cat_properties_tensor.pt')

#get each_user_tweets
user_index_to_uid = list(user.id)
tweet_index_to_tid = list(tweet.id)
        
uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}
tid_to_tweet_index = {x : i for i, x in enumerate(tweet_index_to_tid)}

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
for i in tqdm(range(len(edge))):
    dict[edge.iloc[i]['source_id']].append(edge.iloc[i]['target_id'])


# Preprocess_2.py
print("entering preprocess_2.py")


each_user_tweets = dict

user,tweet=fast_merge(dataset="cresci-2015")

user_text=list(user['description'])
tweet_text = [text for text in tweet.text]
# each_user_tweets=torch.load('./processed_data/each_user_tweets.npy')
# feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=3,padding=True, truncation=True,max_length=50, add_special_tokens = True)

# from transformers import RobertaTokenizer, RobertaModel
# tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
# model = RobertaModel.from_pretrained('roberta-base')
# feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer,device=0)

# from transformers import BertTokenizer, BertModel
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
# feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer)

# from transformers import BartTokenizer, BartModel
# tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
# model = BartModel.from_pretrained('facebook/bart-base')
# feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer)

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
model = AutoModel.from_pretrained('intfloat/e5-base-v2')
feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer)

# from transformers import DebertaTokenizer, DebertaModel
# tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
# model = DebertaModel.from_pretrained('microsoft/deberta-base')
# feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer)

def Des_embbeding():
        print('Running feature1 embedding')
        path="./processed_data/des_tensor.pt"
        if not os.path.exists(path):
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if each is None:
                    des_vec.append(torch.zeros(768))
                else:
                    feature=torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor=tensor
                        else:
                            feature_tensor+=tensor
                    feature_tensor/=feature.shape[1]
                    des_vec.append(feature_tensor)
                    
            des_tensor=torch.stack(des_vec,0)
            torch.save(des_tensor,path)
        else:
            des_tensor=torch.load(path)
        print('Finished')
        return des_tensor

def tweets_embedding():
        print('Running feature2 embedding')
        path="./processed_data/tweets_tensor.pt"
        if not os.path.exists(path):
            tweets_list=[]
            for i in tqdm(range(len(each_user_tweets))):
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
                        
            tweet_tensor=torch.stack(tweets_list)
            torch.save(tweet_tensor,"./processed_data/tweets_tensor.pt")
            
        else:
            tweets_tensor=torch.load(path)
        print('Finished')

Des_embbeding()
tweets_embedding()
