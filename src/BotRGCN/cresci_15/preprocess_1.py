import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime as dt
from dataset_tool import fast_merge,df_to_mask
import os

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

statues=pd.DataFrame(statues)
statues=(statues-statues.mean())/statues.std()
statues=torch.tensor(np.array(statues),dtype=torch.float32)

num_properties_tensor=torch.cat([followers_count,active_days,screen_name_length,following_count,statues],dim=1)
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

cat_properties_tensor=default_profile_image_tensor.reshape([5301,1])
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

np.save('processed_data/each_user_tweets.npy',dict)