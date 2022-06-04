#!/usr/bin/env python
# coding: utf-8

# In[88]:


import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, HeteroData
from datetime import datetime as dt
from dataset import fast_merge,df_to_mask
from tqdm import tqdm


# In[89]:


node=pd.read_json("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/node.json")
edge=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/edge.csv")
label=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/label.csv")
split=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/split.csv")


# In[135]:


node


# In[132]:


edge=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/edge.csv")


# In[134]:


edge['relation'].value_counts()


# In[90]:


user,tweet=fast_merge(dataset="cresci-2017")


# In[91]:


train_uid_with_label = user[user.split == "train"][["id", "split", "label"]]
valid_uid_with_label = user[user.split == "val"][["id", "split", "label"]]
test_uid_with_label = user[user.split == "test"][["id", "split", "label"]]


# In[92]:


user_index_to_uid = list(user.id)
tweet_index_to_tid = list(tweet.id)
unique_uid=set(user.id)

uid_to_user_index = {x : i for i, x in enumerate(user_index_to_uid)}
tid_to_tweet_index = {x : i for i, x in enumerate(tweet_index_to_tid)}


# In[93]:


train_mask = df_to_mask(train_uid_with_label, uid_to_user_index, "train")
valid_mask = df_to_mask(valid_uid_with_label, uid_to_user_index, "val")
test_mask = df_to_mask(test_uid_with_label, uid_to_user_index, "test")


# In[96]:


torch.save(train_mask,"./data_17/train_idx.pt")
torch.save(valid_mask,"./data_17/val_idx.pt")
torch.save(test_mask,"./data_17/test_idx.pt")


# In[125]:


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


# In[131]:


edge['relation'].value_counts()


# In[98]:


torch.save(torch.tensor(edge_index,dtype=torch.long).t(),'./data_17/edge_index.pt')
torch.save(torch.tensor(edge_type,dtype=torch.long),'./data_17/edge_type.pt')


# In[99]:


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


# In[100]:


tweet_count=[]
for i,each in enumerate(node['public_metrics']):
    if i==len(user):
        break
    if each is not None and isinstance(each,dict):
        if each['tweet_count'] is not None:
            tweet_count.append(each['tweet_count'])
        else:
            tweet_count.append(0)
    else:
        tweet_count.append(0)


# In[101]:


followers_count=[]
for each in user['public_metrics']:
    if each is not None and each['followers_count'] is not None:
        followers_count.append(int(each['followers_count']))
    else:
        followers_count.append(0)


# In[102]:


screen_name_length=[]
for each in user['username']:
    if each is not None:
        screen_name_length.append(len(each))
    else:
        screen_name_length.append(int(0))


# In[103]:


created_at=user['created_at']


# In[104]:


active_days=[]
date0=dt.strptime('Tue Sep 1 00:00:00 +0000 2020','%a %b %d %X %z %Y')
for each in created_at:
    try:
        date=dt.strptime(each,'%a %b %d %X %z %Y')
    except ValueError:
        date=date0
    active_days.append((date0-date).days)


# In[105]:


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
        elif each=='http://abs.twimg.com/sticky/default_profile_images/default_profile_0_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://abs.twimg.com/sticky/default_profile_images/default_profile_1_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://abs.twimg.com/sticky/default_profile_images/default_profile_2_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://abs.twimg.com/sticky/default_profile_images/default_profile_3_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://abs.twimg.com/sticky/default_profile_images/default_profile_4_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://abs.twimg.com/sticky/default_profile_images/default_profile_5_normal.png':
            default_profile_image.append(int(1))
        elif each=='http://abs.twimg.com/sticky/default_profile_images/default_profile_6_normal.png':
            default_profile_image.append(int(1))
        else:
            default_profile_image.append(int(0))
    else:
        default_profile_image.append(int(1))


# In[106]:


protected=user['protected']
protected.value_counts()


# In[107]:


verified=user['verified']
verified.value_counts()


# In[108]:


df_followers_count=pd.DataFrame(followers_count)
followers_count=(df_followers_count-df_followers_count.mean())/df_followers_count.std()
followers_count=torch.tensor(np.array(followers_count),dtype=torch.float32)


# In[109]:


active_days=pd.DataFrame(active_days)
active_days=(active_days-active_days.mean())/active_days.std()
active_days=torch.tensor(np.array(active_days),dtype=torch.float32)


# In[110]:


screen_name_length=pd.DataFrame(screen_name_length)
screen_name_length=(screen_name_length-screen_name_length.mean())/screen_name_length.std()
screen_name_length=torch.tensor(np.array(screen_name_length),dtype=torch.float32)


# In[111]:


following_count=pd.DataFrame(following_count)
following_count=(following_count-following_count.mean())/following_count.std()
following_count=torch.tensor(np.array(following_count),dtype=torch.float32)


# In[112]:


tweet_count=pd.DataFrame(tweet_count)
tweet_count=(tweet_count-tweet_count.mean())/tweet_count.std()
tweet_count=torch.tensor(np.array(tweet_count),dtype=torch.float32)


# In[113]:


num_properties_tensor=torch.cat([followers_count,active_days,screen_name_length,following_count,tweet_count],dim=1)


# In[114]:


torch.save(num_properties_tensor,'./data_17/num_properties_tensor.pt')


# In[115]:


protected_list=[]
for each in protected:
    if each == 1.0:
        protected_list.append(1)
    else:
        protected_list.append(0)


# In[116]:


verified_list=[]
for each in verified:
    if each == 1.0:
        verified_list.append(1)
    else:
        verified_list.append(0)


# In[117]:


protected_tensor=torch.tensor(protected_list,dtype=torch.float32)
verified_tensor=torch.tensor(verified_list,dtype=torch.float32)
default_profile_image_tensor=torch.tensor(default_profile_image,dtype=torch.float32)


# In[118]:


cat_properties_tensor=torch.cat([protected_tensor.reshape([14368,1]),verified_tensor.reshape([14368,1]),default_profile_image_tensor.reshape([14368,1])],dim=1)


# In[119]:


torch.save(cat_properties_tensor,'./data_17/cat_properties_tensor.pt')


# In[120]:


label_list=[]
for i in range(len(label)):
    if label['label'][i]=='human':
        label_list.append(int(0))
    else:
        label_list.append(int(1))


# In[121]:


label_tensor=torch.tensor(label_list,dtype=torch.long)
torch.save(label_tensor,'./data_17/label.pt')


# In[122]:


label


# In[ ]:




