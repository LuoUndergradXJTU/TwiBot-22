
import torch
import numpy as np
import pandas as pd
from datetime import datetime as dt
from dataset import fast_merge,df_to_mask
from tqdm import tqdm
dataset_names = [
    'botometer-feedback-2019', 'botwiki-2019', 'celebrity-2019', 'cresci-2015', 'cresci-2017', 'cresci-rtbust-2019', 'cresci-stock-2018', 'gilani-2017', 'midterm-2018', 'political-bots-2019', 'pronbots-2019', 'vendor-purchased-2019', 'verified-2019', "Twibot-20"
]
path='./cresci-19/'
file='cresci-rtbust-2019'
from utils import entropy,bigrams_freq




node=pd.read_json("/data2/whr/czl/TwiBot22-baselines/datasets/"+file+"/node.json")




label=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/"+file+"/label.csv")
split=pd.read_csv("/data2/whr/czl/TwiBot22-baselines/datasets/"+file+"/split.csv")




user,tweet=fast_merge(dataset=file)




split=split[split.id.isin(list(user.id))]
label=label[label.id.isin(list(user.id))]




split.index=list(range(len(user)))
label.index=list(range(len(user)))




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
        if each['tweet_count'] is not None:
            statues.append(each['tweet_count'])
        else:
            statues.append(0)
    else:
        statues.append(0)




listed_count=[]
for i,each in enumerate(node['public_metrics']):
    if i==len(user):
        break
    if each is not None and isinstance(each,dict):
        if each['tweet_count'] is not None:
            listed_count.append(each['listed_count'])
        else:
            listed_count.append(0)
    else:
        listed_count.append(0)




followers_count=[]
for each in user['public_metrics']:
    if each is not None and each['followers_count'] is not None:
        followers_count.append(int(each['followers_count']))
    else:
        followers_count.append(0)




num_username=[]
for each in user['username']:
    if each is not None:
        num_username.append(len(each))
    else:
        num_username.append(int(0))




created_at=user['created_at']
created_at=pd.to_datetime(created_at,unit='s')




followers_count=pd.DataFrame(followers_count)
followers_count=(followers_count-followers_count.mean())/followers_count.std()
followers_count=torch.tensor(np.array(followers_count),dtype=torch.float32)




date0=dt.strptime('Tue Sep 1 00:00:00 +0000 2020 ','%a %b %d %X %z %Y ')
active_days=[]
for each in created_at:
    active_days.append((date0-each).days)




active_days=pd.DataFrame(active_days)
active_days=active_days.fillna(int(1)).astype(np.float32)




listed_count=pd.DataFrame(listed_count)




tweet_freq=listed_count/active_days
tweet_freq=(tweet_freq-tweet_freq.mean())/tweet_freq.std()
tweet_freq=torch.tensor(np.array(tweet_freq),dtype=torch.float32)




listed_count=(listed_count-listed_count.mean())/listed_count.std()
listed_count=torch.tensor(np.array(listed_count),dtype=torch.float32)




active_days=(active_days-active_days.mean())/active_days.std()
active_days=torch.tensor(np.array(active_days),dtype=torch.float32)




num_username=pd.DataFrame(num_username)
num_username=(num_username-num_username.mean())/num_username.std()
num_username=torch.tensor(np.array(num_username),dtype=torch.float32)




following_count=pd.DataFrame(following_count)
following_count=(following_count-following_count.mean())/following_count.std()
following_count=torch.tensor(np.array(following_count),dtype=torch.float32)




statues=pd.DataFrame(statues)
statues=(statues-statues.mean())/statues.std()
statues=torch.tensor(np.array(statues),dtype=torch.float32)




screen_name_freq=[]
for each in user['name']:
    if each is None or len(each)<=1:
        screen_name_freq.append(0)
    else:
        screen_name_freq.append(bigrams_freq(each))
screen_name_freq=pd.DataFrame(screen_name_freq)
screen_name_freq=(screen_name_freq-screen_name_freq.mean())/screen_name_freq.std()
screen_name_freq=torch.tensor(np.array(screen_name_freq),dtype=torch.float32)
name_entropy=[]
for each in user['name']:
    if each is None or len(each)==0:
        name_entropy.append(0)
    else:
        name_entropy.append(entropy(each))
name_entropy=pd.DataFrame(name_entropy)
name_entropy=(name_entropy-name_entropy.mean())/name_entropy.std()
name_entropy=torch.tensor(np.array(name_entropy),dtype=torch.float32)
des_entropy=[]
for each in user['description']:
    if each is None or len(each)==0:
        des_entropy.append(0)
    else:
        des_entropy.append(entropy(each))
des_entropy=pd.DataFrame(des_entropy)
des_entropy=(des_entropy-des_entropy.mean())/des_entropy.std()
des_entropy=torch.tensor(np.array(des_entropy),dtype=torch.float32)




num_properties_tensor=torch.cat([followers_count,tweet_freq,num_username,following_count,statues,listed_count,screen_name_freq,name_entropy,des_entropy],dim=1)




torch.save(num_properties_tensor,path+'num_properties_tensor.pt')




label_list=[]
for i in range(len(label)):
    if label['label'][i]=='human':
        label_list.append(int(0))
    else:
        label_list.append(int(1))




label_tensor=torch.tensor(label_list,dtype=torch.long)
torch.save(label_tensor,path+'label.pt')




user['description']=user['description'].replace('', 'missing')
user['description']=user['description'].fillna('missing')




np.save(path+'des.npy',list(user['description']))

