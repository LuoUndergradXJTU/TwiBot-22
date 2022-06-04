import json
import csv
import os
import string
import numpy as np
from tqdm import tqdm
dataset_name='Twibot-22'
dataset_path='/data2/whr/czl/TwiBot22-baselines/datasets'+'/'+dataset_name
file0='user.json'
file1='label.csv'
file3='split.csv'

if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)
with open(dataset_path +'/'+ file0,'r') as f:
    nodes=json.load(f)
    print()
    
# clean nodes
for user in tqdm(nodes):
    user['profile']={}
    user['profile']['default_profile']=0
    user['profile']['geo_enabled']=0
    try:
        user['profile']['protected']=int(user['protected'])
    except:
        user['profile']['protected']=0
    try:
        user['profile']['verified']=int(user['verified'])
    except:
        user['profile']['verified']=0
    try:
        user['profile']['friends_count']=user['public_metrics']['following_count']
        user['profile']['followers_count']=user['public_metrics']['followers_count']
        user['profile']['favourites_count']=0
        user['profile']['listed_count']=user['public_metrics']['listed_count']
        user['profile']['statuses_count']=user['public_metrics']['tweet_count']
    except:
        user['profile']['friends_count']=0
        user['profile']['followers_count']=0
        user['profile']['favourites_count']=0
        user['profile']['listed_count']=0
        user['profile']['statuses_count']=0
    user['profile']['profile_use_background_image']=0
    try:
        
        user['profile']['screen_name']=user['name']
        isinstance(user['profile']['screen_name'], str)
    except:
        user['profile']['screen_name']=''
    try:
        user['profile']['name']=user['username']
        isinstance(user['username'], str)
    except:
        user['profile']['name']=''
    #print()


# load id and label
id_list=[]
label_list=[]
with open(dataset_path +'/'+ file1,'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        id,label =row[0].split(',')
        if id[0] == 'i':
            continue
        id_list.append(eval(id[1:]))
        if(label[0]=='h'):
            label=0
        else:
            label=1
        label_list.append(label)
np.save(dataset_name+'/'+'id.npy',np.array(id_list))
# np.save(dataset_name+'/'+'label.npy',np.array(label_list))


# train test split
train_nodes=[]
val_nodes=[]
test_nodes=[]
label_train=[]
label_val=[]
label_test=[]
#id_include=list(np.load('/data2/whr/lyh/baseline2/Twibot-22/id_include.npy'))
with open(dataset_path +'/'+ file3,'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in tqdm(spamreader):
        id,split=row[0].split(',')
        if id[0] == 'i':
            continue
        id=eval(id[1:])
        index=id_list.index(id)
        # if(not index in id_include):
        #      continue
        try:
            if(split == 'train'):
                train_nodes.append(nodes[index])
                label_train.append(label_list[index])
                
            elif(split == 'valid'):
                val_nodes.append(nodes[index])
                label_val.append(label_list[index])
            else:
                test_nodes.append(nodes[index])
                label_test.append(label_list[index])
        except:
            pass

# with open(dataset_name+'/'+'train_new.json','w') as f:
#     json.dump(train_nodes,f)
# with open(dataset_name+'/'+'val_new.json','w') as f:
#     json.dump(val_nodes,f)
# with open(dataset_name+'/'+'test_new.json','w') as f:
#     json.dump(test_nodes,f)
            
# np.save(dataset_name+'/'+'label_train_new.npy',np.array(label_train))
# np.save(dataset_name+'/'+'label_val_new.npy',np.array(label_val))
# np.save(dataset_name+'/'+'label_test_new.npy',np.array(label_test))



with open(dataset_name+'/'+'train.json','w') as f:
    json.dump(train_nodes,f)
with open(dataset_name+'/'+'val.json','w') as f:
    json.dump(val_nodes,f)
with open(dataset_name+'/'+'test.json','w') as f:
    json.dump(test_nodes,f)
            
np.save(dataset_name+'/'+'label_train.npy',np.array(label_train))
np.save(dataset_name+'/'+'label_val.npy',np.array(label_val))
np.save(dataset_name+'/'+'label_test.npy',np.array(label_test))
