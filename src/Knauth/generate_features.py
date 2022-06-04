from email.policy import default
import json
from tqdm import tqdm
import numpy as np

import os


import numpy as np

dataset_name='Twibot-22'
dataset_path=dataset_name+'/'

if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

def Lev_distance(A,B):
    #A = "fafasa"
    #B = "faftreassa"

    dp = np.array(np.arange(len(B)+1))

    for i in range(1, len(A)+1):
        temp1 = dp[0]
        dp[0] += 1
        for j in range(1, len(B)+1):
            temp2 = dp[j]
            if A[i-1] == B[j-1]:
                dp[j] = temp1
            else:
                dp[j] = min(temp1, min(dp[j-1], dp[j]))+1
            temp1 = temp2

    return dp[len(B)]



'''
train 8278
dev 2365
test 1183
support  217754
'''

def get_id(data):
    id_list=[]
    for users in data:
        id_list.append(eval(users['ID']))
    np.save('id.npy',np.array(id_list))
    

def get_gt(data,dataset):
    gt_list=[]
    for users in data[:8278+2365+1183]:
        gt_list.append(eval(users['label']))
    np.save(dataset+'/'+'label.npy',np.array(gt_list))

def get_num_digits(a):
    num=0
    for i in a:
        if i.isdigit():
            num=num+1
    return num



def tweet_behav(data):
    pass

def tweet_cont(data):
    pass

def get_uni(word):
    classes=[]
    uni_class=np.load('uni_class.npy')
    for i in word:
        uni=0
        for j,k in enumerate(uni_class):
            if(k>ord(i)):
                uni=j-1
                break
        classes.append(uni)
    try:
        return max(classes)
    except:
        return 0

def account(data):
    user_profile=[]
    user_name=[]
    for user in tqdm(data):
        user_pro_temp=[]
        user_pro_temp.append(user['profile']['default_profile'])
        user_pro_temp.append(user['profile']['geo_enabled'])
        user_pro_temp.append(user['profile']['protected'])
        user_pro_temp.append(user['profile']['verified'])
        #user_pro_temp.append('False')
        user_pro_temp.append(user['profile']['friends_count'])
        user_pro_temp.append(user['profile']['followers_count'])
        user_pro_temp.append(user['profile']['favourites_count'])
        user_pro_temp.append(user['profile']['listed_count'])
        user_pro_temp.append(user['profile']['statuses_count'])
        user_pro_temp.append(user['profile']['profile_use_background_image'])
        try:
            user_pro_temp=[int(eval(x)) for x in user_pro_temp]
        except:
            pass
        user_profile.append(user_pro_temp)
        #profile name
        try:
            #screen_name_length=len(user['profile']['screen_name'].rstrip())
            screen_name_length=len(user['name'].rstrip())
            user['profile']['screen_name']=user['name']
        except:
            print(user['profile']['screen_name'])
            screen_name_length=0
        try:
            #user_name_length=len(user['profile']['name'].rstrip())
            user_name_length=len(user['username'].rstrip())
            user['profile']['name']=user['username']
        except:
            user_name_length=0
        screen_name_digits=get_num_digits(user['profile']['screen_name'].rstrip())
        #user naem unicode group
        user_uni=get_uni(user['profile']['name'].rstrip())
        # screen name unicode group
        screen_uni=get_uni(user['profile']['screen_name'].rstrip())
        lev=Lev_distance(user['profile']['name'],user['profile']['screen_name'])
        user_name.append([screen_name_length,user_name_length,screen_name_digits,user_uni,screen_uni,lev])
    return np.concatenate((np.array(user_profile), np.array(user_name)),1)
        
#/data2/whr/lyh/project3/Twibot-20
if __name__ == '__main__':
    files=['train','val','test']
    #files=['user']
    data=[]
    for file in files:
        #/data2/whr/lyh/twibot22_baseline/Twibot-2
        name='/data2/whr/lyh/twibot22_baseline/Twibot-22/'+file +'.json'
        #name='/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/'+file+'.json'
        f=open(name)
        users=json.load(f)
        print('{} {}'.format(name,len(users)))
        data=data+users
    
    #get_gt(data,dataset_name)
    ac_matr=account(data)
    lev_matr=ac_matr[:,-1]
    np.save(dataset_name+'/'+'ac.npy',ac_matr)
    np.save(dataset_name+'/'+'lev.npy',lev_matr)
    print()