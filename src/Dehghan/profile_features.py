import json
import numpy as np
import os
from tqdm import tqdm
dataset_name='Twibot-22'
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

edge_path=r'Twibot-22/edges.npy'
edge_index=list(np.load(edge_path))
edges=[]
dataset_name='Twibot-22'
id_include=(np.load(dataset_name+'/id_include.npy',allow_pickle=True))
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

id_include=list(id_include.item())

id_list=np.load('/data2/whr/lyh/twibot22_baseline/'+dataset_name+'/id.npy')
id_list=list(id_list)
def profile(data):
    fea=[]
    for user in tqdm(data):
        try:
            protected=(user['profile']['protected'])
        except:
            protected=0
            
        try:
            followers_count=(user['profile']['followers_count'])
        except:
            followers_count=0
        
        try:
            friends_count=(user['profile']['friends_count'])
        except:
            friends_count=0
        try:
            listed_count=(user['profile']['listed_count'])
        except:
            listed_count=0
        try:
            favourites_count=(user['profile']['favourites_count'])
        except:
            favourites_count=0
        try:
            #print(user['profile']['statuses_count'])
            statuses_count=(user['profile']['statuses_count'])
        except:
            statuses_count=0
        try:
            geo_enabled=(user['profile']['geo_enabled'])
        except:
            geo_enabled=0
        try:
            default_profile=(user['profile']['default_profile'])
        except:
            default_profile=0
        try:
            default_profile_image=(user['profile']['default_profile_image'])
        except:
            default_profile_image=0
        try:
            business='Business' in user['domain']
            entertainment='Entertainment' in user['domain']
            politics='Politics' in user['domain']
            sports='Sports' in user['domain']
        except:
            business=0
            entertainment=0
            politics=0
            sports=0
        try:
            verified=(user['profile']['verified'])
        except:
            verified=0
        fea.append([protected,followers_count,friends_count,listed_count,favourites_count,statuses_count,geo_enabled,default_profile,default_profile_image,business,entertainment,politics,sports,verified])
        #print()
    return fea
        
    
if __name__ == '__main__':
    files=['train','val','test']
    #files=['node']
    data=[]
    for file in files:
        #name='/data2/whr/lyh/project3/Twibot-20/'+file +'.json'
        name='/data2/whr/lyh/twibot22_baseline/'+dataset_name+'/'+file+'.json'
        f=open(name)
        users=json.load(f)
        print('{} {}'.format(name,len(users)))

        data=data+users
    
    
    profile_fea=profile(data)
    profile_fea=np.array(profile_fea)
    np.save(dataset_name+'/'+'profile.npy',profile_fea)
    print()