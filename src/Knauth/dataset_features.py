from generate_features import account
import json
import numpy as np
dataset_name='Twibot-22'
#dataset_path='/data2/whr/czl/TwiBot22-baselines/datasets'+'/'+dataset_name
files=['train','test']
for file in files:
    with open(dataset_name+'/' + file + '.json','r') as f:
        data=json.load(f)
        acc_mat=account(data)
        lev_matr=acc_mat[:,-1]
        np.save(dataset_name+'/' + file+'_ac.npy',acc_mat)
        np.save(dataset_name+'/' + file+'_lev.npy',lev_matr)