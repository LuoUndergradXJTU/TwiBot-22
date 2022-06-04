import torch
from tqdm import tqdm
import numpy as np
from transformers import pipeline, T5Tokenizer, T5EncoderModel
import os
import pandas as pd
path='/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-22/'

user=pd.read_json(path+'user.json')

pretrained_weights = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(pretrained_weights)
model = T5EncoderModel.from_pretrained(pretrained_weights)

feature_extract=pipeline('feature-extraction',model=model,tokenizer=tokenizer,device=5,padding=True, truncation=True,max_length=50, add_special_tokens = True)

user_text=list(user['description'])

def Des_embbeding():
        print('Running feature1 embedding')
        if True:
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if k>500000:
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
                        
                    if k%10000==0 and k!=0:
                        des_tensor=torch.stack(des_vec,0)
                        file_name="des_tensor"+str(k)+'.pt'
                        torch.save(des_tensor,"/data2/whr/czl/TwiBot22-baselines/data/twibot22/T5-tweet"+file_name)
                        des_vec=[]
                        
            des_tensor=torch.stack(des_vec,0)
            file_name="des_tensor"+str(k)+'.pt'
            torch.save(des_tensor,"/data2/whr/czl/TwiBot22-baselines/data/twibot22/T5-tweet"+file_name)
        else:
            des_tensor=torch.load(path)
        print('Finished')
        return des_tensor
    
Des_embbeding()