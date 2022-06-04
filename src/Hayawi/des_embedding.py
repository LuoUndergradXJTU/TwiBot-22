import numpy as np
from gensim.parsing.preprocessing import preprocess_string,strip_non_alphanum,preprocess_documents
from tqdm import tqdm
from gensim.models import KeyedVectors
import torch
from torch.nn.utils.rnn import pad_sequence
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='name of the dataset used for training')
args = parser.parse_args()

model=KeyedVectors.load('glove.6B.100d.kv')

path='./'+args.dataset+'/'
des=list(np.load(path+'des.npy',allow_pickle=True))
doc=[]
for each_des in tqdm(des):
    each_des=strip_non_alphanum(each_des)
    doc.append(preprocess_string(each_des))
    
des_vec=[]
for each in tqdm(doc):
    each_vec=[]
    if len(each)==0:
        each_vec.append(model['missing'])
    else:
        for word in each:
            try:
                each_vec.append(model[word])
            except KeyError:
                each_vec.append(np.zeros(50))
                continue
    each_vec=torch.tensor(each_vec,dtype=torch.float32)
    #each_vec=list(each_vec)
    des_vec.append(each_vec)
    
des_vec=pad_sequence(des_vec)
des_tensor=des_vec.clone().detach().requires_grad_(False)
torch.save(des_tensor,path+'des_tensor.pt')