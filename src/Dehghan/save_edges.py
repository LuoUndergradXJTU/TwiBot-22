
import networkx as nx
import numpy as np
import os

from tqdm import tqdm

# edge_path=r'/data2/whr/lyh/baseline2/Twibot-22/edges.npy'
# edge_index=list(np.load(edge_path))
# edges=[]

dataset_name='Twibot-22'

edge_path=r'/data2/whr/lyh/baseline2/Twibot-22/edges.npy'
edge_index=list(np.load(edge_path))
edges=[]
dataset_name='Twibot-22'
id_include=np.load('/data2/whr/lyh/baseline2/Twibot-22/id_include.npy',allow_pickle=True)
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

id_include=list(id_include.item())
for edge in tqdm(edge_index):
    try:
        edges.append([id_include.index(edge[0]),id_include.index(edge[1])])
    except:
        pass
np.save(dataset_name+'/edges_cleaned.npy',np.array(edges))