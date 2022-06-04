import networkx as nx
from karateclub import Role2Vec
import numpy as np
import os
import numpy
from tqdm import tqdm
edge_path=r'Twibot-22/edges.npy'
edge_index=list(np.load(edge_path))
edges=[]
dataset_name='Twibot-22'
id_include=(np.load(dataset_name+'/id_include.npy',allow_pickle=True))
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

id_include=list(id_include.item())
for edge in tqdm(edge_index):
    try:
        edges.append([id_include.index(edge[0]),id_include.index(edge[1])])
    except:
        pass

G = nx.Graph()
G.add_edges_from(edges)


model=Role2Vec(seed=42)
model.fit(G)
np.save(dataset_name+'/role2vec_fea.npy',np.array(model.get_embedding()))
#print(model.get_embedding())
