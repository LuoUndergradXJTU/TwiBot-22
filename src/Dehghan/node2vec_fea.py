import networkx as nx
from node2vec import Node2Vec
import numpy as np
import os
from tqdm import tqdm

edge_path=r'Twibot-22/edges.npy'
edge_index=list(np.load(edge_path))
edges=edge_index
dataset_name='Twibot-22'
id_include=(np.load(dataset_name+'/id_include.npy',allow_pickle=True))
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

id_include=list(id_include.item())


G = nx.Graph()
G.add_edges_from(edges)


node2vec = Node2Vec(G, p=0.25,q=0.25, walk_length=100, num_walks=18)  
model = node2vec.fit(window=10, min_count=1, batch_words=4)
model.save(dataset_name+'/node2vec.model')
print()