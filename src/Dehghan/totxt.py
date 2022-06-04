
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import networkx as nx

dataset_name='Twibot-22'
edge_path=r'Twibot-22/edges.npy'
edge_index=list(np.load(edge_path))
edges=edge_index
dataset_name='Twibot-22'
id_include=(np.load(dataset_name+'/id_include.npy',allow_pickle=True))
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

id_include=list(id_include.item())
# for edge in tqdm(edge_index):
#     try:
#         edges.append([id_include.index(edge[0]),id_include.index(edge[1])])
#     except:
#         pass

# graph=nx.Graph()
# graph.add_edges_from(edges)
# g=nx.Graph(graph)
# g.remove_edges_from(nx.selfloop_edges(g))
# #nodes=list(g.nodes())[:5000]
# nodes=list(g.nodes())
# edges=g.subgraph(nodes).edges()



with open (dataset_name+'/edges.txt','w') as f:
    for edge in tqdm(edges):
        f.write(f'{edge[0]} {edge[1]}\n')