from graphrole import RecursiveFeatureExtractor, RoleExtractor
import networkx as nx
from karateclub import Role2Vec
import numpy as np
import os
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
#graph=G.subgraph(list(G.nodes)[:100])
feature_extractor = RecursiveFeatureExtractor(G)
features = feature_extractor.extract_features()
role_extractor = RoleExtractor(n_roles=None)
role_extractor.extract_role_factors(features)
emb=[]
for key in list(role_extractor.roles().keys()):
    k_emb=eval(role_extractor.roles()[key].split('_')[-1])
    emb.append()

emb=role_extractor.roles()
np.save(dataset_name+'/roix_22.npy',np.array(emb))

 