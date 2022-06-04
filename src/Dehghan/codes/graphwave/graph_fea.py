import networkx as nx
import numpy as np
import os
from graphwave import *
from tqdm import tqdm

# edge_path=r'/data2/whr/lyh/baseline2/Twibot-22/edges.npy'
# edge_index=list(np.load(edge_path))
# edges=[]

dataset_name='Twibot-22'
edges=np.load('/data2/whr/lyh/baseline2/Twibot-22/edges.npy')

G = nx.Graph()
G.add_edges_from(edges)

#g.remove_edges_from(nx.selfloop_edges(g))
k=int(len(G.nodes)/1000)+1
res=np.zeros((int(len(G.nodes)),12))
all_nodes=list(G.nodes)
for i in tqdm(range(k)):
    if(i>=int(k/100)-1):
        nodes=all_nodes[i*100:]
    else:
        nodes=all_nodes[i*100:(i+1)*100]
    graph=G.subgraph(nodes)
    chi,heat_print, taus=graphwave_alg(graph, [100,200,300], verbose=False)
    print(np.array(chi).shape)
    np.save('/data2/whr/lyh/baseline2/'+dataset_name+'/graph_fea_chi.npy',np.array(chi))
    res[i*100:(i+1)*100]=np.array(chi)
    
    
#chi,heat_print, taus=graphwave_alg(g, [100,200,300], verbose=False)
np.save('/data2/whr/lyh/baseline2/'+dataset_name+'/graph_fea.npy',np.array(res))
print(np.array(res).shape)