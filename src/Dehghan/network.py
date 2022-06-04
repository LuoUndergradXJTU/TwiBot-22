import networkx as nx
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
def node_fea(graph):
    degree_cen_dict=nx.degree_centrality(graph) 

    eigen_c_dict=nx.eigenvector_centrality(graph,max_iter=100)
 
    #eigen_c_dict=None
    closeness_dict=nx.closeness_centrality(graph)
    #closeness_dict=None
    harmonic_dict=nx.harmonic_centrality(graph)
    between_dict=nx.betweenness_centrality(graph)
    #auth_dict,hub_dict=nx.hits(graph)
    auth_dict=None
    hub_dict=None
    cons_dict=nx.constraint(graph)
    g=nx.Graph(graph)
    g.remove_edges_from(nx.selfloop_edges(g))
    core=nx.core_number(g)
    #ecc=nx.eccentricity(graph)
    ecc=None
    p_rank=nx.pagerank(graph)
    d_c=[]
    e_c=[]
    cl_c=[]
    har=[]
    b=[]
    aut=[]
    hub=[]
    cons=[]
    cor=[]
    e=[]
    p=[]
    #8278+2365+1183
    for i ,id_num in tqdm(enumerate(list(id_include))):
        d_c.append(degree_cen_dict[i])
        try:
            
            e_c.append(eigen_c_dict[i])
        except:
            e_c.append(-1)
        try:
            
            cl_c.append(closeness_dict[i])
        except:
            cl_c.append(-1)
        #cl_c.append(closeness_dict[i])
        har.append(harmonic_dict[i])
        b.append(between_dict[i])
        try:
            aut.append(auth_dict[i])
        except:
            aut.append(-1)
        try:
            hub.append(hub_dict[i])
        except:
            hub.append(-1) 
        cons.append(cons_dict[i])
        cor.append(core[i])
        try:
            e.append(ecc[i])
        except:
            e.append(-1)
        p.append(p_rank[i])
    return [d_c,e_c,cl_c,har,b,aut,hub,cons,cor,e,p]
    #strength
    #eigen
   
fea=node_fea(G)  
np.save(dataset_name+'/'+'node_fea.npy',np.array(fea))
print(dataset_name+'/'+'node_fea.npy'+' saved!')