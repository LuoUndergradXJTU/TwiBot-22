import networkx as nx
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
from gensim.models import Word2Vec
dataset_name='Twibot-22'
edge_path=dataset_name+'/edges.npy'
edge_index=list(np.load(edge_path))
edges=[]
#id_list=np.load('/data2/whr/lyh/twibot22_baseline/'+dataset_name+'/id.npy')
id_include=(np.load(dataset_name+'/id_include.npy',allow_pickle=True))
if not os.path.exists(dataset_name):
    os.mkdir(dataset_name)

# id_include=list(id_include.item())
# for edge in tqdm(edge_index):
#     try:
#         edges.append([str(id_include.index(edge[0])),str(id_include.index(edge[1]))])
#     except:
#         pass

G = nx.Graph()
#G.add_edges_from(edge_index)
for edge in tqdm(edge_index):
    G.add_edge(edge[0],edge[1])

def get_randomwalk(node, path_length):

    random_walk = [node]

    for i in range(path_length-1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))    
        if len(temp) == 0:
            break

        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node

    return random_walk

if __name__ == '__main__':
    #walk=get_randomwalk(0,10)
    #G=nx.subgraph(G,id_list)
    all_nodes = list(G.nodes())

    random_walks = []
    for n in tqdm(all_nodes):
        for i in range(5):
            random_walks.append(get_randomwalk(n,80))

    model = Word2Vec(window = 10, size=128,sg = 1, hs = 0,
                 negative = 10, 
                 alpha=0.03, min_alpha=0.0007,
                 seed = 14,workers=4,min_count=1)

    model.build_vocab(random_walks, progress_per=2)

    model.train(random_walks, total_examples = model.corpus_count, epochs=20, report_delay=1)
    model.save(dataset_name+'/'+'deepwalk.model')
    model.wv.save(dataset_name+'/'+'deepwalk.wv')
    vector=[]
    for i in range(len(G.nodes())):
        vector.append(model.wv[str(i)])
    np.save(dataset_name+'/'+'deepwalk_22.npy',np.array(vector))
    print()
    