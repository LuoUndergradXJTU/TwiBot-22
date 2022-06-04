import csv
from cv2 import IMREAD_REDUCED_GRAYSCALE_2
import numpy as np
from pyparsing import lineStart
from tqdm import tqdm
import networkx as nx
dataset_name='Twibot-22'
edges=[]
id_list=np.load('/data2/whr/lyh/twibot22_baseline/'+dataset_name+'/id.npy')
id_list=list(id_list)
id_include=set()
G=nx.Graph()
with open('/data2/whr/czl/TwiBot22-baselines/datasets/'+dataset_name+'/'+ 'edge.csv','r') as csvfile:
    lines=csvfile.readlines()
    for row in tqdm(lines):

        row=row.rstrip()
        id1,relation,id2=row.split(',')
        if id1[0] == 's':
            continue
        id1=eval(id1[1:])
        id2=eval(id2[1:])
        #G.add_edges_from([[id1,id2]])
        edges.append([id1,id2])
edges=np.array(edges)
np.save(dataset_name+'/edges.npy',edges)
np.save(dataset_name+'/id_include.npy',id_include)