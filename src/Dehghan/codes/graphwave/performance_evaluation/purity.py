# -*- coding: utf-8 -*-
"""
Created on Wed May 10 19:18:46 2017

"""
import numpy as np
#### Assess purity of a graph

def get_purity(node,neighbors,colors):
    l=len([i for i in range(len(neighbors)) if colors[neighbors[i]]==colors[node]])
    #return l*1.0/len(neighbors)
    return l

def compute_overall_purity(node,neighbors,colors):
    ### takes as inut a sorted list of neighbors
    ### Ranks them according to their 
    purity=np.zeros((len(neighbors)))
    for k in range(1,len(neighbors)+1):
        purity[k-1]=get_purity(node,neighbors[:k],colors)
    return purity

def purity(D,colors,m):
    purity=np.zeros((D.shape[0],m))
    for node in range(D.shape[0]):
        list_neighbors=np.argsort(D)[node,:(m+1)].tolist()
        try:
            list_neighbors.remove(node)
        except:
            list_neighbors=list_neighbors[:-1]
        purity[node,:]=compute_overall_purity(node,list_neighbors,colors)
    return purity
        
    
        
def compare_purities(purity1,purity2):
    diff=purity1-purity2
    sb.heatmap(diff)
    return diff
                
