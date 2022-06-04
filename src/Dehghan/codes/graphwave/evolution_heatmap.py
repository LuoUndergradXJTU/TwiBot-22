from distances.distances_between_graphs import *
import pygsp
import matplotlib.pyplot as plt
import networkx as nx 
import numpy as np
import pandas as pd
from utils.graph_tools import *


def compute_evolution_heat_diff(node1, mode, list_heat_df1, list_heat_df2,
                                type_distance="auc", normalize=True, mode_diff="agg",
                                plot=False, savefig=False):
    
    n = len(list_heat_df1[mode])
    ind1 = np.argsort((list_heat_df1[mode]).iloc[:, node1])
    ind2 = np.argsort((list_heat_df2[mode]).iloc[:, node1])
    test1 = (list_heat_df1[mode]).iloc[ind1,node1]
    test2 = (list_heat_df2[mode]).iloc[ind2,node1]
    if plot == True:
        w1 = np.array([(test1[i+1] - test1[i]) for i in range(n-1)])
        w2 = np.array([(test2[i+1] - test2[i]) for i in range(n-1)])
        m1 = np.array([0.5 * (test1[i+1] + test1[i]) for i in range(n-1)])
        m2 = np.array([0.5 * (test2[i+1] + test2[i]) for i in range(n-1)])
        plt.plot(test1, test2, c="red", label=str(node1) + "at time 0")
        plt.plot(test1, test1, c="black", label="45 degree line")
        plt.title("qqplot of node "+ str(node1))
        plt.legend(loc="upper left")
        plt.xlabel("node "+str(node1)+ "at time 0")
        plt.ylabel("new node "+str(node1)+ "at time 1")
        area1=[ 0.5*np.abs(m2[i]-m1[i])*(w1[i]+w2[i]) for i in range(n-1)]
        if savefig is True:
            plt.savefig("plots/evolution_qq_heat_profile"+str(mode)+"_nd"+str(node1)+".png")
    if type_distance == "auc":
        d=compute_auc(test1, test2, normalize=normalize, mode_diff=mode_diff)
        #d=np.sum(area1)
    elif type_distance == "ks":
        d=np.max(np.abs(np.sort(test1)-np.sort(test2)))
    elif type_distance == "ks_p":
        stats = sc.stats.ks_2samp(test1, test2)
        d = stats[0]
    elif type_distance == "corr":
        test11 = (list_heat_df1[mode]).iloc[ind1, node1]
        test12 = (list_heat_df2[mode]).iloc[ind1, node1]
        m11 = np.array([0.5 * (test11[i+1] + test11[i]) for i in range(n-1)])
        m12 = np.array([0.5 * (test12[i+1] + test12[i]) for i in range(n-1)])
        test21 = (list_heat_df1[mode]).iloc[ind2, node1]
        test22 = (list_heat_df2[mode]).iloc[ind2, node1]
        m21 = np.array([0.5 * (test21[i+1] + test21[i]) for i in range(n-1)])
        m22 = np.array([0.5 * (test22[i+1] + test22[i]) for i in range(n-1)])
    #area1=[(delta2[i]-delta1[i])*m1[i]-delta2[i]*test1[i]+test2[i]*delta1[i] for i in range(n-1)]
        area1=[0.5*(np.abs(m12[i] - m11[i]) * w1[i] + 
               np.abs(m21[i] - m22[i]) * w2[i]) for i in range(n-1)]
        d=np.sum(area1)
    else:
        print "distance type not recognized. Switching to auc"
        d=np.sum(area1)
    
    return d
    

def compare_heat_histograms(node1, node2, mode, list_heat_df,savefig=False):
    plt.hist((list_heat_df[mode]).loc[:, node1], bins=20, color="salmon",
              label="mode "+str(mode)+" for node "+str(node1), alpha=0.5)
    plt.hist((list_heat_df[mode]).loc[:, node2], bins=20, color="lightblue",
              label="mode "+str(mode)+" for node "+str(node2), alpha=0.5)
    plt.legend(loc="upper right",fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node1)+" and "+str(node2) )
    if savefig is True:
        plt.savefig("plots/comp_heat_hist"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    return True


def plot_centered_heat_diffusion(node, mode, G, list_heat_df,
                                 type_graph="nx", savefig=False):
    if type_graph == "pygsp":
        G_nx = nx.from_scipy_sparse_matrix(G.W)
    else:
        G_nx  =G
    pos = nx.spring_layout(G_nx)
    Sf = list_heat_df[mode]
    nodes = nx.draw_networkx_nodes(G_nx, pos, node_color=Sf.loc[:,node],
                                   cmap="hot", label=range(100))
    edges = nx.draw_networkx_edges(G_nx, pos, edge_color="black",width=0.5)
    labels = nx.draw_networkx_labels(G_nx, pos)
    plt.sci(nodes)
    plt.colorbar()
    if savefig == True:
        plt.savefig("plots/heat"+str(mode)+"_nd"+str(node)+".png")
    return True


def compare_heat_profiles(node1,node2, mode,list_heat_df,savefig=False):
    Sf=list_heat_df[mode]
    plt.plot(np.sort(Sf.loc[:,node1]),c="red",label=str(node1))
    plt.plot(np.sort(Sf.loc[:,node2]),c="blue",label=str(node2))
    if savefig==True:
        plt.savefig("plots/comp_heat_profile"+str(mode)+"_nds"+str(node1)+"_"+str(node2)+".png")
    return True


def compare_heat_profiles_tot(node1, node2, list_heat_df, savefig=False):
    plt.plot(np.sort((list_heat_df[0]).loc[:, node1]), c="red", label="tau1, node "+str(node1))
    plt.plot(np.sort((list_heat_df[0]).loc[:, node2]), c="blue", label="tau1, node "+str(node2))
    plt.plot(np.sort((list_heat_df[1]).loc[:, node1]), ':',c="red", label="tau2, node "+str(node1))
    plt.plot(np.sort((list_heat_df[1]).loc[:, node2]), ':',c="blue", label="tau2, node "+str(node2))
    plt.plot(np.sort((list_heat_df[2]).loc[:, node1]), '--',c="red", label="tau3, node "+str(node1))
    plt.plot(np.sort((list_heat_df[2]).loc[:, node2]), '--',c="blue", label="tau3, node "+str(node2))
    plt.plot(np.sort((list_heat_df[3]).loc[:, node1]), '-.',c="red", label="tau4, node "+str(node1))
    plt.plot(np.sort((list_heat_df[3]).loc[:, node2]), '-.',c="blue", label="tau4, node "+str(node2))
    plt.legend(loc="upper left", fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node1)+" and "+str(node2) )
    if savefig is True:
        plt.savefig("plots/comp_all_heat_profile"+"_nds"+str(node1)+"_"+str(node2)+".png")
    return True 


def plot_heat_distribution(list_heat_df, node,
                           colors=["salmon", "orange", "lightblue", "green",
                                   "pink", "red", "purple", "grey"]):
    for mode in range(len(list_heat_df)):
        plt.hist(1.0 / np.max((list_heat_df[mode]).loc[:, node])
                 * (list_heat_df[mode]).loc[:, node],
                 bins=20,color=colors[mode],
                 label="mode "+str(mode)+" for node "+str(node), alpha=0.5)
    plt.legend(loc="upper right", fontsize=7)
    plt.title("Comparison of the heat profiles for nodes "+str(node) )    
    return True
