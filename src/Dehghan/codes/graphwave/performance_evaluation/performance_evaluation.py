# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 20:16:53 2017

"""

import numpy as np
import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import matplotlib.pyplot as plt


def density(chi, colors):
    N=len(colors)
    C=len(np.unique(colors))
    list_cols=np.unique(colors)
    D=pd.DataFrame(np.zeros((C,C)),index=np.unique(colors),columns=np.unique(colors))
    med=pd.DataFrame(np.zeros((C,chi.shape[1])),index=np.unique(colors))
    def square(x): return x**2
    vfunc = np.vectorize(square)
    for c in np.unique(colors):
        ind_c=[i for i in range(N) if colors[i]==c]
        med.loc[c,:]=chi[ind_c,:].mean(0)
        med_rep=np.array([med.loc[c,:]]*len(ind_c))
        if len(ind_c)>1:
            D.loc[c,c]=1.0/(len(ind_c)-1)*(vfunc(chi[ind_c,:]-med_rep)).sum()
        else:
            D.loc[c,c]=0
        
    for i in range(1,C):
        c=list_cols[i]
        for j in range(i):
            cc=list_cols[j]
            D.loc[c,cc]=(vfunc(med.loc[c,:]-med.loc[cc,:])).sum()
            D.loc[cc,c]=D.loc[c,cc]
    return D
    
def F_test(chi, colors):
    N=len(colors)
    C=len(np.unique(colors))
    list_cols=np.unique(colors)
    D=pd.DataFrame(np.zeros((C,C)),index=np.unique(colors),columns=np.unique(colors))
    med=pd.DataFrame(np.zeros((C,chi.shape[1])),index=np.unique(colors))
    def square(x): return x**2
    vfunc = np.vectorize(square)
    Var_between=0
    Var_within=0
    overall_mean=chi.mean(0)
    #print 'shape is', overall_mean.shape
    for c in np.unique(colors):
        ind_c=[i for i in range(N) if colors[i]==c]
        med.loc[c,:]=chi[ind_c,:].mean(0)
        med_rep=np.array([med.loc[c,:]]*len(ind_c))
        if len(ind_c)>1:
            D.loc[c,c]=1.0/(len(ind_c)-1)*(vfunc(chi[ind_c,:]-med_rep)).sum()
        else:
            D.loc[c,c]=0
        Var_within+=vfunc(chi[ind_c,:]-med_rep).sum()
        #print Var_within
        Var_between+=len(ind_c)*vfunc(med.loc[c,:].values-overall_mean).sum()
        #print Var_between
    Var_within*=1.0/(C-1)
    Var_between*=1.0/(N-C)
    for i in range(1,C):
        c=list_cols[i]
        for j in range(i):
            cc=list_cols[j]
            D.loc[c,cc]=(vfunc(med.loc[c,:]-med.loc[cc,:])).sum()
            D.loc[cc,c]=D.loc[c,cc]
    return D, Var_between/Var_within,Var_within, Var_between
        
        
