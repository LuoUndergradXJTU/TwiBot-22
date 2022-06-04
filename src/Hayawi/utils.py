import torch
from torch import nn
import math
import nltk
import pandas as pd

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        
def count_each_char(string):
    res = {}
    for i in string:
        if i not in res:
            res[i] = 1
        else:
            res[i] += 1
    return res

def entropy(string):
    length = len(string)
    h=0
    freq=count_each_char(string)
    for i in string:
        p_i=freq[i]/length
        h-=p_i*math.log(p_i,2)
    return h/length

def bigrams_freq(string):
    C=0
    bigrams_unique=set(nltk.bigrams(string))
    K=len(bigrams_unique)
    freq=dict(pd.DataFrame(nltk.bigrams(string)).value_counts())
    for each in bigrams_unique:
        C+=freq[each]
    return C/K

