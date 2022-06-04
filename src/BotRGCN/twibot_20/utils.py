import torch
from torch import nn

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def init_weights(m):
    if type(m)==nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        
