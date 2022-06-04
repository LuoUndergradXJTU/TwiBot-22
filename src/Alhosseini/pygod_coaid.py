import os.path as osp
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygod.models import DONE
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data
import scipy.sparse
from pygod.utils import gen_attribute_outliers, gen_structure_outliers
import scipy.io as scio
from pygod.utils.metric import \
    eval_roc_auc, \
    eval_recall_at_k, \
    eval_precision_at_k
import heapq
import pandas as pd
from sklearn.metrics import roc_auc_score
data2=scio.loadmat('gossipcop.mat')
x1=data2['X1']
x2=data2['X2']
bu=np.zeros((x2.shape[0],x1.shape[1]-x2.shape[1]))
X=np.vstack([x1,np.hstack([x2,bu])])
# X=np.vstack([x1,x2])
np.nan_to_num(X)
np.nan_to_num(data2['edge_index'])
data1 = Data(x=torch.FloatTensor(X), edge_index=torch.LongTensor(data2['edge_index']), y=torch.LongTensor(data2['label'][0,:]))
model=DONE(epoch=10,lr=0.01)
for i in range(10):
    model.fit(data1)
    outlier_scores= model.decision_function(data1)
    df = pd.DataFrame(data=outlier_scores)
    outlier_scores = df.fillna(0)
    np.nan_to_num(data1.y.numpy())
    auc_score = eval_roc_auc(data1.y.numpy(), outlier_scores)
    print('AUC Score:', auc_score)

