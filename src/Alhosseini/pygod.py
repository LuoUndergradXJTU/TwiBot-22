import os.path as osp
import pandas as pd
import scipy.io as scio
import torch
from pygod.models import MLPAE
from pygod.utils.metric import \
    eval_roc_auc
from torch_geometric.data import Data

from sklearn.metrics import roc_auc_score
from torch_scatter import scatter
from torch_geometric.utils import to_dense_adj
data2=scio.loadmat('ACMnew.mat')
data1 = Data(x=torch.FloatTensor(data2['X']), edge_index=torch.LongTensor(data2['edge_index']), y=torch.IntTensor(data2['label']))
# data1, ya = gen_attribute_outliers(data1, n=100, k=50)
# data1, ys = gen_structure_outliers(data1, m=10, n=10)
# data1.y = torch.logical_or(ys, ya).int()
# scio.savemat(dataNew, {'edge_index': np.array(data1.edge_index),'X': np.array(data1.x), 'label': np.array(data1.y)})
model=MLPAE()
model.fit(data1)
outlier_scores = model.decision_function(data1)
df=pd.DataFrame(data=outlier_scores)
outlier_scores = df.fillna(0)
auc_score = eval_roc_auc(data1.y.numpy()[0,:], outlier_scores)


print('AUC Score:', auc_score)