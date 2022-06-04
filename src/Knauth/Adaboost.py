from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score,roc_auc_score
from imblearn.combine import SMOTEENN
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import logging
import os
from sklearn.tree import DecisionTreeClassifier
random_state=[0,12345,45678,191018,991237]
#random_state=[0,100,200,300,400]
debug=False
dataset='botometer-feedback-2019'
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug else logging.INFO)
log_dir = 'results'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
#log_dir.mkdir(exist_ok=True, parents=True)
log_file = log_dir +'/'+ "dataset.log"
#log_file.touch(exist_ok=True)
logging_handler = logging.FileHandler(log_file)
logging_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(logging_handler)
all_train=np.load(dataset+'/'+'train_ac.npy')
all_test=np.load(dataset+'/'+'test_ac.npy')
# div=all.max(axis=0)
# div[div==0]=1
# all=all/div
#label=np.load(dataset+'/'+'label.npy')
# std=StandardScaler()
# all=std.fit_transform(all)
# lev=std.fit_transform(np.expand_dims(lev,1))
# lev_train=lev[:8278+2365]
# lev_test=lev[8278+2365:]
# all_train=all[:700000]
# all_test=all[900000:]

#label_train=label[:8278]
#label_test=label[8278+2365:]
# all_train=np.load(dataset+'/'+'train_ac.npy')
# all_test=np.load(dataset+'/'+'test_ac.npy')

label_train=np.load(dataset+'/'+'label_train.npy')
label_test=np.load(dataset+'/'+'label_test.npy')
#all_train
# oversample = SMOTEENN()
# counter = Counter(label_train)
# print(counter)
# all_train, label_train = oversample.fit_resample(all_train, label_train)
# counter = Counter(label_train)
# print(counter)
n_estimators=100
#random_state=200
#clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
#kf = KFold(n_splits=5,shuffle=False) 


# clf.fit(all_train, label_train)
# predict=clf.predict(all_test)
# p=precision_score(predict,label_test)
# r=recall_score(predict,label_test)
# f1=f1_score(predict,label_test)
# acc=accuracy_score(predict,label_test)
# roc_auc=roc_auc_score(predict,label_test)
# print(f"p {p:.4f} r {r:.4f} f1 {f1:.4f} acc {acc:.4f} roc_auc {roc_auc:.4f}")
# logger.info(f"dataset: {dataset} n_estimators {n_estimators} p {p:.4f} r {r:.4f} f1 {f1:.4f} acc {acc:.4f} roc_auc {roc_auc:.4f}")
p=np.zeros(5)
r=np.zeros(5)
f1=np.zeros(5)
acc=np.zeros(5)
roc_auc=np.zeros(5)
for i in range(5):
    # all_train=lev[train_index]
    # all_test=lev[test_index]
    # label_train=label[train_index]
    # label_test=label[test_index]
    #print(i)
    # all_traincart=DecisionTreeClassifier(random_state=random_state[i])
    clf = AdaBoostClassifier(n_estimators=n_estimators,random_state=random_state[i])
    clf.fit(all_train, label_train)
    predict=clf.predict(all_test)
    p[i]=precision_score(predict,label_test)
    r[i]=recall_score(predict,label_test)
    f1[i]=f1_score(predict,label_test)
    acc[i]=accuracy_score(predict,label_test)
    roc_auc[i]=roc_auc_score(predict,label_test)
    print(f"p {p[i]:.4f} r {r[i]:.4f} f1 {f1[i]:.4f} acc {acc[i]:.4f} roc_auc {roc_auc[i]:.4f}")
print(f"p {p.mean():.4f} r {r.mean():.4f} f1 {f1.mean():.4f} acc {acc.mean():.4f} roc_auc {roc_auc.mean():.4f}")
logger.info(f"random state {random_state}")
logger.info(f"dataset: {dataset} mean p {p.mean():.4f} r {r.mean():.4f} f1 {f1.mean():.4f} acc {acc.mean():.4f} roc_auc {roc_auc.mean():.4f}")
logger.info(f"dataset: {dataset} var p {p.var():.4f} r {r.var():.4f} f1 {f1.var():.4f} acc {acc.var():.4f} roc_auc {roc_auc.var():.4f}")