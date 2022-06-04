import xgboost as xgb
import numpy as np
import logging
import os
from argparse import ArgumentParser
from sklearn.metrics import f1_score,precision_score,accuracy_score,recall_score,roc_auc_score
from imblearn.combine import SMOTEENN
from collections import Counter
parser= ArgumentParser()
parser.add_argument('--dataset',type=str,default='Twibot-20' )

args = parser.parse_args()
dataset=args.dataset
debug=False
random_state=[0,12345,45678,191018,991237]
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug else logging.INFO)
log_dir = 'results'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
#log_dir.mkdir(exist_ok=True, parents=True)
log_file = log_dir +'/'+ "dataset.log"
#log_file.touch(exist_ok=True)
# train_id=np.load(dataset+'/train_id.npy')
# val_id=np.load(dataset+'/val_id.npy')
# test_id=np.load(dataset+'/test_id.npy')
logging_handler = logging.FileHandler(log_file)
logging_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(logging_handler)
'''
train 8278
dev 2365
test 1183
'''
#(deepwalk,node2vec,graph_wave)
emb_list=['graph_wave','deepwalk','node2vec','roix','role2vec','struct']
#emb_list=['node2vec']
#emb_list=['struct']
#emb_list=['deepwalk']
# id_include=(np.load(dataset+'/id_include.npy',allow_pickle=True))
# id_include=list(id_include.item())
for emb_name in emb_list:
    emb=np.load(dataset+'/'+emb_name+'_emb.npy')
    #nlp features 
    nlp= np.load(dataset+'/nlp.npy')

    #profile features
    p=np.load(dataset+'/profile.npy')

    #graph features
    #gf=np.load(dataset+'/node_fea.npy').T[:24679]
    fea=np.concatenate((nlp,p,emb),1)
    label_train=np.load(dataset+'/label_train.npy')
    label_val=np.load(dataset+'/label_val.npy')
    label_test=np.load(dataset+'/label_test.npy')
    label=np.concatenate((label_train,label_val,label_test))
    
    # label=label[id_include]
    # label_train=label[train_id]
    # label_test=label[test_id]
    # label_val=label[val_id]
    train=fea[:len(label_train)]
    val=fea[len(label_train):len(label_train)+len(label_val)]
    test=fea[-len(label_test):]
    
    print(f'train size:{len(train)} val size: {len(val)} test size:{len(test)}')
    
    # print('test')
    # counter = Counter(label_test)
    # print(counter)
    # test, label_test = oversample.fit_resample(test,label_test)
    # counter = Counter(label_test)
    # print(counter)
    # # print('val')
    # counter = Counter(label_val)
    # print(counter)
    # val, label_val = oversample.fit_resample(val,label_val)
    # counter = Counter(label_val)
    # print(counter)
    
    p=np.zeros(5)
    r=np.zeros(5)
    f1=np.zeros(5)
    acc=np.zeros(5)
    roc_auc=np.zeros(5)
    for i in range(5):
         
        # oversample = SMOTEENN(0.5,random_state=random_state[i])
        # print('train')
        # counter = Counter(label_train)
        # print(counter)
        # train_new, label_train_new = oversample.fit_resample(train, label_train)
        # counter = Counter(label_train_new)
        # print(counter)
        
        
        # print('test')
        # counter = Counter(label_test)
        # print(counter)
        # test_new, label_test_new = oversample.fit_resample(test,label_test)
        # counter = Counter(label_test_new)
        # print(counter)
        
        model = xgb.XGBClassifier(learning_rate=0.2,
                                n_estimators=50,         
                                max_depth=3,  
                                max_leaves=3,             
                                min_child_weight = 1,      
                                gamma=0.,                  
                                subsample=0.8,             
                                colsample_btree=0.8,       
                                objective='multi:softmax', 
                                num_class=2,
                                scale_pos_weight=1,        
                                random_state=random_state[i])

        model.fit(train,
                    label_train,
                    eval_set = [(val,label_val)],
                    eval_metric = "mlogloss",
                    early_stopping_rounds = 10,
                    verbose = True)

        predict = model.predict(test) 

        p[i]=precision_score(predict,label_test)
        r[i]=recall_score(predict,label_test)
        f1[i]=f1_score(predict,label_test)
        acc[i]=accuracy_score(predict,label_test)
        try:
            roc_auc[i]=roc_auc_score(predict,label_test)
        except:
            pass  
    print(f"p {p.mean():.4f} r {r.mean():.4f} f1 {f1.mean():.4f} acc {acc.mean():.4f} roc_auc {roc_auc.mean():.4f}")
    logger.info(f"random state {random_state}")
    logger.info(f"dataset: {dataset} emb {emb_name} mean p {p.mean():.4f} r {r.mean():.4f} f1 {f1.mean():.4f} acc {acc.mean():.4f} roc_auc {roc_auc.mean():.4f}")
    logger.info(f"dataset: {dataset} emb {emb_name} var p {p.var():.4f} r {r.var():.4f} f1 {f1.var():.4f} acc {acc.var():.4f} roc_auc {roc_auc.var():.4f}")