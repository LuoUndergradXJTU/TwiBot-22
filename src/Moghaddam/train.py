import numpy as np
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import RandomForestClassifier
import json
import argparse
import pandas as pd

def get_accuracy(AL, y, verbose=1):    
    try:
        AL = np.array(AL)
        y = np.array(y)

        AL = AL.reshape(-1)
        y = y.reshape(-1)

        AL = AL > 0.5
        AL = AL.astype(int)

        y = y > 0.5
        y = y.astype(int)
        
        AUC = auc(y,AL)
        total = AL.shape[0]

        TP = np.sum(np.logical_and(AL==1, y==1))
        TN = np.sum(np.logical_and(AL==0, y==0))

        FP = np.sum(np.logical_and(AL==1, y==0))
        FN = np.sum(np.logical_and(AL==0, y==1))

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = (2 * P * R) / (P + R)
        acc = np.sum(AL == y)/total
        
        if verbose == 1:
            print("\nAccuracy: {} \n".format(acc))
            #print("True Positive: {} \nTrue Negative: {}\nFalse Positive: {} \nFalse Negative: {}\n".format(TP, TN, FP, FN))
            print("Precision: {} \nRecall: {} \nF1 Score: {}\n".format(P, R, F1))
            print("AUC: {}".format(AUC))
        
        return acc , P , R,F1,AUC
    
    except:
        return 0
    
def calc(cnt):
    accuracy = []
    precision = []
    recall = []
    F1_score = []
    auc = []
    for i in range(5):
        n_X , n_y = true_X[train_mask],y[train_mask].reshape(-1)
        a = RandomForestClassifier(n_estimators=50,random_state=i*100,max_features=3)
        a.fit(n_X,n_y)
        y_predict = a.predict(true_X[test_mask])
        acc , P ,R,F1,AUC = get_accuracy(y_predict, y[test_mask],0)
        accuracy.append(acc)
        precision.append(P)
        recall.append(R)
        F1_score.append(F1)
        auc.append(AUC)
        
    print("\nAccuracy mean: {} std: {} \n".format(np.array(accuracy).mean(),np.array(accuracy).std(ddof=1)))
    print("Precision mean: {} std: {} \n Recall mean: {} std: {}\nF1 Score mean: {} std: {}\n".format(
        np.array(precision).mean(),np.array(precision).std(ddof=1),
        np.array(recall).mean(),np.array(recall).std(ddof=1),
        np.array(F1_score).mean(),np.array(F1_score).std(ddof=1)))
    print("AUC mean: {} std: {}".format(np.array(auc).mean(), np.array(auc).std(ddof=1)))


parser = argparse.ArgumentParser(description="Reproduction of Kudugunta et al. with SMOTENN and rain forest")
parser.add_argument("--datasets", type=str, default="Twibot-22", help="dataset name")
args = parser.parse_args()
dir = "../../datasets/"

print("loading data...")
labels = pd.read_csv(dir+args.datasets+"/label.csv")
user_num = len(labels)
y = np.zeros(user_num)
dict={}

split= pd.read_csv(dir+args.datasets+'/split.csv')
for i in range(user_num):
    dict[split['id'][i]] = i
for i in range(user_num):
    if split['split'][i] == "valid" or split['split'][i] == 'val':
        j=i
        break
for i in range(j,user_num):
    if split['split'][i] == "test":
        k=i
        break
for i in range(user_num):
    if labels['label'][i]=="bot":
        y[dict[labels['id'][i]]] = 1
train_mask=range(j)
val_mask=range(j,k)
test_mask=range(k,user_num)

true_X = np.load("./"+args.datasets+".npy")
print("begin to train")    
calc(5)