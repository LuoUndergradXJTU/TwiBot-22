import pandas as pd
import numpy as np
import math
import json
import argparse
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.metrics import roc_auc_score as auc
from sklearn.ensemble import AdaBoostClassifier

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
            print("True Positive: {} \nTrue Negative: {}\nFalse Positive: {} \nFalse Negative: {}\n".format(TP, TN, FP, FN))
            print("Precision: {} \nRecall: {} \nF1 Score: {}\n".format(P, R, F1))
            print("AUC: {}".format(AUC))
        
        return acc,P ,R,F1,AUC
    
    except:
        print("ERROR")
        return 0
    
def calculate(cnt):
    accuracy = []
    precision = []
    recall = []
    F1_score = []
    auc = []
    for i in range(5):
        n_X , n_y = SMOTE().fit_resample(X[train_mask],y[train_mask].reshape(-1))
        n_X,n_y = EditedNearestNeighbours().fit_resample(n_X,n_y.reshape(-1))
        a = AdaBoostClassifier(n_estimators=100,random_state=i*100)
        a.fit(n_X,n_y)
        y_predict = a.predict(X[test_mask])

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


def handle_data(df):
    for i in range(len(df)):
        if 'protected' not in df[i].keys() or df[i]['protected'] is None or df[i]['protected'] == 'False' or df[i]['protected'] == 0 :
            df[i]['protected'] = 0
        else:
            df[i]['protected'] = 1 
        if 'verified' not in df[i].keys() or df[i]['verified'] is None or df[i]['verified'] == 'False' or df[i]['verified'] == 0 :
            df[i]['verified'] = 0
        else:
            df[i]['verified'] = 1
        if args.datasets == "Twibot-22":
            if df[i]['location'] is None:
                df[i]['location'] = 0
            else:
                df[i]['location'] = 1
            if df[i]['profile_image_url'] == '':
                df[i]['profile_image_url'] = 0
            else:
                df[i]['profile_image_url'] = 1
    return df

parser = argparse.ArgumentParser(description="Reproduction of Kudugunta et al. with SMOTENN and rain forest")
parser.add_argument("--datasets", type=str, default="Twibot-22", help="dataset name")

args = parser.parse_args()
dir = "../../datasets/"

print("loading data...")
labels = pd.read_csv(dir+args.datasets+"/label.csv")
user_num = len(labels)
y = np.zeros(user_num)
dict={}
if args.datasets == "Twibot-22":
    data = json.load(open(dir+args.datasets+"/user.json"))
else:
    data = json.load(open(dir+args.datasets+"/node.json"))
all = {}
for i in range(user_num):
    all[data[i]['id']] = 1
split= pd.read_csv(dir+args.datasets+'/split.csv')
t,j,k=0,0,0
for i in range(len(split)):
    if split['id'][i] in all.keys() and split['id'][i] not in dict.keys(): 
        if split['split'][i] == "train" :
            dict[split['id'][i]] = j
            j+=1
        if split['split'][i] == "val":
            dict[split['id'][i]] = k
            k+=1
        if split['split'][i] == "test":
            dict[split['id'][i]] = t
            t+=1
al = {}      
for i in range(len(split)):
    if split['id'][i] in all.keys() and split['id'][i] not in al.keys():
        al[split['id'][i]] = 1
        if split['split'][i] == "val":
            dict[split['id'][i]] += j
        if split['split'][i] == "test":
            dict[split['id'][i]] += j+k
#print(j),print(k),print(user_num)
#print(dict.values())
for i in range(user_num):
    if labels['id'][i] in all.keys():
        if labels['label'][i]=="bot":
            #print(dict[labels['id'][i]],labels['id'][i])
            y[dict[labels['id'][i]]] = 1
train_mask=range(j)
val_mask=range(j,j+k)
test_mask=range(j+k,user_num)

#print(user_num-k)
#print(j),print(k),print(user_num)
#print(y.sum()),print(len(y))


print("begin to process data...")
data = handle_data(data)
feature_num = 6
if args.datasets == "Twibot-22":
    feature_num = 8
feature = ['followers_count','following_count','tweet_count','listed_count','protected','verified','profile_image_url','location']

X = np.zeros((user_num,feature_num))
for i in range(user_num):
        for j in range(4):
            X[dict[data[i]['id']]][j] = data[i]['public_metrics'][feature[j]]
        for j in range(4,feature_num):
            X[i][j] = data[i][feature[j]]

print("begin to train...")
calculate(5)
