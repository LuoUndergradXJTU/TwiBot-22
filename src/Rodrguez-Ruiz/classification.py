from preprocess import get_directdata, text_feature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
import sys

sys.path.append("..")
from dataset import merge_and_split
import pandas as pd


def preprocess_dataset(dataset, server_id="209"):
    train, valid, test = merge_and_split(dataset=dataset, server_id=server_id)
    return train, valid, test


def standardize(X):
    """特征标准化处理
    Args:
        X: 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return  X



def normalize(X):
    """Min-Max normalization     sklearn.preprocess 的MaxMinScalar
    Args:
        X: 样本集
    Returns:
        归一化后的样本集
    """
    m, n = X.shape
    for j in range(n):
        features = X[:,j]
        minVal = features.min(axis=0)
        maxVal = features.max(axis=0)
        diff = maxVal - minVal
        if diff != 0:
           X[:,j] = (features-minVal)/diff
        else:
           X[:,j] = 0
    return X


def get_feature(dataset, server_id="209"):
    train, valid, test = preprocess_dataset(dataset=dataset)
    train = pd.concat([train, valid])
    
    friend, followers, tcount, listcount, seconds = get_directdata(dataset, server_id=server_id)
    mentions, hashtags, urls, uniquementions, uniquehashtag, uniqueurl = text_feature(dataset, server_id = server_id)
    tcount = np.array(tcount)
    mentions = (np.array(mentions) / (tcount + 1e-10)).tolist()
    hashtags = (np.array(hashtags) / (tcount + 1e-10)).tolist()
    urls = (np.array(urls) / (tcount + 1e-10)).tolist()
    uniquementions = (np.array(uniquementions) / (tcount + 1e-10)).tolist()
    uniquehashtag = (np.array(uniquehashtag) / (tcount + 1e-10)).tolist()
    uniqueurl =  (np.array(uniqueurl) / (tcount + 1e-10)).tolist()
    
    intertime = (np.array(seconds) / (tcount + 1e-10)).tolist()
    ffratio = (np.array(friend) / (np.array(followers) + 1e-10)).tolist() 
    
    feature = hashtags
    feature = np.expand_dims(feature, axis=1).tolist()
    
    
    for i in range(len(train) + len(test)):
        feature[i].append(float(listcount[i]))
        feature[i].append(float(urls[i]))
        feature[i].append(float(mentions[i]))   
        feature[i].append(float(intertime[i]))   
        feature[i].append(float(ffratio[i])) 
        feature[i].append(float(uniquehashtag[i])) 
        feature[i].append(float(uniquementions[i])) 
        feature[i].append(float(uniqueurl[i])) 
        
    train_features = standardize(np.array(feature[0:len(train)]))
    test_features = standardize(np.array(feature[len(train):]))
    
    
    return train_features, test_features

def get_label(data):
    labels = list(data["label"])
    labels = list(map(lambda x: 0 if x == "human" else 1, labels))
    return labels

def data_load(dataname, server_id):
    train, valid, test = preprocess_dataset(dataset=dataname, server_id=server_id)
    train = pd.concat([train,valid])
    #print(train)
    #print(test)

    y_train = get_label(train)
    y_test = get_label(test)

    X_train, X_test = get_feature(dataname)
    return X_train, X_test, y_train, y_test
    



def Botclassifier(X_train, X_test, Y_train, Y_test, random_state = None):
    #classifier = LogisticRegression(solver = 'saga', max_iter = 2000, tol = 1e-7, C = 10, verbose = 2).fit(X_train, Y_train)

    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    classifier = SVC(kernel = 'rbf', C = 2, gamma = 'auto', random_state = random_state).fit(X_train, Y_train)
    
    y_pred = classifier.predict(X_test)
    
    acc = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)    
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_pred)
    
    return acc, precision, recall, f1, auc
    

def metric_get(X_train, X_test, y_train, y_test):
    acc, precision, recall, f1, auc = Botclassifier(X_train, X_test, y_train, y_test)


    metric = [acc, precision, recall, f1, auc]
    print(metric)
    
if __name__ == '__main__':
    datasetname = ["cresci-2017"]
    for dataname in datasetname:
        X_train, X_test, y_train, y_test = data_load(dataname = dataname, server_id = "209")
        metric_get(X_train, X_test, y_train, y_test)
    
    