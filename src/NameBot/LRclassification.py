import preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score,roc_auc_score
import numpy as np
from sklearn.svm import SVC, LinearSVC
from pathlib import Path
import sys

sys.path.append("..")

#from utils.dataset import merge_and_split
import pandas as pd

def standardize(X):
    """特征标准化处理
    Args:
        X: 样本集
    Returns:
        标准后的样本集
    """
    m, n = X.shape
    # 归一化每一个特征
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
    # 归一化每一个特征
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


def get_data_dir(server_id):
    if server_id == "206":
        return Path("/new_temp/fsb/Twibot22-baselines/datasets")
    elif server_id == "208":
        return Path("")
    elif server_id == "209":
        return Path("/data2/whr/czl/TwiBot22-baselines/datasets")
    else:
        raise NotImplementedError


dataset_names = [
    'botometer-feedback-2019', 'botwiki-2019', 'celebrity-2019', 'cresci-2015', 'cresci-2017', 'cresci-rtbust-2019', 'cresci-stock-2018', 'gilani-2017', 'midterm-2018', 'political-bots-2019', 'pronbots-2019', 'vendor-purchased-2019', 'verified-2019', "Twibot-20", "Twibot-22"
]

def merge_and_split(dataset="botometer-feedback-2019", server_id="209"):
    assert dataset in dataset_names, f"Invalid dataset {dataset}"
    dataset_dir = get_data_dir(server_id) / dataset
    if dataset == "Twibot-22": 
        node_info = pd.read_json(dataset_dir / "user.json")
    else:
        node_info = pd.read_json(dataset_dir / "node.json")
    label = pd.read_csv(dataset_dir / "label.csv")
    split = pd.read_csv(dataset_dir / "split.csv")
    node_info = pd.merge(node_info, label)
    node_info = pd.merge(node_info, split)
    
    
    train = node_info[node_info["split"] == "train"]
    valid = node_info[node_info["split"] == "val"]
    test = node_info[node_info["split"] == "test"]
    
    
    return train, valid, test


def preprocess_dataset(dataset, server_id="209"):
    train, valid, test = merge_and_split(dataset=dataset, server_id=server_id)
    return train, valid, test

def get_feature(traindata,testdata):
    trainname = list(traindata["username"])
    testname = list(testdata["username"])
    name = trainname + testname
        
    entropy, upper_list, lower_list = preprocess.ShannonEntropyAndNomalize(name)
    tfidf = preprocess.TFIDF(name)

    feature = tfidf
    for i in range(len(name)):
        feature[i].append(float(entropy[i]))
        feature[i].append(float(upper_list[i]))
        feature[i].append(float(lower_list[i]))   
        
    train_features = feature[0:len(trainname)]
    test_features = feature[len(trainname):]
    
    train_features = standardize(np.array(train_features))
    test_features = standardize(np.array(test_features))
    
    return train_features, test_features

def get_label(data):
    labels = list(data["label"])
    labels = list(map(lambda x: 0 if x == "human" else 1, labels))
    return labels

def data_load(dataname):
    train, valid, test = preprocess_dataset(dataname, "209")
    train = pd.concat([train,valid])
    #print(train)
    #print(test)

    y_train = get_label(train)
    y_test = get_label(test)

    X_train, X_test = get_feature(train, test)
    return X_train, X_test, y_train, y_test
    



def Botclassifier(X_train, X_test, Y_train, Y_test):
    #classifier = SVC(kernel = 'rbf', C = 2, gamma = 'auto', verbose = 2).fit(X_train, Y_train)
    #classifier = LogisticRegression(solver = 'liblinear', max_iter = 500, tol = 1e-7, C = 0.1, verbose = 2).fit(X_train, Y_train)
    classifier = LogisticRegression(solver = 'saga', max_iter = 2000, tol = 1e-7, C = 50, verbose = 2).fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    print(y_pred)
    #print(y_pred)
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
    datasetname = ["botometer-feedback-2019"]
    for dataname in datasetname:
        X_train, X_test, y_train, y_test = data_load(dataname)
        metric_get(X_train, X_test, y_train, y_test)
 