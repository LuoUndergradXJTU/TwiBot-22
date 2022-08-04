import os
import numpy as np
import pandas as pd
import datetime as dt
import sklearn.metrics as mt
from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

datasets = ['botometer-feedback-2019', 'cresci-2015', 'cresci-2017', 'cresci-rtbust-2019', 'cresci-stock-2018', 'gilani-2017', 'midterm-2018', 'Twibot-20', 'gilani-2017', 'Twibot-22']
# datasets = ['Twibot-22']

path1 = Path('datasets')
# f = open('src/efthimion/result.txt', 'w')

for dataset in tqdm(datasets):
    
    if dataset == 'Twibot-22':
        nodes = pd.read_json(path1 / dataset / 'user.json')
        labels = pd.read_csv(path1 / dataset / 'label.csv')
        split = pd.read_csv(path1 / dataset / 'split.csv')
    else:
        nodes = pd.read_json(path1 / dataset / 'node.json')
        labels = pd.read_csv(path1 / dataset / 'label.csv')
        split = pd.read_csv(path1 / dataset / 'split.csv')
    
    users = pd.merge(labels, nodes)
    users = pd.merge(users, split)


    scores = pd.DataFrame()
    def scoring1(row):
        if 'default' in row:
            return 1
        else:
            return 0
    scores['profile_pic'] = list(map(scoring1, users['profile_image_url']))
    
    def scoring2(row):
        if pd.isnull(row) or row == '':
            return 1
        else:
            return 0
        
    scores['has_screen_name'] = users['name'].apply(scoring2)
    
    def scoring3(row):
        if row['followers_count'] < 30:
            return 1
        else:
            return 0
        
    scores['30followers'] = users['public_metrics'].apply(scoring3)
    
    def scoring4(row):
        if pd.isnull(row) or row == '':
            return 1
        else:
            return 0
        
    scores['geoloc'] = users['location'].apply(scoring4)
    
    def scoring5(row):
        if not pd.isnull(row) and 'http' not in row:
            return 0
        else:
            return 1
    scores['banner_link'] = users['description'].apply(scoring5)
    
    
    def scoring7(row):
        if row['tweet_count'] > 50:
            return 0
        else:
            return 1
        
    scores['50tweets'] = users['public_metrics'].apply(scoring7)
    
    def scoring8(row):
        if 2 * row['followers_count'] >= row['following_count']:
            return 0
        else:
            return 1
        
    scores['twice_num_followers'] = users['public_metrics'].apply(scoring8)
    
    def scoring9(row):
        if row['following_count'] > 1000:
            return 1
        else:
            return 0
        
    scores['1000friends'] = users['public_metrics'].apply(scoring9)
    
    def scoring(row):
        if row['tweet_count'] == 0:
            return 1
        else:
            return 0
        
    scores['NeverTweeted'] = users['public_metrics'].apply(scoring)
    
    def scoring10(row):
        if 50*row['followers_count'] <= row['following_count']:
            return 1
        else:
            return 0
        
    scores['fifty_FriendsFollowersRatio'] = users['public_metrics'].apply(scoring10)
    
    def scoring11(row):
        if 100*row['followers_count'] <= row['following_count']:
            return 1
        else:
            return 0
        
    scores['hundred_FriendsFollowersRatio'] = users['public_metrics'].apply(scoring11)
    
    def scoring12(row):
        if pd.isnull(row) or row == '':
            return 1
        else:
            return 0
        
    scores['has_description'] = users['description'].apply(scoring12)
    
    def scoring13(row):
        if row == 'bot':
            return 1
        else:
            return 0
        
    scores['label'] = users['label'].apply(scoring13)
    
    scores['id'] = users['id']
    scores['split'] = users['split']


    def scoring6(row):
        if pd.isnull(row):
            return 1
        else:
            return 0
        
    scores['id'] = users['id'].apply(scoring6)

    train_set = scores[scores['split'] == 'train']
    del train_set['split']
    test_set = scores[scores['split'] == 'test']
    del test_set['split']
    train_label = train_set['label'].values
    del train_set['label']
    train_set = train_set.values
    test_label = test_set['label'].values
    del test_set['label']
    test_set = test_set.values
    lr_clf = LogisticRegression(penalty='l2', C=1.0, class_weight=None)
    svc_clf = SVC(C=0.5, kernel='rbf', degree=3, gamma='auto')
    std_scale = StandardScaler()
    std_scale.fit(train_set)
    train_set = std_scale.transform(train_set)
    test_set = std_scale.transform(test_set)
    
    piped_obj = Pipeline([('svc', svc_clf)])
    piped_obj.fit(train_set, train_label)
    y_hat = piped_obj.predict(test_set)

    acc = mt.accuracy_score(test_label, y_hat)
    precision = mt.precision_score(test_label, y_hat)
    f1_score = mt.f1_score(test_label, y_hat)
    auc = mt.roc_auc_score(test_label, y_hat)
    recall = mt.recall_score(test_label, y_hat)

    # print(acc, file=f)
    # print(precision, file=f)
    # print(recall, file=f)
    # print(f1_score, file=f)
    # print(auc, dataset, end='\n\n', file=f)
    print(acc)
    print(precision)
    print(recall)
    print(f1_score)
    print(auc, dataset, end='\n\n')
    
    
    acc = mt.accuracy_score(1 - test_label, y_hat)
    precision = mt.precision_score(1 - test_label, y_hat)
    f1_score = mt.f1_score(1 - test_label, y_hat)
    auc = mt.roc_auc_score(1 - test_label, y_hat)
    recall = mt.recall_score(1 - test_label, y_hat)
    
    print(acc)
    print(precision)
    print(recall)
    print(f1_score)
    print(auc, dataset, end='\n\n')