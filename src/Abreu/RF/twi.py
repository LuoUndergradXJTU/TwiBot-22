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
from sklearn.ensemble import RandomForestClassifier
# datasets = ['botometer-feedback-2019', 'gilani-2017', 'Twibot-22']
for i in range(1):
  datasets = ['midterm-2018']
  #datasets = ['cresci-2015']
  path1 = '../../../datasets/'
  
  
  for dataset in tqdm(datasets):
  
      if dataset == 'Twibot-22':
          nodes = pd.read_json(path1 + dataset + '/user.json')
          labels = pd.read_csv(path1 + dataset+  '/label.csv')
          split = pd.read_csv(path1 + dataset + '/split.csv')
      else:
          nodes = pd.read_json(path1 + dataset+  '/node.json')
          labels = pd.read_csv(path1 + dataset + '/label.csv')
          split = pd.read_csv(path1 + dataset + '/split.csv')
  
      users = pd.merge(labels, nodes)
      users = pd.merge(users, split)
  
      scores = pd.DataFrame()
  
      #
      def scoring2(row):
          if row['listed_count'] is None:
              return 0
          else:
              return int(row['listed_count'])
  
  
      scores['listed_count'] = users['public_metrics'].apply(scoring2)
  
      #
      def scoring3(row):
          if row['followers_count'] is None:
              return 0
          else:
              return int(row['followers_count'])
  
  
      scores['followers'] = users['public_metrics'].apply(scoring3)
      #
  
      #
      def scoring7(row):
          if row['tweet_count'] is None:
              return 0
          else:
              return int(row['tweet_count'])
  
  
      scores['statuses_count'] = users['public_metrics'].apply(scoring7)
      #
      #
      def scoring8(row):
          if row['following_count'] is None:
              return 0
          else:
              return int(row['following_count'])
      #
      #
      scores['num_followings'] = users['public_metrics'].apply(scoring8)
  
    #  def scoring11(row):
    #      if  row['favourites_count'] is None:
     #         return 0
    #      else:
     #         return int(row['favourites_count'])
  
  
     # scores['favourites_count'] = users['public_metrics'].apply(scoring11)
      #
  
      def scoring13(row):
        if row == 'bot':
            return 1
        else:
            return 0
  
  
      scores['label'] = users['label'].apply(scoring13)
  
      scores['id'] = users['id']
      scores['split'] = users['split']
  
      train_set = scores[scores['split'] == 'train']
      del train_set['split']
      test_set = scores[scores['split'] == 'test']
      del test_set['split']
      train_label = train_set['label'].values
      del train_set['label']
      test_label = test_set['label'].values
      del test_set['label']
      del train_set['id']
      del test_set['id']
      train_set = train_set.values
      test_set = test_set.values
      Random_Forest = RandomForestClassifier(n_estimators=100)
      Random_Forest.fit(train_set, train_label)
  
      y_hat = Random_Forest.predict(test_set)
  
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
      final = []
      final.append(acc)
      final.append(precision)
      final.append(recall)
      final.append(f1_score)
      final.append(auc)
      final.append(dataset)
      
      acc = mt.accuracy_score(1 - test_label, y_hat)
      precision = mt.precision_score(1 - test_label, y_hat)
      f1_score = mt.f1_score(1 - test_label, y_hat)
      auc = mt.roc_auc_score(1 - test_label, y_hat)
      recall = mt.recall_score(1 - test_label, y_hat)
      final.append(acc)
      final.append(precision)
      final.append(recall)
      final.append(f1_score)
      final.append(auc)
      final.append(dataset)
      with open('./'+dataset+'.txt', 'a') as f:
        f.write(str(final))
        f.write('\r\n')
