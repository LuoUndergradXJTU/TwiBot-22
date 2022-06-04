# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:19:01 2020
读取数据并对数据做预处理
统计出训练数据中出现频次最多的5k个单词，用这出现最多的5k个单词创建词表（词向量）
对于测试数据，直接用训练数据构建的词表
@author: 
"""
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import csv
import pandas as pd
import numpy as np
import sklearn

glove_path = r"/data2/whr/zqy/glove.twitter.27B.25d.txt"
torch.manual_seed(100)
class DataProcessor(object):
    def read_text(self,is_train_data):
        #读取原始文本数据
        #is_train_data==True表示读取训练数据
        #is_train_data==False表示读取测试数据
        datas = []
        labels = []

        """
        if(is_train_data):
            #训练数据目录
            pos_path = "./datasets/aclImdb/train/pos/" 
            neg_path = "./datasets/aclImdb/train/neg/" 
        else:
            #测试数据目录
            pos_path = "./datasets/aclImdb/test/pos/" 
            neg_path = "./datasets/aclImdb/test/neg/"
        pos_files= os.listdir(pos_path)  #获取文件夹下的所有文件名称
        neg_files = os.listdir(neg_path)
        
        for file_name in pos_files: #遍历文件夹
            file_position = pos_path + file_name
            with open(file_position, "r",encoding='utf-8') as f:  #打开文件
                data = f.read()   #读取文件
                datas.append(data)
                labels.append([1,0]) #正类标签维[1,0]
        
        for file_name in neg_files:
            file_position = neg_path + file_name 
            with open(file_position, "r",encoding='utf-8') as f:
                data = f.read()
                datas.append(data)
                labels.append([0,1]) #负类标签维[0,1]
        """
        if (is_train_data):
            json_file = pd.read_json('./cresci-20151.json')
            csv_file = sklearn.utils.shuffle(json_file)  # 随机打乱
            for line in csv_file.itertuples():
                if (getattr(line, 'split') == "train") & (getattr(line, 'label') == "human"):
                    datas.append(getattr(line, 'text'))
                    labels.append([1, 0])  #负类标签维[1,0]
                elif (getattr(line, 'split') == "train") & (getattr(line, 'label') == "bot"):
                    datas.append(getattr(line, 'text'))
                    labels.append([0,1])  # 正类标签维[0,1]


        else:
            json_file = pd.read_json('./cresci-20151.json')
            csv_file = sklearn.utils.shuffle(json_file)  # 随机打乱
            for line in csv_file.itertuples():
                if (getattr(line, 'split') == "test") & (getattr(line, 'label') == "human"):
                    datas.append(getattr(line, 'text'))
                    labels.append([1, 0])   #负类标签维[1,0]
                elif (getattr(line, 'split') == "test") & (getattr(line, 'label') == "bot"):
                    datas.append(getattr(line, 'text'))
                    labels.append([0,1]) # 正类标签维[0,1]


        return datas, labels

    def word_count(self, datas):
        #统计单词出现的频次，并将其降序排列，得出出现频次最多的单词
        dic = {}
        for data in datas:
            data_list = data.split()
            for word in data_list:
                word = word.lower() #所有单词转化为小写
                if(word in dic):
                    dic[word] += 1
                else:
                    dic[word] = 1
        word_count_sorted = sorted(dic.items(), key=lambda item:item[1], reverse=True)
        return  word_count_sorted

    def word_index(self, datas, vocab_size):
        #创建词表
        word_count_sorted = self.word_count(datas)
        word2index = {}
        #词表中未出现的词
        word2index["<unk>"] = 0
        #句子添加的padding
        word2index["<pad>"] = 1

        #词表的实际大小由词的数量和限定大小决定
        vocab_size = min(len(word_count_sorted), vocab_size)
        for i in range(vocab_size):
            word = word_count_sorted[i][0]
            word2index[word] = i + 2

        return word2index, vocab_size

    def get_datasets(self, vocab_size, embedding_size, max_len):
        #注，由于nn.Embedding每次生成的词嵌入不固定，因此此处同时获取训练数据的词嵌入和测试数据的词嵌入
        #测试数据的词表也用训练数据创建
        train_datas, train_labels = self.read_text(is_train_data=True)

        word2index, vocab_size = self.word_index(train_datas, vocab_size)

        test_datas, test_labels = self.read_text(is_train_data = False)

        train_features = []
###添加glove
        embedding_index = dict()

        with open(glove_path, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:])
                embedding_index[word] = coefs
                line = f.readline()
        embedding_matrix = np.zeros((vocab_size + 2, embedding_size))
        for word, i in word2index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector


        for data in train_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower() #词表中的单词均为小写
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"]) #词表中未出现的词用<unk>代替
                if(len(feature)==max_len): #限制句子的最大长度，超出部分直接截断
                    break
            #对未达到最大长度的句子添加padding
            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            train_features.append(feature)

        test_features = []
        for data in test_datas:
            feature = []
            data_list = data.split()
            for word in data_list:
                word = word.lower() #词表中的单词均为小写
                if word in word2index:
                    feature.append(word2index[word])
                else:
                    feature.append(word2index["<unk>"]) #词表中未出现的词用<unk>代替
                if(len(feature)==max_len): #限制句子的最大长度，超出部分直接截断
                    break
            #对未达到最大长度的句子添加padding
            feature = feature + [word2index["<pad>"]] * (max_len - len(feature))
            test_features.append(feature)

        #将词的index转换成tensor,train_features中数据的维度需要一致，否则会报错
        train_features = torch.LongTensor(train_features)
        train_labels = torch.FloatTensor(train_labels)

        test_features = torch.LongTensor(test_features)
        test_labels = torch.FloatTensor(test_labels)

        #将词转化为embedding
        #词表中有两个特殊的词<unk>和<pad>，所以词表实际大小为vocab_size + 2
        embed = nn.Embedding(vocab_size + 2, embedding_size)
        embed.weight = torch.nn.Parameter(torch.from_numpy(embedding_matrix))
        embed.weight.requires_grad = False
        train_features = embed(train_features)
        test_features = embed(test_features)

        # 将词变为float
        # train_features = torch.FloatTensor(train_features)
        # test_features = torch.FloatTensor(test_features)

        train_features = train_features.float()
        test_features = test_features.float()

        #指定输入特征是否需要计算梯度
        train_features = Variable(train_features, requires_grad=False)
        train_datasets = torch.utils.data.TensorDataset(train_features, train_labels)

        test_features = Variable(test_features, requires_grad=False)
        test_datasets = torch.utils.data.TensorDataset(test_features, test_labels)
        return train_datasets, test_datasets
