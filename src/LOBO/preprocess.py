import argparse
import re
import Levenshtein
import json
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--datasets')
args = parser.parse_args()

datasets = args.datasets

if datasets in ('Twibot-20', 'cresci-2017', 'cresci-2015'):
    path = f'../../datasets/{datasets}/'
    print('Loading raw dataset')
    node = json.load(open(path+'node.json'))
    for i in range(len(node)):
        if 't' in node[i]['id']:
            break
    node_user = node[:i]
    node_tweet = node[i:]
    
    tid_to_idx = {}
    for i in range(len(node_tweet)):
        tid_to_idx[node_tweet[i]['id']] = i
    uid_to_idx = {}
    for i in range(len(node_user)):
        uid_to_idx[node_user[i]['id']] = i
    post_dic = {}
    for line in open(path+'edge.csv'):
        source, relation, target = line.strip().split(',')
        if relation == 'post':
            if source not in post_dic.keys():
                post_dic[source] = [target]
            else:
                post_dic[source].append(target)
    print('done')
    train_uid = []
    valid_uid = []
    test_uid = []
    label_dic = {'human':0, 'bot':1}
    uid_label = {}
    for line in open(path+'label.csv'):
        id, label = line.strip().split(',')
        if label == 'human' or label == 'bot':
            uid_label[id] = label_dic[label]

    for line in open(path+'split.csv'):
        id, split = line.strip().split(',')
        if split == 'train':
            train_uid.append(id)
        elif split == 'val':
            valid_uid.append(id)
        elif split == 'test':
            test_uid.append(id)
            
    def convert_from_None_num(feature):
        if feature is None:
            return 0
        else:
            return feature

    def convert_from_None_text(feature):
        if feature is None:
            return ''
        else:
            return feature

    def construct_feature(id):
        user_info = node_user[uid_to_idx[id]]
        
        obj1 = re.compile(r'@(\w*)', re.S)
        
        obj2 = re.compile(r'#([A-Za-z0-9]*)', re.S)

        user_feature = []
        
        hashtags_list = []

        tweet_mentions_list = []

        retweet_mentions_list = []

        tweet_list = []

        retweet_list = []

        tweet_edit_distance = 0

        retweet_edit_distance = 0

        num_urls = 0

        tweet_length = 0

        retweet_length = 0

        if id in post_dic.keys():
            num_total = len(post_dic[id])
        else:
            num_total = 0
        
        user_feature.append(int(id.replace('u', '')))

        num_followers = convert_from_None_num(user_info['public_metrics']['followers_count'])

        num_friends = convert_from_None_num(user_info['public_metrics']['following_count'])

        user_feature.append(num_followers)

        user_feature.append(num_friends)

        if num_followers == 0:
            user_feature.append(0)
        else:
            user_feature.append(num_friends / num_followers)

        user_feature.append(len(convert_from_None_text(user_info['username'])))

        user_feature.append(convert_from_None_num(user_info['public_metrics']['tweet_count']))

        user_feature.append(len(convert_from_None_text(user_info['description'])))

        if num_total > 0:
            for tid in post_dic[id]:

                tidx = tid_to_idx[tid]
                piece = convert_from_None_text(node_tweet[tidx]['text'])

                if piece[0:2] == 'RT':
                    if 0 < len(retweet_list) < 200:
                        for retweet in retweet_list:
                            retweet_edit_distance += Levenshtein.distance(retweet, piece)
                    retweet_list.append(piece)

                    retweet_length += len(piece)

                    text = piece.replace('\n', ' ').split(' ')
                    for token in text:
                        if '#' in token:
                            hashtags_list.append(obj2.findall(token)[0])
                        elif '@' in token:
                            retweet_mentions_list.append(obj1.findall(token)[0])
                        elif 'http' in token:
                            num_urls += 1
                
                else:
                    if 0 < len(tweet_list) < 200:
                        for tweet in tweet_list:
                            tweet_edit_distance += Levenshtein.distance(tweet, piece)
                
                    tweet_list.append(piece)

                    tweet_length += len(piece)
                    
                    text = piece.replace('\n', ' ').split(' ')
                    for token in text:
                        if '#' in token:
                            hashtags_list.append(obj2.findall(token)[0])
                        elif '@' in token:
                            tweet_mentions_list.append(obj1.findall(token)[0])
                        elif 'http' in token:
                            num_urls += 1
            
            num_tweet = len(tweet_list)

            num_retweet = len(retweet_list)

            user_feature.append(len(retweet_mentions_list))

            if num_retweet in [0, 1]:
                user_feature.append(0) 
            else:
                if num_retweet >= 200:
                    user_feature.append(2 * retweet_edit_distance / 200 / 199) 
                else:
                    user_feature.append(2 * retweet_edit_distance / num_retweet / (num_retweet - 1))
                        
            user_feature.append(len(hashtags_list))

            user_feature.append(len(tweet_mentions_list))

            if num_tweet in [0,1]:
                user_feature.append(0)
            else:
                if num_tweet >= 200:
                    user_feature.append(2 * tweet_edit_distance / 200 / 199) 
                else:
                    user_feature.append(2 * tweet_edit_distance / num_tweet / (num_tweet - 1))
                    
            user_feature.append(num_urls)

            user_feature.append(num_urls / num_total)

            user_feature.append(len(set(hashtags_list)))
            
            if len(hashtags_list) == 0:
                user_feature.append(0)
            else:
                user_feature.append(len(set(hashtags_list)) / len(hashtags_list))

            tweet_mentions_list.extend(retweet_mentions_list)
            user_feature.append(len(set(tweet_mentions_list)))

            if num_tweet > 0:
                user_feature.append(tweet_length / num_tweet)
            else:
                user_feature.append(0)
            
            if num_retweet > 0:
                user_feature.append(retweet_length / num_retweet)
            else:
                user_feature.append(0)
            
        else:
            for i in range(12):
                user_feature.append(0)

        return user_feature

    def construct_dataset(mode):
        features = []
        labels = []
        if mode == 'train':
            for uid in tqdm(train_uid):
                features.append(construct_feature(uid))
                labels.append(uid_label[uid])
            return features, labels
        
        elif mode == 'valid':
            for uid in tqdm(valid_uid):
                features.append(construct_feature(uid))
                labels.append(uid_label[uid])
            return features, labels

        elif mode == 'test':
            for uid in tqdm(test_uid):
                features.append(construct_feature(uid))
                labels.append(uid_label[uid])
            return features, labels
        
    print('Constructing training set')
    train_features, train_labels = construct_dataset('train')
    print('Constructing validation set')
    valid_features, valid_labels = construct_dataset('valid')
    print('Constructing test set')
    test_features, test_labels = construct_dataset('test')

    print('Save processed data to corresponding directory')
    train_features, train_labels = np.array(train_features), np.array(train_labels)
    valid_features, valid_labels = np.array(valid_features), np.array(valid_labels)
    test_features, test_labels = np.array(test_features), np.array(test_labels)
    
    np.savetxt(f'{datasets}/train_features.txt', np.array(train_features))
    np.savetxt(f'{datasets}/train_labels.txt', np.array(train_labels))
    np.savetxt(f'{datasets}/valid_features.txt', np.array(valid_features))
    np.savetxt(f'{datasets}/valid_labels.txt', np.array(valid_labels))
    np.savetxt(f'{datasets}/test_features.txt', np.array(test_features))
    np.savetxt(f'{datasets}/test_labels.txt', np.array(test_labels))
    print('done')


elif datasets == 'Twibot-22':
    
    print('You can view our source code for preprocessing in Twibot-22/semi-processed/preprocess.py')
    
    print('Loading semi-processed datasets...')
    path_raw = '../../datasets/Twibot-22/'
    uid_to_info = {}
    user_node = json.load(open(path_raw+'user.json'))
    for i in range(len(user_node)):
        uid_to_info[user_node[i]['id']] = user_node[i]

    path = './Twibot-22/semi-processed_dataset/'
    feature = np.concatenate([np.loadtxt(path+'tweet_'+str(i)+'.txt') for i in range(9)])
    feature = feature.tolist()
    user = []
    for i in range(9):
        user.extend(json.load(open(path+'u_list_'+str(i)+'.json')))
        
    user_feature = {}
    for i in range(len(user)):
        if user[i] not in user_feature.keys():
            user_feature[user[i]] = [feature[i]]
        else:
            user_feature[user[i]].append(feature[i])
            
    def convert_from_None_num(feature):
        if feature is None:
            return 0
        else:
            return feature

    def convert_from_None_text(feature):
        if feature is None:
            return ''
        else:
            return feature
        
    train_uid = []
    valid_uid = []
    test_uid = []
    label_dic = {'human':0, 'bot':1}
    uid_label = {}
    for line in open(path_raw+'label.csv'):
        id, label = line.strip().split(',')
        if label == 'human' or label == 'bot':
            uid_label[id] = label_dic[label]

    for line in open(path_raw+'split.csv'):
        id, split = line.strip().split(',')
        if split == 'train':
            train_uid.append(id)
        elif split == 'val':
            valid_uid.append(id)
        elif split == 'test':
            test_uid.append(id)
            
    def user_without_tweet(id):
        user_feature = []
        user_info = uid_to_info[id]
        user_feature.append(int(id.replace('u', ''))) #1

        num_followers = convert_from_None_num(user_info['public_metrics']['followers_count'])

        num_friends = convert_from_None_num(user_info['public_metrics']['following_count'])

        user_feature.append(num_followers) #2

        user_feature.append(num_friends) #3

        if num_followers == 0:
            user_feature.append(0) #4
        else:
            user_feature.append(num_friends / num_followers) #4

        user_feature.append(len(convert_from_None_text(user_info['username']))) #5

        user_feature.append(convert_from_None_num(user_info['public_metrics']['tweet_count'])) #6

        user_feature.append(len(convert_from_None_text(user_info['description']))) #7
        
        for i in range(19):
            user_feature.append(0)
        
        return user_feature
    def construct_dataset(mode):
        features = []
        labels = []
        if mode == 'train':
            for uid in tqdm(train_uid):
                if uid not in user_feature.keys():
                    features.append(user_without_tweet(uid))
                else:
                    features.append(user_feature[uid][0])
                labels.append(uid_label[uid])
            return features, labels
        
        elif mode == 'valid':
            for uid in tqdm(valid_uid):
                if uid not in user_feature.keys():
                    features.append(user_without_tweet(uid))
                else:
                    features.append(user_feature[uid][0])
                labels.append(uid_label[uid])
            return features, labels

        elif mode == 'test':
            for uid in tqdm(test_uid):
                if uid not in user_feature.keys():
                    features.append(user_without_tweet(uid))
                else:
                    features.append(user_feature[uid][0])
                labels.append(uid_label[uid])
            return features, labels
        
    print('done')
    
    print('Constructing training set')
    train_features, train_labels = construct_dataset('train')
    print('Constructing validation set')
    valid_features, valid_labels = construct_dataset('valid')
    print('Constructing test set')
    test_features, test_labels = construct_dataset('test')

    print('Save processed data to corresponding directory')
    train_features, train_labels = np.array(train_features), np.array(train_labels)
    valid_features, valid_labels = np.array(valid_features), np.array(valid_labels)
    test_features, test_labels = np.array(test_features), np.array(test_labels)
    
    np.savetxt(f'{datasets}/train_features.txt', np.array(train_features))
    np.savetxt(f'{datasets}/train_labels.txt', np.array(train_labels))
    np.savetxt(f'{datasets}/valid_features.txt', np.array(valid_features))
    np.savetxt(f'{datasets}/valid_labels.txt', np.array(valid_labels))
    np.savetxt(f'{datasets}/test_features.txt', np.array(test_features))
    np.savetxt(f'{datasets}/test_labels.txt', np.array(test_labels))
    print('done')
