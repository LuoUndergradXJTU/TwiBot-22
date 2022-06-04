import re
import Levenshtein
import json
import numpy as np
import datetime
import ijson
path = '../czl/TwiBot22-baselines/datasets/Twibot-22/'
uid_to_info = {}
user_node = json.load(open(path+'user.json'))
for i in range(len(user_node)):
    uid_to_info[user_node[i]['id']] = user_node[i]


label_dic = {'human':0, 'bot':1}
uid_label = {}
for line in open(path+'label.csv'):
    id, label = line.strip().split(',')
    if label == 'human' or label == 'bot':
        uid_label[id] = label_dic[label]
        
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
    user_info = uid_to_info[id]

    user_feature = []
    
    hashtags_list = []

    tweet_mentions_list = []
    
    num_retweet_mentions = 0

    tweet_list = []
    
    retweet_list = []
    
    source_list = []
    
    days_active_list = []
    
    seconds_active_list = []

    num_geo = 0

    tweet_edit_distance = 0
    
    retweet_edit_distance = 0

    num_urls = 0

    tweet_length = 0
    
    retweet_length = 0
    
    num_favorites = 0

    
    account_creation_time = datetime.datetime.strptime(user_info['created_at'].replace('+00:00', ''), "%Y-%m-%d %H:%M:%S")
    
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

    
    for piece in post_dic[id]:

        if piece['geo'] is not None:
            num_geo += 1
        
        if piece['entities'] is not None:
            if 'hashtags' in piece['entities'].keys():
                for hashtag in piece['entities']['hashtags']:
                    if 'tag' in hashtag.keys():
                        hashtags_list.append(hashtag['tag'])
                    elif 'text' in hashtag.keys():
                        hashtags_list.append(hashtag['text'])

            if 'urls' in piece['entities'].keys():
                num_urls += len(piece['entities']['urls'])
            if 'media' in piece['entities'].keys():
                num_urls += len(piece['entities']['media'][0]['url'])
        
        num_favorites += piece['public_metrics']['like_count']
        
        obj = re.compile(r'rel="nofollow">(.*?)</a>', re.S)
        if piece['source'] is not None:
            if 'rel="nofollow">' in piece['source']:
                source_list.append(obj.findall(piece['source'])[0])
            else:
                source_list.append(piece['source'])
                
        
        tweet_creation_time = datetime.datetime.strptime(piece['created_at'].replace('+00:00', ''), "%Y-%m-%d %H:%M:%S")
        days_active = (tweet_creation_time - account_creation_time).days
        seconds_active = days_active * 86400 + (tweet_creation_time - account_creation_time).seconds
        days_active_list.append(days_active)
        seconds_active_list.append(seconds_active)
        
        if piece['text'][0:2] == 'RT':

            if piece['entities'] is not None:
                if 'user_mentions' in piece['entities'].keys():
                    num_retweet_mentions += len(piece['entities']['user_mentions'])
                elif 'mentions' in piece['entities'].keys():
                    num_retweet_mentions += len(piece['entities']['mentions'])
            retweet_length += len(piece['text'])
            if 0 < len(retweet_list) < 200:
                for retweet in retweet_list:
                    retweet_edit_distance += Levenshtein.distance(retweet, piece['text'])
                    
            retweet_list.append(piece['text'])
        else:
            if piece['entities'] is not None:
                if 'user_mentions' in piece['entities'].keys():
                    for mention in piece['entities']['user_mentions']:
                        tweet_mentions_list.append(mention['screen_name'])
                elif 'mentions' in piece['entities'].keys():
                    for mention in piece['entities']['mentions']:
                        tweet_mentions_list.append(mention['username'])

            if 0 < len(tweet_list) < 200:
                for tweet in tweet_list:
                    tweet_edit_distance += Levenshtein.distance(tweet, piece['text'])
        
            tweet_list.append(piece['text'])

            tweet_length += len(piece['text'])

        num_tweet = len(tweet_list)
        
        num_retweet = len(retweet_list)
            
        num_total = len(post_dic[id])
      
    user_feature.append(num_geo) #8 s
    
    user_feature.append(num_geo / num_total) #9 
    
    user_feature.append(num_retweet_mentions) #10 s
    
    if num_retweet in [0, 1]:
        user_feature.append(0) #11
    else:
        if num_retweet >= 200:
            user_feature.append(2 * retweet_edit_distance / 200 / 199) 
        else:
            user_feature.append(2 * retweet_edit_distance / num_retweet / (num_retweet - 1))  #11 ws
    
    user_feature.append(len(hashtags_list)) #12 s
    
    user_feature.append(len(tweet_mentions_list)) #13 s
    
    if num_tweet in [0, 1]: 
        user_feature.append(0) #14
    else:
        if num_tweet >= 200:
            user_feature.append(2 * tweet_edit_distance / 200 / 199)
        else:
            user_feature.append(2 * tweet_edit_distance / num_tweet / (num_tweet - 1))  #14 ws

    
    
    user_feature.append(num_urls) #15 s

    user_feature.append(num_urls / num_total) #16 
    
    user_feature.append(num_favorites) #17 s
    
    user_feature.append(num_favorites / num_total) #18 
    
    user_feature.append(len(set(source_list))) #19 g

    user_feature.append(len(set(hashtags_list))) #20 g
     
    if len(hashtags_list) == 0:
        user_feature.append(0) #21 
    else:
        user_feature.append(len(set(hashtags_list)) / len(hashtags_list)) #21 ws

    user_feature.append(len(set(tweet_mentions_list))) #22 g
    if num_tweet == 0:
        user_feature.append(0)
    else:
        user_feature.append(tweet_length / num_tweet) #23 ws
    
    if num_retweet == 0:
        user_feature.append(0)
    else:
        user_feature.append(retweet_length / num_retweet) #24 ws
    
    if max(days_active_list) > 1095:
        user_feature.append(1095) #25
    else:
        user_feature.append(max(days_active_list)) #25 g
    
    if max(seconds_active_list) > 2592000:
        user_feature.append(2592000) #26
    else:
        user_feature.append(max(seconds_active_list)) #26 g
    
    return user_feature

for tweet_idx in range(9):
    post_dic = json.load(open('post_dic.json'))
    with open(path+'tweet_'+str(tweet_idx)+'.json') as f:
        iter = ijson.items(f, 'item')
        for item in iter:
            id = 'u'+str(item['author_id'])
            if id in uid_label.keys():
                if id not in post_dic.keys():
                    post_dic[id] = [item]
                else:
                    post_dic[id].append(item)
    user_list = list(post_dic.keys())

    feature = []
    u_list = []
    for i in range(len(user_list)-1):
        id = user_list[i]
        u_list.append(id)
        feature.append(construct_feature(id))
        post_dic.pop(id)
    np.savetxt('tweet_'+str(tweet_idx)+'.txt', np.array(feature))
    json.dump(post_dic, open('post_dic.json', 'w'), indent=4)
    json.dump(u_list, open('u_list_'+str(tweet_idx)+'.json', 'w'), indent=4)