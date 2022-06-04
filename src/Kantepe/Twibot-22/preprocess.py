# user feature generation: Alhosseini
import re
import json
import datetime
from operator import itemgetter
from math import log
import pandas as pd
import numpy as np
import csv
import collections

def read_json_1dict(path):
    """
    读取json，用于一个dict
    :param path: 路径
    :return: 一个dict
    """
    with open(path, "r", encoding='utf-8') as r:
        dic = json.load(r)
    #print(dic)
    return dic
# path="/data2/whr/zqy/327/"
# chooses = ["gilani-2017","cresci-stock-2018","cresci-rtbust-2019","cresci-2015","botometer-feedback-2019"]
chooses = ["Twibot-22"]

for choose in chooses:
    path = "/data2/whr/TwiBot22-baselines/datasets/" + choose + "/"

    dl = pd.read_csv(path + "label.csv")
    ds = pd.read_csv(path + "split.csv")
    ds = ds[ds.split != "support"]
    ds = pd.merge(ds, dl, left_on='id', right_on='id')

    #de = pd.read_csv(path + 'edge.csv')
    #de = de[de.relation == "post"]
    #de = de[de.source_id.isin(ds.id)]

    #dsde = pd.merge(ds, de, left_on='id', right_on='source_id')
    #del dsde["source_id"]
    print( ds)

    ###大表拼接user和id_tweet
    data = pd.read_json(path + "user.json")
    data2 = read_json_1dict("/data2/whr/TwiBot22-baselines/src/twibot22_Botrgcn_feature/id_tweet.json")
    df = pd.DataFrame.from_dict(data2, orient='index')
    df.transpose()
    df.fillna('null', inplace=True)
    df['text'] = [''.join(i + ' ') for i in df.values]
    df = df.drop(df.columns[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]], axis=1)
    #data = pd.concat([data, df], axis=1)
    print(df)
    print(data)

    #data=pd.merge(data,df,left_index=True,right_index=True)
    #data=pd.concat([data, df], axis=1, join='inner')
    data=data.reset_index()
    df=df.reset_index()
    print(data.index)
    print(df.index)
    data = pd.merge(data, df, left_index=True, right_index=True,how='outer')
    #data=data.insert(df)
    print(data)


###
    dsde = pd.merge(ds, data, left_on='id', right_on='id')

    list_temp = dsde['public_metrics'].values.tolist()
    temp = pd.DataFrame(list_temp)
    dsde['followers_count'] = temp['followers_count']
    dsde['following_count'] = temp['following_count']
    dsde['tweet_count'] = temp['tweet_count']
    dsde['listed_count'] = temp['listed_count']
    dsde[' Followers_div_Following'] = temp['followers_count']/(temp['following_count']+1)









    print(dsde)
    print(dsde.columns)


    out2=dsde
    sex_mapping = {"human": 0, "bot": 1}
    out2['label'] = out2['label'].map(sex_mapping)


    def get_user_id(user):
        return user['id']


    def get_username(user):
        return user['username']


    def get_screen_name(user):
        return user['name']


    def has_screen_name(user):
        scr = user['username']
        if scr is None or scr == ' ':
            return False
        else:
            return True


    def has_description(user):
        des = user['description']
        if des is None or des == ' ':
            return False
        else:
            return True


    def get_screen_name_length(user):
        screen_name = get_screen_name(user)
        if screen_name is None:
            return 0
        else:
            return len(screen_name) - 1


    def has_default_profile_image(user):
        if user["profile_image_url"] is None or user["profile_image_url"] == ' ':
            return False
        else:
            return True


    def get_screen_name_entropy(user):
        screen_name = get_screen_name(user)
        counter = collections.Counter(screen_name)
        screen_name_entropy = 0
        for key, cnt in counter.items():
            prob = float(cnt) / len(screen_name)
            screen_name_entropy += -1 * prob * log(prob, 2)
        return screen_name_entropy


    def get_tweet_entropy(user):
        screen_name = user['text']
        counter = collections.Counter(screen_name)
        del counter['"']
        screen_name_entropy = 0
        for key, cnt in counter.items():
            prob = float(cnt) / len(screen_name)
            screen_name_entropy += -1 * prob * log(prob, 2)
        return screen_name_entropy


    def has_location(user):
        loc = user['location']
        des = user['description']
        if loc is None or des == ' ':
            return False
        else:
            return True


    def get_total_tweets(user):
        tweets = user['tweet_count']
        if tweets is None:
            return 0
        else:
            return int(tweets)


    def get_friends(user):
        following_count = user['following_count']
        if following_count is None:
            return 0
        else:
            return int(following_count)


    def get_followers(user):
        followers_count = user['followers_count']
        if followers_count is None:
            return 0
        else:
            return int(followers_count)


    def is_last_status_retweet(user):
        return 0


    def get_hashtags_count(user):
        if has_description(user):
            hashtags = re.findall(r'#\w', user['description'])
            return len(hashtags)
        else:
            return 0


    def get_mentions_count(user):
        if has_description(user):
            mentions = re.findall(r'@', user['description'])
            return len(mentions)
        else:
            return 0


    def get_mentions_count_text(user):
        if has_description(user):
            mentions = re.findall(r'@', user['text'])
            return len(mentions)
        else:
            return 0


    def has_bot_reference(user):
        if has_description(user) or has_screen_name(user):
            botnum = re.findall(r'bot', user["description"])
            if len(botnum) > 0:
                return True
            else:
                return False
        else:
            return False


    def description_has_url(user):
        if has_description(user):
            botnum = re.findall(r'http://', user["description"])
            if len(botnum) > 0:
                return True
            else:
                return False
        else:
            return False


    def get_account_age(user):
        # GMT_FORMAT = '%a %b %d %H:%M:%S %z %Y '
        # midterm-2018 GMT_FORMAT = '%a %b %d %H:%M:%S %Y'
        # Twibot-20 GMT_FORMAT = '%a %b %d %H:%M:%S %z %Y '
        # create_time = user['created_at']
        # user_create_time = datetime.datetime.strptime(create_time, GMT_FORMAT)
        user_create_time = user['created_at']
        d1 = user_create_time.date()
        # d1 = datetime.datetime.strptime(user_create_time, "%Y-%m-%d").date()
        present_time = datetime.datetime.now()
        d2 = present_time.date()
        # d2 = datetime.datetime.strptime(present_time, "%Y-%m-%d").date()
        user_age = (d2 - d1).days
        return (int(user_age))


    def get_avg_tweets(user):
        tweetsnum = get_total_tweets(user)
        age = get_account_age(user)
        avg_tweets = float(tweetsnum / age)
        return (avg_tweets)


    def verified(user):
        try:
            inverified = eval(user['verified'])
        except:
            inverified = 0
        return inverified


    def protected(user):
        try:
            inprotected = eval(user['protected'])
        except:
            inprotected = 0
        return inprotected


    out2['account_age'] = out2.apply(lambda x: get_account_age(x), axis=1)
    out2['avg_tweets'] = out2.apply(lambda x: get_avg_tweets(x), axis=1)


    out2['has_description'] = out2.apply(lambda x: has_description(x), axis=1)
    out2['has_default_profile_image'] = out2.apply(lambda x: has_default_profile_image(x), axis=1)
    out2['verified'] = out2.apply(lambda x: verified(x), axis=1)
    out2['protected'] = out2.apply(lambda x: protected(x), axis=1)
    out2['has_location'] = out2.apply(lambda x: has_location(x), axis=1)
    out2['has_screen_name'] = out2.apply(lambda x: has_screen_name(x), axis=1)

    out2['mentions'] = out2.apply(lambda x: get_mentions_count_text(x), axis=1)
    out2['Tweeting entropy'] = out2.apply(lambda x: get_tweet_entropy(x), axis=1)
    out2['description_has_url'] = out2.apply(lambda x: description_has_url(x), axis=1)

    out2['screen_name_length']=out2.apply(lambda x: get_screen_name_length(x), axis=1)
    out2['screen_name_entropy'] = out2.apply(lambda x: get_screen_name_entropy(x), axis=1)
    out2['hashtags_count'] = out2.apply(lambda x: get_hashtags_count(x), axis=1)
    out2['bot_reference'] = out2.apply(lambda x: has_bot_reference(x), axis=1)







    print(out2.columns)

    out2 = out2.drop(['created_at', 'description', 'entities', 'location', 'name', 'pinned_tweet_id',
                      'profile_image_url', 'url', 'username', 'withheld',"text"], axis=1)

    print(out2.columns)
    print(out2)

    out2.to_json("./" + choose + "second" + ".json")