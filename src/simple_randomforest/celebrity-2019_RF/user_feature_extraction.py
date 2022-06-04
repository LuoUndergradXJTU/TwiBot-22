import user_features as uf
import csv
import time
import pandas as pd
import json


def get_user_vector(user):
    user_vector = {
        "user_id": uf.get_user_id(user),
        "profile_description": uf.has_description(user),
        "profile_location": uf.has_location(user),
        "profile_url": uf.has_url(user),
        "verified": uf.is_verified(user),
        "bot_word_in_name": uf.has_bot_word_in_username(user),
        "bot_word_in_screen_name": uf.has_bot_word_in_screen_name(user),
        "bot_word_in_description": uf.has_bot_word_in_description(user),
        "username_length":uf.get_username_length(user),
        "screen_name_length": uf.get_screen_name_length(user),
        "description_length": uf.get_description_length(user),
        "followees_count": uf.get_followees(user),
        "followers_count": uf.get_followers(user),
        "followers_to_followees": uf.get_followers_followees(user),
        "tweets_count": uf.get_tweets(user),
        "listed_count": uf.get_lists(user),
        "numerics_in_username_count": uf.get_number_count_in_username(user),
        "numerics_in_screen_name_count": uf.get_number_count_in_screen_name(user),
        "hashtags_in_username": uf.hashtags_count_in_username(user),
        "hashtags_in_description": uf.hashtags_count_in_description(user),
        "urls_in_description": uf.urls_count_in_description(user),
    }

    return user_vector

def get_user_list():
    data = pd.read_csv('/data2/whr/czl/TwiBot22-baselines/datasets/celebrity-2019/label.csv')
    length = len(data)
    all_user = data.loc[0:length-1]
    all_user = list(all_user['id'])
    return all_user

def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic])


if __name__ == '__main__':
    start = time.time()
    filepath = './user_feature.csv'
    user_list = get_user_list()
    node = open('/data2/whr/czl/TwiBot22-baselines/datasets/celebrity-2019/node.json', "rb")
    f = json.load(node)
    for user in f:
        if user['id'][0] == 'u':
            if user['id'] in user_list:
                user_feature = get_user_vector(user)
                header = list(user_feature.keys())
                to_csv(header, user_feature, filepath)
        else:
            break
    end = time.time()
    print(end-start)

