import user_features as uf
import csv
import ijson
import pandas as pd
import time


def get_user_vector(user):
    user_vector = {
        "id": uf.get_user_id(user),
        "screen_name_length": uf.get_screen_name_length(user),
        "description_length": uf.get_description_length(user),
        "account_longvity": uf.get_longevity(user),
        "following_count": uf.get_following(user),
        "followers_count": uf.get_followers(user),
        "following_to_followers": uf.get_following_to_followers(user),
        "tweets_count": uf.get_tweets(user),
        "tweets_count_per_day": uf.get_tweets_per_day(user),
    }

    return user_vector


def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic]) 


if __name__ == '__main__':
    start = time.time()
    filepath = './user_feature.csv'
    df = pd.read_csv("./user_post.csv")
    print(len(df))
    user_final = list(df['id'])
    the_header = ["id", "screen_name_length", "description_length", 
                  "account_longvity", "following_count", "followers_count", 
                  "following_to_followers", "tweets_count", "tweets_count_per_day"]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(the_header)
        
    with open('/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2017/node.json') as f:
        obj = ijson.items(f, 'item')
        while True:
            try:
                user = obj.__next__()
                #print(user)
                if user['id'][0] == 'u':
                    if user['id'] in user_final: 
                        user_feature = get_user_vector(user)
                        header = list(user_feature.keys())
                        to_csv(header, user_feature, filepath)
                else:
                    break
            except StopIteration as e:
                break
    end = time.time()
    print(end-start)

