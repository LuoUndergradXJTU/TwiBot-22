#-*- coding : utf-8-*-

import csv
import pandas as pd
import time

    
def caculate_feature(df ,length, bias):
    data = df.loc[bias:bias+length-1]
    characters = sum(list(data["characters"]))
    hashtags = sum(list(data["hashtags"]))
    words = sum(list(data["words"]))
    mentions = sum(list(data["mentions"]))
    numeric_chacacters = sum(list(data["numeric_chacacters"]))
    symbols = sum(list(data["symbols"]))
    urls = sum(list(data["urls"]))
    if words == 0:
        hashtags_to_words = 0
        urls_to_words = 0
    else:
        hashtags_to_words = hashtags/words
        urls_to_words = urls/words
    vec = {
        "characters":characters,
        "hashtags":hashtags,
        "words":words,
        "hashtags_to_words":hashtags_to_words,
        "mentions":mentions,
        "numeric_chacacters":numeric_chacacters,
        "symbols":symbols,
        "urls":urls,
        "urls_to_words":urls_to_words,
    }
    return vec
    

def process_posts(posts):
    post = posts.replace("'","").strip("[").strip("]")
    post = post.split(", ")
    post = post[:20]
    return post


def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic])


if __name__ == '__main__':
    start = time.time()
    filepath = './content_feature.csv'
    df = pd.read_csv('./user_post.csv')
    df2 = pd.read_csv('./post_feature.csv')
    bias = 0
    for index, row in df.iterrows():
        user = row['id']
        tweets_list = process_posts(row['posts'])
        length = len(tweets_list)
        vec = caculate_feature(df2, length, bias)
        vec['id'] = user
        bias = bias+length
        header = list(vec.keys())
        to_csv(header, vec, filepath)

    end = time.time()
    print(end - start)

