import pandas as pd
import time
import ijson
import content_feature_vec as cf
import csv

def get_content_vector(tweets_text):
    content_vector = {
        "characters":cf.count_characters(tweets_text),
        "hashtags":cf.count_hashtags(tweets_text),
        "words":cf.count_words(tweets_text),
        "hashtags_to_words":cf.hastags_words(tweets_text),
        "mentions":cf.count_mentions(tweets_text),
        "numeric_chacacters":cf.count_numeric_chars(tweets_text),
        "symbols":cf.count_symbols(tweets_text),
        "urls":cf.count_urls(tweets_text),
        "urls_to_words":cf.urls_words(tweets_text),
    }
    return content_vector

def process_posts(posts):
    post = posts.replace("'","").strip("[").strip("]")
    post = post.split(", ")
    post = post[:20]
    return post
    
def get_all_tweet():
    df = pd.read_csv('./user_post.csv')
    all_tweet_list = []
    for index, row in df.iterrows():
        tweets_list = process_posts(row['posts'])
        for item in tweets_list:
            all_tweet_list.append(item)
    return all_tweet_list
    
def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic])

if __name__ == '__main__':
    start = time.time()
    all_tweet_list = get_all_tweet()
    filepath = './post_feature.csv'
    with open("/data2/whr/czl/TwiBot22-baselines/datasets/Twibot-20/node.json") as f:
        obj = ijson.items(f, 'item')
        while True:
            try:
                data = obj.__next__()
                text = ''
                if data['id'][0] == 'u':
                    pass
                else:
                    if data['id'] in all_tweet_list:
                        text = data['text']
                        content_feature = get_content_vector(text)
                        header = list(content_feature.keys())
                        to_csv(header, content_feature, filepath)
            except StopIteration as e:
                break
    print(time.time() - start)
        