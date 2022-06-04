import pandas as pd
import csv
import time

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
    print(len(all_tweet_list))
    filepath = './post_feature.csv'
    df = pd.read_csv('./post_feature.csv')
    df['id']=all_tweet_list
    df.to_csv('./new_post_feature.csv',index=False)