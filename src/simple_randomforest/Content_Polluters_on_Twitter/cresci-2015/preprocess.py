import pandas as pd
import ijson
import csv
import time

def process_posts(posts):
    post = posts.replace("'","").strip("[").strip("]")
    post = post.split(", ")
    post = post[:200]
    return post
    
def get_all_tweet():
    df = pd.read_csv('./user_post.csv')
    all_tweet_list = []
    for index, row in df.iterrows():
        tweets_list = process_posts(row['post'])
        all_tweet_list += tweets_list
    return all_tweet_list

if __name__ == '__main__':
    start = time.time()
    all_tweet_list = get_all_tweet()
    need_tid = []
    print(len(all_tweet_list))
    with open("/data2/whr/czl/TwiBot22-baselines/datasets/cresci-2015/node.json") as f:
        obj = ijson.items(f, 'item')
        while True:
            try:
                data = obj.__next__()
                if data['id'][0] == 'u':
                    continue
                if data['id'] in all_tweet_list:
                    need_tid.append(data)
            except StopIteration as e:
                break
    
    with open('./new_node.json', 'w') as f:
        json.dump(need_tid,f)
    
    print(time.time() - start)
    
    
    
    