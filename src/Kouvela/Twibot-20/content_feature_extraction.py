#-*- coding : utf-8-*-

import os
import csv
import pandas as pd
import time
import content_features as cf
from tqdm import tqdm

    
def caculate_feature(all_text, all_text_like, all_text_retweet):
    vec = {
        "id":'u',
        "characters":cf.count_characters(all_text),
        "hashtags":cf.count_hashtags(all_text),
        "words":cf.count_words(all_text),
        "hashtags_to_words":cf.hastags_words(all_text),
        "mentions":cf.count_mentions(all_text),
        "numeric_chacacters":cf.count_numeric_chars(all_text),
        "symbols":cf.count_symbols(all_text),
        "urls":cf.count_urls(all_text),
        "urls_to_words":cf.urls_words(all_text),
    }
    return vec
    

def process_posts(posts):
    post = eval(posts)
    post = post[:20]
    return post

def get_all_tid(text):
    tid = text['id']
    tid = list(tid)
    return tid

def to_csv(header, dic, filename):
    with open(filename, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerows([dic])


if __name__ == '__main__':
    start = time.time()
    filepath = './content_feature.csv'
    dir_ = './small_split_dataset/tweet{}.csv'
    flag = []

    var = locals()
    ids = locals()
    number = len(os.listdir('./small_split_dataset'))
    for i in range(number):
        var['t{}'.format(i)] = pd.read_csv(dir_.format(i), lineterminator="\n")
        ids['t{}_id'.format(i)] = get_all_tid(eval('t{}'.format(i)))
        '''if i%8 == 0:
            print('t{}:'.format(i), 'time:{}'.format(time.time()-start), 'length:{}'.format(len(eval('t{}'.format(i)))))'''
    
    # Calculate the max user_id and user_min id of each tweets file
    for i in range(number):
        flag.append(eval('t{}_id[0]'.format(i)))
        flag.append(eval('t{}_id[-1]'.format(i)))
    # print(flag)
    flag_lens = len(flag)
    
    the_header = ["id","characters", "hashtags", "words", "hashtags_to_words", "mentions",
                  "numeric_chacacters", "symbols", "urls", "urls_to_words"]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(the_header)

    df = pd.read_csv('./user_to_post.csv')
    print('reading df finish', time.time()-start)
    row_num = df.shape[0]
    
    for index, row in tqdm(df.iterrows(), total=row_num):
        user = row['id']
        tweets_list = process_posts(row['post'])
        texts = []
        like = []
        retweet = []
        for item in tweets_list:
            for k in range(0, flag_lens, 2):
                if item>= flag[k] and item<=flag[k+1]:
                    i = int(k/2)
                    ti = eval('t{}'.format(i))
                    ti_id = eval('t{}_id'.format(i))
                    idx = ti_id.index(item)
                    msg = ti.loc[idx]
                    texts.append(str(msg['text']))
                    break
                    
        vec = caculate_feature(texts, like, retweet)
        vec['id'] = user
        to_csv(the_header, vec, filepath)
        
    end = time.time()
    print(end - start)

