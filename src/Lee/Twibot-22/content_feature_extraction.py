import pandas as pd
import time
import csv
import spacy
from tqdm import tqdm
import content_features as cf
import os

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

def get_content_vector(tweets):
    links = cf.count_urls(tweets)
    mentions = cf.count_mentions(tweets)
    content_vector = {
        "id":'u',
        "links_ratio":links[0],
        "unique_links_ratio":links[1],
        "mention_ratio":mentions[0],
        "unique_mention_ratio":mentions[1],
        "compression_ratio":cf.zip_ratio(tweets),
        "similarity":cf.count_similarity(nlp,tweets), 
    }
    return content_vector

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
        '''if i%11 == 0:
            print('t{}:'.format(i), 'time cost:{}'.format(time.time()-start), 'length:{}'.format(len(eval('t{}'.format(i)))))'''
        
    for i in range(number):
        flag.append(eval('t{}_id[0]'.format(i)))
        flag.append(eval('t{}_id[-1]'.format(i)))
    #print(flag)
    flag_lens = len(flag)

    the_header = ["id","links_ratio", "unique_links_ratio", "mention_ratio", "unique_mention_ratio", "compression_ratio", "similarity"]
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(the_header)
        
    df = pd.read_csv("./user_to_post.csv")
    print('reading df finish', time.time()-start)
    row_num = df.shape[0]
    
    for index, row in tqdm(df.iterrows(), total=row_num):
        user = row['id']
        tweets_list = process_posts(row['post'])
        texts = []
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
                    
        vec = get_content_vector(texts)
        vec['id'] = user
        to_csv(the_header, vec, filepath)
    
    
    print(time.time() - start)
        