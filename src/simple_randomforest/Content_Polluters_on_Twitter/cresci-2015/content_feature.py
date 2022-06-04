import re
import os
import spacy
import zipfile

#nlp = spacy.load('en_core_web_lg')

def count_urls(tweets):
    total = len(tweets)
    unique =[]
    for text in tweets:
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        unique += urls
    count = len(unique)
    unique_count = len(set(unique))
    return count/total, unique_count/total


def count_mentions(tweets):
    total = len(tweets)
    unique =[]
    for text in tweets:
        mentions = re.findall(r'@(?:[a-zA-Z]|[0-9]|[_+])+',text)
        unique += mentions
    count = len(unique)
    unique_count = len(set(unique))
    return count/total, unique_count/total


def count_similarity(nlp, tweets):
    num = len(tweets)
    if num == 1:
        return 1
    else:
        for i in range(num):
            tweets[i] = nlp(tweets[i])
        
        similarity = []
        for i in range(num):
            for j in range(i+1, num):
                similarity.append(tweets[i].similarity(tweets[j]))

        sim_sum = sum(similarity)
        pairs = num*(num-1)/2
        avg = sim_sum / pairs
        return avg
    
def zip_ratio(tweets):
    filepath = "./temp_tweets.txt"
    zip_filepath = './temp_tweets.txt.zip'
    with open(filepath, "w") as f:
        for text in tweets:
            f.write(text+'\n')
    unzip_size = os.path.getsize(filepath)
    
    if not os.path.exists(filepath):
        print('File not exist')
    else:
        zips = zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED)
        zips.write(filepath)
        
    zip_size = os.path.getsize(zip_filepath)
    return unzip_size/zip_size







