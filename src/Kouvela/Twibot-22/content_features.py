import re


def count_characters(all_text):
    cnt = 0
    for item in all_text:
        item = item.replace('\n',' ')
        cnt += len(item)
    return cnt

def count_hashtags(all_text):
    cnt = 0
    for item in all_text:
        hashtags = re.findall(r'#\w', item)
        cnt += len(hashtags)
    return cnt

def count_words(all_text):
    cnt = 0
    for item in all_text:
        item = item.replace('\n', ' ')
        cnt += item.count(' ')
    return cnt

def hastags_words(all_text):
    hastags = count_hashtags(all_text)
    words = count_words(all_text)
    if words == 0:
        return 0
    else:
        return hastags/words

def count_mentions(all_text):
    cnt = 0
    for item in all_text:
        cnt += item.count('@')
    return cnt

# maybe '\d'
def count_numeric_chars(all_text):
    cnt = 0
    for item in all_text:
        numbers = re.findall(r'\d+', item)
        cnt += len(numbers)
    return cnt

def count_symbols(all_text):
    cnt = 0
    for item in all_text:
        item = item.lower()
        for i in item:
            if (i>='0' and i<='9') or (i>='a' and i<='z'):
                continue
            else:
                cnt += 1
    return cnt


def count_urls(all_text):
    cnt = 0
    for item in all_text:
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', item)
        cnt += len(urls)
    return cnt

def urls_words(all_text):
    urls = count_urls(all_text)
    words = count_words(all_text)
    if words == 0:
        return 0
    else:
        return urls/words
        
def times_like(all_text_like):
    length = len(all_text_like)
    for i in range(length):
        if all_text_like[i] is None:
            all_text_like[i] = 0
            
    cnt = sum(all_text_like)
    return cnt
      
def times_retweeted(all_text_retweet):
    length = len(all_text_retweet)
    for i in range(length):
        if all_text_retweet[i] is None:
            all_text_retweet[i] = 0
            
    cnt = sum(all_text_retweet)
    return cnt

