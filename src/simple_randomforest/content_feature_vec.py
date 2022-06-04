import re

# user_to_post.csv获取tweet列表
# node.json获取post推文
# post是个字典
# maybe item = item.replace('\n',' ')

def count_characters(text):
    text = text.replace('\n',' ')
    return len(text)

def count_hashtags(text):
    hashtags = re.findall(r'#\w', text)
    return len(hashtags)

def count_words(text):
    text = text.replace('\n',' ')
    return text.count(' ')

def hastags_words(text):
    hastags = count_hashtags(text)
    words = count_words(text)
    if words == 0:
        return 0
    else:
        return hastags/words

def count_mentions(text):
    return text.count('@')

# maybe '\d'
def count_numeric_chars(text):
    numbers = re.findall(r'\d+', text)
    return len(numbers)

def count_symbols(text):
    cnt = 0
    text = text.lower()
    for i in text:
        if (i>='0' and i<='9') or (i>='a' and i<='z'):
            continue
        else:
            cnt += 1
    return cnt


def count_urls(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text) 
    return len(urls)

def urls_words(text):
    urls = count_urls(text)
    words = count_words(text)
    if words == 0:
        return 0
    else:
        return urls/words







