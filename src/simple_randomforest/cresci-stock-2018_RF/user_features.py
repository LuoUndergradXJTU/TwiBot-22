import re


def get_user_id(user):
    return str(user['id'])


def get_username(user):
    return str(user['username'])


def get_screen_name(user):
    return str(user['name'])


def has_description(user):
    des = str(user['description'])
    if des is None or des == ' ':
        return False
    else:
        return True


def has_location(user):
    location = str(user['location'])
    if location is None or location == ' ':
        return False
    else:
        return True


def has_url(user):
    url = str(user['url'])
    if url is None:
        return False
    else:
        return True


def is_verified(user):
    if user['verified'] == 'True ':
        return True
    else:
        return False


def has_bot_word_in_description(user):
    if has_description(user):
        matchObj = re.search('bot', str(user['description']), flags=re.IGNORECASE)
        if matchObj:
            return True
        else:
            return False
    else:
        return False


def has_bot_word_in_screen_name(user):
    screen_name = get_screen_name(user)
    if screen_name is None or screen_name == ' ':
        return False
    else:
        matchObj = re.search('bot', str(screen_name), flags=re.IGNORECASE)
        if matchObj:
            return True
        else:
            return False


def has_bot_word_in_username(user):
    username = get_username(user)
    if username is None or username == ' ':
        return False
    else:
        matchObj = re.search('bot', str(username), flags=re.IGNORECASE)
        if matchObj:
            return True
        else:
            return False


def get_screen_name_length(user):
    screen_name = get_screen_name(user)
    if screen_name is None:
        return 0
    else:
        return len(str(screen_name))


def get_username_length(user):
    username = get_username(user)
    if username is None:
        return 0
    else:
        return len(str(username))


def get_description_length(user):
    if has_description(user):
        return len(str(user['description']))
    else:
        return 0


def get_followees(user):
    following_count = user['public_metrics']['following_count']
    if following_count is None:
        return 0
    else:
        return int(following_count)


def get_followers(user):
    followers_count = user['public_metrics']['followers_count']
    if followers_count is None:
        return 0
    else:
        return int(followers_count)


def get_followers_followees(user):
    followers_count = str(user['public_metrics']['followers_count'])
    following_count = str(user['public_metrics']['following_count'])
    if following_count is None or following_count == '0':
        return 0.0
    else:
        return int(followers_count) / int(following_count)


def get_tweets(user):
    tweets = user['public_metrics']['tweet_count']
    if tweets is None:
        return 0
    else:
        return int(tweets)


def get_lists(user):
    lists = user['public_metrics']['listed_count']
    if lists is None:
        return 0
    else:
        return int(lists)


def get_number_count_in_screen_name(user):
    screen_name = get_screen_name(user)
    if screen_name is None:
        return 0
    else:
        numbers = re.findall(r'\d+', str(screen_name))
        return len(numbers)


def get_number_count_in_username(user):
    username = get_username(user)
    if username is None:
        return 0
    else:
        numbers = re.findall(r'\d+', str(username))
        return len(numbers)


def hashtags_count_in_username(user):
    username = user["username"]
    if username is None:
        return 0
    else:
        hashtags = re.findall(r'#\w', str(username))
    return len(hashtags)


def hashtags_count_in_description(user):
    if has_description(user):
        hashtags = re.findall(r'#\w', str(user['description']))
        return len(hashtags)
    else:
        return 0


def urls_count_in_description(user):
    if has_description(user):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(user["description"]))
        return len(urls)
    else:
        return 0
