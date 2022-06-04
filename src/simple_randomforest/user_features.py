import re


def get_user_id(user):
    return user['id']


def get_username(user):
    return user['username']


def get_screen_name(user):
    return user['name']


def has_description(user):
    des = user['description']
    if des is None or des == ' ':
        return False
    else:
        return True


def has_location(user):
    location = user['location']
    if location is None or location == ' ':
        return False
    else:
        return True


def has_url(user):
    url = user['url']
    if url is None or url == ' ':
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
        matchObj = re.search('bot', user['description'], flags=re.IGNORECASE)
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
        matchObj = re.search('bot', screen_name, flags=re.IGNORECASE)
        if matchObj:
            return True
        else:
            return False


def has_bot_word_in_username(user):
    username = get_username(user)
    if username is None or username == ' ':
        return False
    else:
        matchObj = re.search('bot', username, flags=re.IGNORECASE)
        if matchObj:
            return True
        else:
            return False


def get_screen_name_length(user):
    screen_name = get_screen_name(user)
    if screen_name is None:
        return 0
    else:
        return len(screen_name) - 1


def get_username_length(user):
    username = get_username(user)
    if username is None:
        return 0
    else:
        return len(username) - 1


def get_description_length(user):
    if has_description(user):
        return len(user['description']) - 1
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
    followers_count = user['public_metrics']['followers_count']
    following_count = user['public_metrics']['following_count']
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


# maybe r'\d'
def get_number_count_in_screen_name(user):
    screen_name = get_screen_name(user)
    if screen_name is None:
        return 0
    else:
        numbers = re.findall(r'\d+', screen_name)
        return len(numbers)


def get_number_count_in_username(user):
    username = get_username(user)
    if username is None:
        return 0
    else:
        numbers = re.findall(r'\d+', username)
        return len(numbers)


def hashtags_count_in_username(user):
    username = user["username"]
    if username is None:
        return 0
    else:
        hashtags = re.findall(r'#\w', username)
    return len(hashtags)


def hashtags_count_in_description(user):
    if has_description(user):
        hashtags = re.findall(r'#\w', user['description'])
        return len(hashtags)
    else:
        return 0


def urls_count_in_description(user):
    if has_description(user):
        urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+] |[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', user["description"])
        return len(urls)
    else:
        return 0
