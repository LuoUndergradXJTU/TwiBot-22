import re
import datetime
import calendar

def get_user_id(user):
    return user['id']


def get_longevity(user):
    create = user['created_at'].split(' ')
    month = list(calendar.month_abbr).index(create[1])
    day = int(create[2])
    year = int(create[5])
    create_time = datetime.datetime(year, month, day)
    collect_time = datetime.datetime(2020, 10, 1)
    longevity = (collect_time - create_time).days
    return longevity


def get_screen_name_length(user):
    screen_name = user['name']
    if screen_name is None:
        return 0
    else:
        return len(screen_name) - 1


def get_description_length(user):
    des = user['description']
    if des is None or des == ' ':
        return 0
    else:
        return len(des) - 1


def get_following(user):
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


def get_following_to_followers(user):
    followers_count = user['public_metrics']['followers_count']
    following_count = user['public_metrics']['following_count']
    if int(followers_count) == 0:
        return 0.0        
    else:
        return int(following_count) / int(followers_count)


def get_tweets(user):
    tweets = user['public_metrics']['tweet_count']
    if tweets is None:
        return 0
    else:
        return int(tweets)


def get_tweets_per_day(user):
    total = get_tweets(user)
    days = get_longevity(user)
    return total / days
